#include "pipeline/base_pipeline.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/config.h"
#include "transform/perspective_warper.h"
#include "utils/logger.h"
#include "utils/timer.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

// TIFF 常见于高位深场景，优先走保留原始位深的读取分支。
bool hasTiffExtension(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext == ".tif" || ext == ".tiff";
}

// 在保留数值位深的前提下完成灰度化，为后续统一归一化预处理提供输入。
bool toGrayPreserveDepth(const cv::Mat& src, cv::Mat& gray) {
    if (src.channels() == 1) {
        gray = src;
        return true;
    }
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        return true;
    }
    if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
        return true;
    }
    return false;
}

// 将单通道图像压缩到 8 位动态范围，兼容多数 OpenCV 算子对输入类型的要求。
bool convertGrayTo8U(const cv::Mat& gray, cv::Mat& gray8) {
    if (gray.empty() || gray.channels() != 1) {
        return false;
    }
    if (gray.depth() == CV_8U) {
        gray8 = gray.clone();
        return true;
    }

    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(gray, &minVal, &maxVal);
    if (!std::isfinite(minVal) || !std::isfinite(maxVal) || maxVal <= minVal) {
        gray8 = cv::Mat::zeros(gray.size(), CV_8U);
        return true;
    }

    const double scale = 255.0 / (maxVal - minVal);
    gray.convertTo(gray8, CV_8U, scale, -minVal * scale);
    return true;
}

// 为配准方法准备一对同步图像：显示用 BGR 图与计算用 8 位灰度图。
bool loadImageForPipeline(const fs::path& path, cv::Mat& color, cv::Mat& gray) {
    color.release();
    gray.release();

    if (!hasTiffExtension(path)) {
        color = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (!color.empty()) {
            cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
            return true;
        }
    }

    cv::Mat raw = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (raw.empty()) {
        return false;
    }

    if (raw.depth() == CV_8U) {
        if (raw.channels() == 1) {
            gray = raw.clone();
            cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
            return true;
        }
        if (raw.channels() == 3) {
            color = raw.clone();
            cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
            return true;
        }
        if (raw.channels() == 4) {
            cv::cvtColor(raw, color, cv::COLOR_BGRA2BGR);
            cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
            return true;
        }
    }

    cv::Mat nativeGray;
    if (!toGrayPreserveDepth(raw, nativeGray) || !convertGrayTo8U(nativeGray, gray)) {
        return false;
    }

    cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
    if (raw.depth() != CV_8U) {
        IR_LOG_INFO("Loaded and normalized high-depth image: ",
                    path.string(),
                    " (depth=",
                    raw.depth(),
                    ", channels=",
                    raw.channels(),
                    ")");
    }
    return true;
}

} // namespace

bool BasePipeline::configure(const PipelineConfig& cfg) {
    _config = cfg;
    _warper.reset();
    resetStages();

    try {
        if (!configureStages(cfg)) {
            IR_LOG_ERROR(name(), "::configureStages returned false.");
            return false;
        }
    } catch (const std::exception& e) {
        IR_LOG_ERROR(name(), "::configure failed: ", e.what());
        return false;
    }

    _warper = std::make_shared<PerspectiveWarper>();
    IR_LOG_INFO(name(), " configured.");
    return true;
}

bool BasePipeline::loadImages(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_load_ms);

    if (ctx.image1_path.empty() || ctx.image2_path.empty()) {
        IR_LOG_ERROR("loadImages: one of the image paths is empty.");
        return false;
    }
    if (!fs::exists(ctx.image1_path) || !fs::exists(ctx.image2_path)) {
        IR_LOG_ERROR("loadImages: image not found. img1=",
                     ctx.image1_path.string(),
                     ", img2=",
                     ctx.image2_path.string());
        return false;
    }

    if (!loadImageForPipeline(
            ctx.image1_path, ctx.feature_data.first.image, ctx.feature_data.first.gray) ||
        !loadImageForPipeline(
            ctx.image2_path, ctx.feature_data.second.image, ctx.feature_data.second.gray)) {
        IR_LOG_ERROR("loadImages: cv::imread failed or image format is unsupported.");
        return false;
    }

    IR_LOG_INFO("Loaded images: ",
                ctx.feature_data.first.image.cols,
                "x",
                ctx.feature_data.first.image.rows,
                " and ",
                ctx.feature_data.second.image.cols,
                "x",
                ctx.feature_data.second.image.rows);
    return true;
}

bool BasePipeline::runWarp(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_warp_ms);

    if (!_config.warp || !_warper)
        return true;

    const auto t = ctx.geometry_data.type;
    if (t != GeometryType::HOMOGRAPHY && t != GeometryType::AFFINE && t != GeometryType::RIGID &&
        t != GeometryType::SIMILARITY) {
        IR_LOG_INFO("Warp skipped (", toString(t), " is not warpable).");
        return true;
    }
    if (!ctx.geometry_data.valid) {
        IR_LOG_WARN("Warp skipped: geometry estimation invalid.");
        return false;
    }
    return _warper->warp(ctx);
}

std::string BasePipeline::buildOutputStem(const RegistrationContext& ctx) const {
    return ctx.image1_path.stem().string() + "_" + ctx.image2_path.stem().string() + "_" + name();
}

bool BasePipeline::saveOutputs(RegistrationContext& ctx) {
    if (_config.output_dir.empty())
        return true;

    const fs::path warped_dir = _config.output_dir / "warped";
    std::error_code ec;
    fs::create_directories(warped_dir, ec);

    const std::string stem = buildOutputStem(ctx);
    if (_config.warp && !ctx.warped_image.empty()) {
        const fs::path out = warped_dir / (stem + "_warped.png");
        cv::imwrite(out.string(), ctx.warped_image);
        IR_LOG_INFO("Wrote warped image: ", out.string());

        if (ctx.warped_image.size() == ctx.feature_data.second.image.size() &&
            ctx.warped_image.type() == ctx.feature_data.second.image.type()) {
            cv::Mat blend;
            cv::addWeighted(ctx.warped_image, 0.5, ctx.feature_data.second.image, 0.5, 0.0, blend);
            const fs::path blend_out = warped_dir / (stem + "_blend.png");
            cv::imwrite(blend_out.string(), blend);
            IR_LOG_INFO("Wrote blend image: ", blend_out.string());
        }
    }

    return true;
}

bool BasePipeline::showWindows(RegistrationContext& ctx) {
    bool shown = false;

    if (_config.show_source_window && !ctx.feature_data.first.image.empty()) {
        cv::imshow("Source Image", ctx.feature_data.first.image);
        shown = true;
    }
    if (_config.show_target_window && !ctx.feature_data.second.image.empty()) {
        cv::imshow("Target Image", ctx.feature_data.second.image);
        shown = true;
    }
    if (_config.show_warped_window) {
        if (!ctx.warped_image.empty()) {
            cv::imshow("Warped Image", ctx.warped_image);
            shown = true;
        } else {
            IR_LOG_WARN("show_warped_window is enabled, but warped_image is empty.");
        }
    }

    if (shown) {
        const int wait = (_config.wait_key < 0) ? 0 : _config.wait_key;
        IR_LOG_INFO("Displaying visualization windows; waitKey=", wait);
        cv::waitKey(wait);
    }

    return true;
}

bool BasePipeline::run(RegistrationContext& ctx) {
    Timer total;

    ctx.reset();
    ctx.image1_path = _config.image1_path;
    ctx.image2_path = _config.image2_path;
    ctx.output_dir = _config.output_dir;

    auto fail = [&](const std::string& msg) {
        ctx.result.success = false;
        ctx.result.message = msg;
        ctx.result.t_total_ms = total.elapsedMs();
        IR_LOG_ERROR("Pipeline failed: ", msg);
        saveOutputs(ctx);
        return false;
    };

    if (!loadImages(ctx))
        return fail("load failed");

    if (!runExtraction(ctx))
        return fail("extract failed");

    if (!runAssociation(ctx))
        return fail("associate failed");

    if (!runEstimation(ctx)) {
        const std::string detail =
            ctx.geometry_data.message.empty()
                ? std::string("estimation failed")
                : std::string("estimation failed: ") + ctx.geometry_data.message;
        return fail(detail);
    }

    runWarp(ctx);
    saveOutputs(ctx);
    showWindows(ctx);

    ctx.result.success = true;
    ctx.result.t_total_ms = total.elapsedMs();
    ctx.result.message = "OK";
    return true;
}

} // namespace ir
