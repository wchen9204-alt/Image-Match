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

// 将图像转换为前景 mask，用于判断 warped source 与 target 是否真正重合。
bool buildForegroundMask(const cv::Mat& image, int thresholdValue, cv::Mat& mask) {
    mask.release();
    if (image.empty()) {
        return false;
    }

    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        return false;
    }

    cv::Mat gray8;
    if (!convertGrayTo8U(gray, gray8)) {
        return false;
    }

    cv::threshold(gray8,
                  mask,
                  static_cast<double>(std::clamp(thresholdValue, 0, 255)),
                  255.0,
                  cv::THRESH_BINARY);
    return true;
}

double computeMaskIou(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return -1.0;
    }

    cv::Mat intersectionMask;
    cv::Mat unionMask;
    cv::bitwise_and(a, b, intersectionMask);
    cv::bitwise_or(a, b, unionMask);

    const int unionCount = cv::countNonZero(unionMask);
    if (unionCount == 0) {
        return -1.0;
    }

    return static_cast<double>(cv::countNonZero(intersectionMask)) /
           static_cast<double>(unionCount);
}

} // namespace

bool BasePipeline::configure(const PipelineConfig& cfg) {
    // 1. 保存配置，并清空上一轮创建的阶段组件。
    _config = cfg;
    _warper.reset();
    resetStages();

    // 2. 委托子类创建提取、关联和估计阶段组件。
    try {
        if (!configureStages(cfg)) {
            IR_LOG_ERROR(name(), "::configureStages returned false.");
            return false;
        }
    } catch (const std::exception& e) {
        IR_LOG_ERROR(name(), "::configure failed: ", e.what());
        return false;
    }

    // 3. 创建通用 warp 组件；仿射族矩阵会在 warper 内部扩展为 3x3。
    _warper = std::make_shared<PerspectiveWarper>();
    IR_LOG_INFO(name(), " configured.");
    return true;
}

bool BasePipeline::loadImages(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_load_ms);

    // 1. 检查输入路径是否完整且文件存在。
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

    // 2. 读取图像，同时准备显示用 BGR 图和算法用 8 位灰度图。
    if (!loadImageForPipeline(ctx.image1_path, ctx.images.first, ctx.images.first_gray) ||
        !loadImageForPipeline(ctx.image2_path, ctx.images.second, ctx.images.second_gray)) {
        IR_LOG_ERROR("loadImages: cv::imread failed or image format is unsupported.");
        return false;
    }

    // 3. 记录输入尺寸，便于排查 warp 和 blend 的画布大小。
    IR_LOG_INFO("Loaded images: ",
                ctx.images.first.cols,
                "x",
                ctx.images.first.rows,
                " and ",
                ctx.images.second.cols,
                "x",
                ctx.images.second.rows);
    return true;
}

bool BasePipeline::runWarp(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_warp_ms);

    // 1. 若配置关闭 warp 或没有 warper，直接跳过。
    if (!_config.warp || !_warper) {
        return true;
    }

    // 2. 检查 runEstimation 是否写入了可用于图像重采样的几何模型。
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

    // 3. 执行 source -> target 画布的图像变换，结果写入 ctx.warped_image。
    return _warper->warp(ctx);
}

bool BasePipeline::validateWarpQuality(RegistrationContext& ctx) {
    ctx.result.warp_overlap_iou = -1.0;
    if (!_config.validate_warp_overlap) {
        return true;
    }

    if (!_config.warp || ctx.warped_image.empty()) {
        ctx.result.message = "warp overlap validation failed: warped image is empty";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }
    if (ctx.images.second.empty() || ctx.warped_image.size() != ctx.images.second.size()) {
        ctx.result.message =
            "warp overlap validation failed: warped image and target have different sizes";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    cv::Mat warpedMask;
    cv::Mat targetMask;
    const int thresholdValue = _config.warp_overlap_foreground_threshold;
    if (!buildForegroundMask(ctx.warped_image, thresholdValue, warpedMask) ||
        !buildForegroundMask(ctx.images.second, thresholdValue, targetMask)) {
        ctx.result.message = "warp overlap validation failed: cannot build foreground masks";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    const double iou = computeMaskIou(warpedMask, targetMask);
    ctx.result.warp_overlap_iou = iou;
    if (iou < 0.0) {
        ctx.result.message = "warp overlap validation failed: empty foreground union";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    IR_LOG_INFO("Warp overlap IoU=", iou, ", min=", _config.min_warp_overlap_iou);
    if (iou < _config.min_warp_overlap_iou) {
        ctx.result.message = "warp overlap IoU below threshold: " + std::to_string(iou) +
                             " < " + std::to_string(_config.min_warp_overlap_iou);
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    return true;
}

std::string BasePipeline::buildOutputStem(const RegistrationContext& ctx) const {
    return ctx.image1_path.stem().string() + "_" + ctx.image2_path.stem().string() + "_" + name();
}

bool BasePipeline::saveOutputs(RegistrationContext& ctx) {
    if (_config.output_dir.empty()) {
        return true;
    }

    // 1. 创建通用输出目录。
    const fs::path originals_dir = _config.output_dir / "originals";
    const fs::path warped_dir = _config.output_dir / "warped";
    const fs::path blend_dir = _config.output_dir / "blend";
    std::error_code ec;
    fs::create_directories(originals_dir, ec);
    fs::create_directories(warped_dir, ec);
    fs::create_directories(blend_dir, ec);

    const std::string stem = buildOutputStem(ctx);
    const std::string sampleStem = ctx.image1_path.stem().string() + "_" +
                                   ctx.image2_path.stem().string();

    // 2. 保存原始 source / target，便于和 warped / blend 对照。
    if (!ctx.images.first.empty()) {
        const fs::path out = originals_dir / (sampleStem + "_source_original.png");
        if (cv::imwrite(out.string(), ctx.images.first)) {
            IR_LOG_INFO("Wrote source original image: ", out.string());
        } else {
            IR_LOG_WARN("Failed to write source original image: ", out.string());
        }
    }
    if (!ctx.images.second.empty()) {
        const fs::path out = originals_dir / (sampleStem + "_target_original.png");
        if (cv::imwrite(out.string(), ctx.images.second)) {
            IR_LOG_INFO("Wrote target original image: ", out.string());
        } else {
            IR_LOG_WARN("Failed to write target original image: ", out.string());
        }
    }

    // 3. 保存 warped source，并和 target 按同尺寸画布生成 blend。
    if (_config.warp && !ctx.warped_image.empty()) {
        const fs::path out = warped_dir / (stem + "_warped.png");
        cv::imwrite(out.string(), ctx.warped_image);
        IR_LOG_INFO("Wrote warped image: ", out.string());

        if (ctx.warped_image.size() == ctx.images.second.size() &&
            ctx.warped_image.type() == ctx.images.second.type()) {
            cv::Mat blend;
            cv::addWeighted(ctx.warped_image, 0.5, ctx.images.second, 0.5, 0.0, blend);
            const fs::path blend_out = blend_dir / (stem + "_blend.png");
            cv::imwrite(blend_out.string(), blend);
            IR_LOG_INFO("Wrote blend image: ", blend_out.string());
        }
    }

    return true;
}

bool BasePipeline::showWindows(RegistrationContext& ctx) {
    bool shown = false;

    // 1. 按配置显示 source 窗口。
    if (_config.show_source_window && !ctx.images.first.empty()) {
        cv::imshow("Source Image", ctx.images.first);
        shown = true;
    }
    // 2. 按配置显示 target 窗口。
    if (_config.show_target_window && !ctx.images.second.empty()) {
        cv::imshow("Target Image", ctx.images.second);
        shown = true;
    }
    // 3. 按配置显示 warped source 窗口。
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

    // 1. 初始化本次运行上下文，并写入输入输出路径。
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

    // 2. 依次执行公共流程：读图、提取、关联、估计、warp 和输出。
    if (!loadImages(ctx)) {
        return fail("load failed");
    }

    if (!runExtraction(ctx)) {
        return fail("extract failed");
    }

    if (!runAssociation(ctx)) {
        std::string detail = "associate failed";
        if (!ctx.structure_match_data.message.empty()) {
            detail += ": " + ctx.structure_match_data.message;
        }
        return fail(detail);
    }

    if (!runEstimation(ctx)) {
        const std::string detail =
            ctx.geometry_data.message.empty()
                ? std::string("estimation failed")
                : std::string("estimation failed: ") + ctx.geometry_data.message;
        return fail(detail);
    }

    runWarp(ctx);
    if (!validateWarpQuality(ctx)) {
        return fail(ctx.result.message.empty() ? "warp validation failed" : ctx.result.message);
    }
    saveOutputs(ctx);
    // 3. 所有阶段完成后记录总耗时和成功状态。
    ctx.result.success = true;
    ctx.result.t_total_ms = total.elapsedMs();
    ctx.result.message = "OK";
    return true;
}

} // namespace ir
