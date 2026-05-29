#include "pipeline/base_pipeline.h"

#include <filesystem>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/config.h"
#include "core/factory.h"
#include "transform/perspective_warper.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/visualization/draw_matches.h"

namespace fs = std::filesystem;

namespace ir {

bool BasePipeline::configure(const PipelineConfig& cfg) {
    // 1. 保存配置，并清空上一次可能残留的组件。
    _config = cfg;
    _extractor.reset();
    _matcher.reset();
    _filters.clear();
    _geometry.reset();
    _warper.reset();

    try {
        // 2. 根据各子 YAML 创建特征、匹配、过滤和几何估计组件。
        _extractor = Factory::createFeatureExtractor(Config::load(cfg.feature_path));
        _matcher   = Factory::createMatcher        (Config::load(cfg.matcher_path));
        for (const auto& fp : cfg.filter_paths) {
            _filters.push_back(Factory::createFilter(Config::load(fp)));
        }
        _geometry  = Factory::createGeometryEstimator(Config::load(cfg.geometry_path));
    } catch (const std::exception& e) {
        IR_LOG_ERROR("BasePipeline::configure failed: ", e.what());
        return false;
    }

    // 3. 创建图像变换器，用于在几何估计成功后生成配准图像。
    _warper = std::make_shared<PerspectiveWarper>();

    IR_LOG_INFO("BasePipeline configured: extractor=", _extractor->name(),
                ", matcher=",  _matcher->name(),
                ", filters=",  static_cast<int>(_filters.size()),
                ", geometry=", _geometry->name());
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
                     ", img2=", ctx.image2_path.string());
        return false;
    }

    ctx.feature_data.first.image  = cv::imread(ctx.image1_path.string(), cv::IMREAD_COLOR);
    ctx.feature_data.second.image = cv::imread(ctx.image2_path.string(), cv::IMREAD_COLOR);

    if (ctx.feature_data.first.image.empty() || ctx.feature_data.second.image.empty()) {
        IR_LOG_ERROR("loadImages: cv::imread failed.");
        return false;
    }

    cv::cvtColor(ctx.feature_data.first.image,  ctx.feature_data.first.gray,  cv::COLOR_BGR2GRAY);
    cv::cvtColor(ctx.feature_data.second.image, ctx.feature_data.second.gray, cv::COLOR_BGR2GRAY);

    IR_LOG_INFO("Loaded images: ",
                ctx.feature_data.first.image.cols, "x",
                ctx.feature_data.first.image.rows, " and ",
                ctx.feature_data.second.image.cols, "x",
                ctx.feature_data.second.image.rows);
    return true;
}

bool BasePipeline::runExtract(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_extract_ms);
    if (!_extractor) {
        IR_LOG_ERROR("runExtract: no extractor configured.");
        return false;
    }
    const bool ok = _extractor->extract(ctx);
    ctx.result.num_keypoints_first  = static_cast<int>(ctx.feature_data.first.keypoints.size());
    ctx.result.num_keypoints_second = static_cast<int>(ctx.feature_data.second.keypoints.size());
    return ok;
}

bool BasePipeline::runMatch(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_match_ms);
    if (!_matcher) {
        IR_LOG_ERROR("runMatch: no matcher configured.");
        return false;
    }
    const bool ok = _matcher->match(ctx);
    ctx.result.num_raw_matches = static_cast<int>(ctx.match_data.raw_knn.size());
    return ok;
}

bool BasePipeline::runFilters(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_filter_ms);
    auto& md = ctx.match_data;

    // 如果没有过滤器，或某些过滤器仅做开关控制未写入 filtered，
    // 默认将 raw_knn 的 top-1 结果提升为 filtered，保证后续几何阶段可继续运行。
    if (md.filtered.empty() && !md.raw_knn.empty()) {
        md.filtered.clear();
        md.filtered.reserve(md.raw_knn.size());
        for (const auto& nb : md.raw_knn) {
            if (!nb.empty()) md.filtered.push_back(nb.front());
        }
        IR_LOG_INFO("Seeded filtered matches from raw_knn top-1: ",
                    md.filtered.size(), " / ", md.raw_knn.size());
    }

    bool ok = true;
    for (const auto& f : _filters) {
        if (!f) continue;
        if (!f->apply(ctx)) {
            IR_LOG_WARN("Filter '", f->name(), "' returned false.");
            ok = false;
        }
    }
    ctx.result.num_filtered_matches = static_cast<int>(md.filtered.size());
    return ok;
}

bool BasePipeline::runGeometry(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_geometry_ms);
    if (!_geometry) {
        IR_LOG_ERROR("runGeometry: no estimator configured.");
        ctx.geometry_data.message = "no geometry estimator configured";
        return false;
    }
    const bool ok = _geometry->estimate(ctx);
    ctx.result.num_inliers   = ctx.geometry_data.num_inliers;
    ctx.result.inlier_ratio  = ctx.geometry_data.inlier_ratio;
    return ok;
}

bool BasePipeline::runWarp(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_warp_ms);
    if (!_config.warp || !_warper) return true;

    const auto t = ctx.geometry_data.type;
    if (t != GeometryType::HOMOGRAPHY &&
        t != GeometryType::AFFINE &&
        t != GeometryType::RIGID &&
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

bool BasePipeline::saveOutputs(RegistrationContext& ctx) {
    if (_config.output_dir.empty()) return true;
    const fs::path keypoints_dir = _config.output_dir / "keypoints";
    const fs::path matches_dir = _config.output_dir / "matches";
    const fs::path warped_dir  = _config.output_dir / "warped";
    std::error_code ec;
    fs::create_directories(keypoints_dir, ec);
    fs::create_directories(matches_dir, ec);
    fs::create_directories(warped_dir,  ec);

    const std::string stem =
        ctx.image1_path.stem().string() + "_" + ctx.image2_path.stem().string()
        + "_" + (_extractor ? _extractor->name() : std::string("UNK"))
        + "_" + (_geometry  ? toString(_geometry->type()) : std::string("UNK"));

    if (_config.draw_keypoints) {
        cv::Mat src_vis;
        cv::Mat dst_vis;
        cv::drawKeypoints(ctx.feature_data.first.image,
                          ctx.feature_data.first.keypoints,
                          src_vis,
                          cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::drawKeypoints(ctx.feature_data.second.image,
                          ctx.feature_data.second.keypoints,
                          dst_vis,
                          cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        if (!src_vis.empty()) {
            const fs::path out = keypoints_dir / (stem + "_source_keypoints.png");
            cv::imwrite(out.string(), src_vis);
            IR_LOG_INFO("Wrote source keypoints visualization: ", out.string());
        }
        if (!dst_vis.empty()) {
            const fs::path out = keypoints_dir / (stem + "_target_keypoints.png");
            cv::imwrite(out.string(), dst_vis);
            IR_LOG_INFO("Wrote target keypoints visualization: ", out.string());
        }
    }

    if (_config.draw_matches) {
        DrawMatches::Options opt;
        opt.draw_inliers_only = _config.draw_inliers_only;
        opt.max_matches       = _config.max_matches_drawn;
        cv::Mat vis = DrawMatches::render(ctx, opt);
        if (!vis.empty()) {
            const fs::path out = matches_dir / (stem + "_matches.png");
            cv::imwrite(out.string(), vis);
            IR_LOG_INFO("Wrote matches visualization: ", out.string());
        }
    }

    if (_config.warp && !ctx.warped_image.empty()) {
        const fs::path out = warped_dir / (stem + "_warped.png");
        cv::imwrite(out.string(), ctx.warped_image);
        IR_LOG_INFO("Wrote warped image: ", out.string());

        // 额外保存与目标图的混合图，便于快速检查配准效果。
        if (ctx.warped_image.size() == ctx.feature_data.second.image.size() &&
            ctx.warped_image.type() == ctx.feature_data.second.image.type()) {
            cv::Mat blend;
            cv::addWeighted(ctx.warped_image, 0.5,
                            ctx.feature_data.second.image, 0.5,
                            0.0, blend);
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

    // 1. 重置上下文，并写入当前配置指定的输入输出路径。
    ctx.reset();
    ctx.image1_path = _config.image1_path;
    ctx.image2_path = _config.image2_path;
    ctx.output_dir  = _config.output_dir;

    auto fail = [&](const std::string& msg) {
        ctx.result.success = false;
        ctx.result.message = msg;
        ctx.result.t_total_ms = total.elapsedMs();
        IR_LOG_ERROR("Pipeline failed: ", msg);
        return false;
    };

    // 2. 读取两张输入图像，并转换为后续特征提取需要的灰度图。
    if (!loadImages(ctx))    return fail("load failed");

    // 3. 提取特征点和描述子；SIFT/ORB 等具体算法在这里执行。
    if (!runExtract (ctx))   return fail("extract failed");

    // 4. 对两张图的描述子进行初始匹配。
    if (!runMatch   (ctx))   return fail("match failed");

    // 5. 按配置顺序执行匹配过滤器，例如 RatioTest 和 CrossCheck。
    if (!runFilters (ctx))   IR_LOG_WARN("Some filter stage reported a soft failure.");

    // 6. 基于过滤后的匹配估计几何模型，并标记内点。
    if (!runGeometry(ctx)) {
        const std::string detail = ctx.geometry_data.message.empty()
            ? std::string("geometry failed")
            : std::string("geometry failed: ") + ctx.geometry_data.message;
        return fail(detail);
    }

    // 7. 根据几何结果生成配准图像，并保存可视化输出。
    runWarp      (ctx);
    saveOutputs  (ctx);
    showWindows  (ctx);

    // 8. 写入成功状态和总耗时。
    ctx.result.success     = true;
    ctx.result.t_total_ms  = total.elapsedMs();
    ctx.result.message     = "OK";
    return true;
}

} // namespace ir

