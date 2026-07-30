#include "pipeline/direct_pipeline.h"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "core/config.h"
#include "core/factory.h"
#include "pipeline/base_pipeline_helpers.h"
#include "pipeline/direct_pipeline_helpers.h"
#include "utils/logger.h"
#include "utils/timer.h"

namespace fs = std::filesystem;

namespace ir {

void DirectPipeline::resetStages() {
    _aligner.reset();
    if (_featureInitializer) {
        _featureInitializer->reset();
    }
    _featureInitializer.reset();
}

bool DirectPipeline::configureStages(const PipelineConfig& cfg) {
    if (cfg.direct_path.empty()) {
        IR_LOG_ERROR("DirectPipeline: missing direct config path.");
        return false;
    }

    _aligner = Factory::createDirectAligner(Config::load(cfg.direct_path));
    if (cfg.feature_initializer.enabled) {
        _featureInitializer = std::make_unique<DirectFeatureInitializer>();
        if (!_featureInitializer->configure(cfg)) {
            return false;
        }
    }

    IR_LOG_INFO("DirectPipeline stages configured: aligner=", _aligner->name());
    return true;
}

bool DirectPipeline::runExtraction(RegistrationContext& ctx) {
    ctx.result.num_keypoints_first = 0;
    ctx.result.num_keypoints_second = 0;
    return true;
}

bool DirectPipeline::runAssociation(RegistrationContext& ctx) {
    ctx.result.num_raw_matches = 0;
    ctx.result.num_filtered_matches = 0;
    return true;
}

bool DirectPipeline::runEstimation(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_geometry_ms);

    /// 先确认当前 pipeline 已经根据配置创建出直接法对齐器。
    if (!_aligner) {
        ctx.geometry_data.message = "no direct aligner configured";
        IR_LOG_ERROR("DirectPipeline::runEstimation: no direct aligner configured.");
        return false;
    }

    /// 若启用了点特征初始值，则先执行它；是否注入直接法由 seed_mode 决定。
    if (_featureInitializer && _featureInitializer->enabled()) {
        _featureInitializer->run(ctx);
        if (ctx.feature_initializer_data.accepted) {
            IR_LOG_INFO("DirectPipeline will use accepted feature initializer: ",
                        ctx.feature_initializer_data.method);
        } else if (ctx.feature_initializer_data.seed_available) {
            IR_LOG_INFO("DirectPipeline will use rejected feature initializer as seed: ",
                        ctx.feature_initializer_data.method,
                        " (",
                        ctx.feature_initializer_data.message,
                        ")");
        } else {
            IR_LOG_INFO("DirectPipeline skipped feature initializer: ",
                        ctx.feature_initializer_data.message);
        }
    }

    /// 对不会自行消费初始值的直接法，先把 source 预 warp 到初始位姿，再让算法估计残差。
    cv::Mat initializerMatrix;
    cv::Mat originalSourceColor;
    cv::Mat originalSourceGray;
    bool usedGenericPrewarp = false;
    if (!direct_pipeline_helpers::applyFeatureInitializerPrewarp(ctx,
                                                                 _aligner->name(),
                                                                 initializerMatrix,
                                                                 originalSourceColor,
                                                                 originalSourceGray,
                                                                 usedGenericPrewarp)) {
        ctx.geometry_data.message = "failed to apply feature initializer prewarp";
        IR_LOG_WARN("DirectPipeline::runEstimation: ", ctx.geometry_data.message);
        return false;
    }

    /// 执行具体直接法，并在需要时把“初始值 + 残差”合成为最终 source -> target 变换。
    ctx.correspondence_source = "DIRECT";
    const bool ok = _aligner->align(ctx);
    if (usedGenericPrewarp) {
        ctx.images.first = std::move(originalSourceColor);
        ctx.images.first_gray = std::move(originalSourceGray);
        if (ok &&
            !direct_pipeline_helpers::mergeFeatureInitializerAndDirectResult(ctx,
                                                                             initializerMatrix)) {
            ctx.geometry_data.message = "failed to compose feature initializer with direct result";
            ctx.direct_data.message = ctx.geometry_data.message;
            IR_LOG_WARN("DirectPipeline::runEstimation: ", ctx.geometry_data.message);
            direct_pipeline_helpers::syncFeatureInitializerDiagnostics(ctx);
            return false;
        }
    }
    direct_pipeline_helpers::syncFeatureInitializerDiagnostics(ctx);

    /// 直接法没有复用点特征法的内点语义，这里只同步通用摘要字段给 CSV/JSON/日志复用。
    const int pairCount = static_cast<int>(std::max(
        ctx.direct_data.matches.size(),
        std::min(ctx.direct_data.points1.size(), ctx.direct_data.points2.size())));
    ctx.result.num_raw_matches = pairCount;
    ctx.result.num_filtered_matches = pairCount;
    ctx.result.num_inliers = 0;
    ctx.result.direct_confidence = ctx.direct_data.valid ? ctx.direct_data.score : -1.0;
    ctx.result.inlier_ratio = ctx.direct_data.valid ? ctx.direct_data.score
                                                    : ctx.geometry_data.inlier_ratio;
    return ok;
}

std::string DirectPipeline::buildOutputStem(const RegistrationContext& ctx) const {
    return ctx.image1_path.stem().string() + "_" + ctx.image2_path.stem().string() + "_" +
           (_aligner ? _aligner->name() : std::string("DIRECT"));
}

bool DirectPipeline::saveOutputs(RegistrationContext& ctx) {
    if (!_config.save_visuals || ctx.output_dir.empty()) {
        return true;
    }

    const fs::path directDir = ctx.output_dir / "direct";
    const std::string stem = buildOutputStem(ctx);
    if (_aligner) {
        direct_pipeline_helpers::removeStaleDirectVisualization(directDir / (stem + "_matches.png"));
        direct_pipeline_helpers::removeStaleDirectVisualization(directDir / (stem + "_flow.png"));
        direct_pipeline_helpers::removeStaleDirectVisualization(directDir / (stem + "_warp_diff.png"));
    }
    direct_pipeline_helpers::removeStaleDirectVisualization(
        ctx.output_dir / "false_color_overlay" / (stem + "_initializer_false_color_overlay.png"));
    direct_pipeline_helpers::removeStaleDirectVisualization(
        ctx.output_dir / "final_false_color_overlay" / (stem + "_final_false_color_overlay.png"));

    if (!direct_pipeline_helpers::applyFinalSelectedWarpedImage(ctx)) {
        IR_LOG_WARN("DirectPipeline failed to build the final selected warped image.");
        return false;
    }

    const bool ok = BasePipeline::saveOutputs(ctx);

    cv::Mat initializerWarped;
    cv::Mat initializerOverlay;
    if (_config.save_false_color_overlay &&
        direct_pipeline_helpers::buildInitializerWarpedSource(ctx, initializerWarped) &&
        base_pipeline_helpers::buildFalseColorOverlay(initializerWarped,
                                                      ctx.images.second,
                                                      _config.false_color_foreground_threshold,
                                                      initializerOverlay)) {
        const fs::path overlayDir = ctx.output_dir / "false_color_overlay";
        std::error_code ec;
        fs::create_directories(overlayDir, ec);
        const fs::path out = overlayDir / (stem + "_initializer_false_color_overlay.png");
        if (cv::imwrite(out.string(), initializerOverlay)) {
            IR_LOG_INFO("Wrote feature initializer false-color overlay image: ", out.string());
        } else {
            IR_LOG_WARN("Failed to write feature initializer false-color overlay image: ",
                        out.string());
        }
    }

    cv::Mat finalOverlay;
    if (_config.save_false_color_overlay &&
        direct_pipeline_helpers::buildFinalSelectedFalseColorOverlay(
            ctx,
            _config.false_color_foreground_threshold,
            finalOverlay)) {
        const fs::path finalOverlayDir = ctx.output_dir / "final_false_color_overlay";
        std::error_code ec;
        fs::create_directories(finalOverlayDir, ec);
        const fs::path out = finalOverlayDir / (stem + "_final_false_color_overlay.png");
        if (cv::imwrite(out.string(), finalOverlay)) {
            IR_LOG_INFO("Wrote final selected false-color overlay image: ", out.string());
        } else {
            IR_LOG_WARN("Failed to write final selected false-color overlay image: ",
                        out.string());
        }
    }

    return ok;
}

} // namespace ir
