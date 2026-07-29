#include "pipeline/base_pipeline.h"

#include <algorithm>
#include <filesystem>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/config.h"
#include "data/correspondence_view.h"
#include "pipeline/base_pipeline_helpers.h"
#include "evaluator/quality/warp_quality_evaluator.h"
#include "transform/affine_warper.h"
#include "transform/perspective_warper.h"
#include "utils/logger.h"
#include "utils/timer.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

/// 将一次 warp 质量评估结果写回 RegistrationResult，供终端摘要、JSON 和 CSV 统一复用。
void syncWarpQualityToResult(RegistrationResult& result,
                             const warp_quality::WarpQualityResult& quality) {
    result.warp_overlap_containment = quality.overlap_containment;
    result.warp_source_coverage = quality.source_coverage;
    result.warp_target_coverage = quality.target_coverage;
    result.warp_edge_alignment_iou = quality.edge_alignment_iou;
    result.warp_photometric_error = quality.photometric_error;
}

/// 比较两个都已成功的 warp 质量结果。
/// 两者已通过各自的质量门槛后，将 containment 和 NMAD 归一化为同向分数，
/// 用综合分选择最终结果；平分时保留直接法结果，避免无意义的来源切换。
bool preferInitializerResult(const PipelineConfig& cfg,
                             const warp_quality::WarpQualityResult& directQuality,
                             const FeatureInitializerData& initializer) {
    const double kEps = 1e-6;

    const auto& validation = cfg.warp_quality;
    const double minContainment = validation.overlap.min_containment;
    const double maxPhotometricError = validation.photometric.max_nmad;
    constexpr double kContainmentWeight = 0.35;
    constexpr double kPhotometricWeight = 0.65;

    // 两项门槛未启用或配置非法时，没有可比较的共同评分尺度，保留直接法结果。
    if (!validation.overlap.containment_enabled || !validation.photometric.enabled ||
        minContainment < 0.0 || minContainment >= 1.0 || maxPhotometricError <= 0.0) {
        return false;
    }

    const auto qualityScore = [&](const double containment, const double nmad) {
        // containment 越接近 1 越好；低于验收线的值归零，验收后只比较剩余提升空间。
        const double containmentScore =
            std::clamp((containment - minContainment) / (1.0 - minContainment), 0.0, 1.0);
        // NMAD 越小越好；达到上限时为零，零误差时为满分。
        const double photometricScore =
            std::clamp(1.0 - nmad / maxPhotometricError, 0.0, 1.0);
        return kContainmentWeight * containmentScore +
               kPhotometricWeight * photometricScore;
    };

    const double initializerScore = qualityScore(initializer.warp_overlap_containment,
                                                 initializer.warp_photometric_error);
    const double directScore = qualityScore(directQuality.overlap_containment,
                                            directQuality.photometric_error);
    return initializerScore > directScore + kEps;
}

/// 将 evaluator 中和通用结果字段同义的指标同步回 RegistrationResult。
void syncEvaluationMetricsToResult(RegistrationContext& ctx) {
    for (const auto& metric : ctx.evaluation.metrics) {
        if (!metric.valid) {
            continue;
        }
        if (metric.name == "REPROJECTION_ERROR") {
            ctx.result.mean_reproj_error = metric.value;
        } else if (metric.name == "INLIER_RATIO") {
            ctx.result.inlier_ratio = metric.value;
        }
    }
}

/// 在 DIRECT_ONLY 模式下，只认 direct 最终结果；
/// 在 BEST_OF_DIRECT_AND_INITIALIZER 模式下，按照 direct / initializer 各自已有成功状态选最终结果。
bool resolveDirectFinalValidationReference(const PipelineConfig& cfg,
                                           RegistrationContext& ctx,
                                           const bool directOk,
                                           const warp_quality::WarpQualityResult& directQuality) {
    if (cfg.methodFamily() != MethodFamily::DIRECT ||
        cfg.feature_initializer.final_validation_reference ==
            DirectValidationReferenceMode::DIRECT_ONLY) {
        ctx.result.final_validation_source = "DIRECT";
        if (directOk) {
            syncWarpQualityToResult(ctx.result, directQuality);
        } else {
            ctx.result.message = directQuality.message;
        }
        return directOk;
    }

    const bool initializerOk = ctx.feature_initializer_data.accepted;
    ctx.result.feature_initializer_used = false;

    if (!directOk && !initializerOk) {
        ctx.result.final_validation_source.clear();
        ctx.result.message = directQuality.message;
        return false;
    }

    if (initializerOk && !directOk) {
        ctx.result.final_validation_source = "INITIALIZER";
        ctx.result.feature_initializer_used = true;
        ctx.result.warp_overlap_containment = ctx.feature_initializer_data.warp_overlap_containment;
        ctx.result.warp_source_coverage = ctx.feature_initializer_data.warp_source_coverage;
        ctx.result.warp_target_coverage = ctx.feature_initializer_data.warp_target_coverage;
        ctx.result.warp_edge_alignment_iou =
            ctx.feature_initializer_data.warp_edge_alignment_iou;
        ctx.result.warp_photometric_error =
            ctx.feature_initializer_data.warp_photometric_error;
        ctx.result.message = "OK";
        ctx.direct_data.addDiagnostic("final_validation_used_initializer",
                                      "final validation used initializer",
                                      1.0);
        return true;
    }

    if (!initializerOk && directOk) {
        ctx.result.final_validation_source = "DIRECT";
        syncWarpQualityToResult(ctx.result, directQuality);
        return true;
    }

    if (preferInitializerResult(cfg, directQuality, ctx.feature_initializer_data)) {
        ctx.result.final_validation_source = "INITIALIZER";
        ctx.result.feature_initializer_used = true;
        ctx.result.warp_overlap_containment = ctx.feature_initializer_data.warp_overlap_containment;
        ctx.result.warp_source_coverage = ctx.feature_initializer_data.warp_source_coverage;
        ctx.result.warp_target_coverage = ctx.feature_initializer_data.warp_target_coverage;
        ctx.result.warp_edge_alignment_iou =
            ctx.feature_initializer_data.warp_edge_alignment_iou;
        ctx.result.warp_photometric_error =
            ctx.feature_initializer_data.warp_photometric_error;
        ctx.result.message = "OK";
        ctx.direct_data.addDiagnostic("final_validation_used_initializer",
                                      "final validation used initializer",
                                      1.0);
        return true;
    }

    ctx.result.final_validation_source = "DIRECT";
    syncWarpQualityToResult(ctx.result, directQuality);
    return true;
}

} // namespace

bool BasePipeline::configure(const PipelineConfig& cfg) {
    // 1. 保存配置，并清空上一轮创建的阶段组件。
    _config = cfg;
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

    // 3. 加载评测指标（可选）。
    _evaluator.clear();
    if (!cfg.evaluator_path.empty()) {
        _evaluator.loadFromYaml(cfg.evaluator_path);
    }

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
    if (!base_pipeline_helpers::loadImageForPipeline(ctx.image1_path,
                                                     ctx.images.first,
                                                     ctx.images.first_gray) ||
        !base_pipeline_helpers::loadImageForPipeline(ctx.image2_path,
                                                     ctx.images.second,
                                                     ctx.images.second_gray)) {
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

    // 1. 若配置关闭 warp，直接跳过。
    if (!_config.warp) {
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

    // 3. 根据当前几何模型在运行时选择 warper，避免 configure 阶段把 warper 预先固定死。
    if (t == GeometryType::HOMOGRAPHY) {
        PerspectiveWarper perspectiveWarper;
        return perspectiveWarper.warp(ctx);
    }

    AffineWarper affineWarper;
    return affineWarper.warp(ctx);
}

bool BasePipeline::validateRegistrationQuality(RegistrationContext& ctx) {
    // 先重置可选验证结果。
    ctx.result.structure_overlap_iou = -1.0;

    // 1. 先做方法特有判定，只检查当前方法族真正拥有的数据和信号。
    if (!validateMethodSpecificQuality(ctx)) {
        return false;
    }
    // 2. 再做所有方法族共享的最终图像级判定，统一 success 口径。
    if (!validateSharedFinalQuality(ctx)) {
        return false;
    }
    return true;
}

bool BasePipeline::validateMethodSpecificQuality(RegistrationContext& ctx) {
    // 方法特征判定：检查当前方法族自身产物是否足够进入后续配准。
    if (!validateMethodFeatureQuality(ctx)) {
        return false;
    }

    // 匹配/几何质量判定：检查内点、内点率、重投影误差等匹配质量信号。
    if (!validateMatchQuality(ctx)) {
        return false;
    }

    // 直接法专属 confidence / response 判定，仅 direct 方法族启用时生效。
    if (!validateDirectQuality(ctx)) {
        return false;
    }

    // 结构法专属响应图重叠判定，仅 structure 方法族启用时生效。
    if (!validateStructureOverlap(ctx)) {
        return false;
    }

    return true;
}

bool BasePipeline::validateMethodFeatureQuality(RegistrationContext& ctx) {
    if (!_config.validate_method_quality) {
        return true;
    }

    auto handleViolation = [&](const std::string& message) {
        if (_config.fail_on_method_quality) {
            ctx.result.message = "method quality validation failed: " + message;
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
        IR_LOG_WARN("Method quality warning: ", message);
        return true;
    };

    const MethodFamily family = _config.methodFamily();
    if (family == MethodFamily::KEYPOINT || family == MethodFamily::LEARNING) {
        const int minKeypoints =
            std::min(ctx.result.num_keypoints_first, ctx.result.num_keypoints_second);
        IR_LOG_INFO("Method quality [keypoints]: first=",
                    ctx.result.num_keypoints_first,
                    ", second=",
                    ctx.result.num_keypoints_second,
                    ", min_required=",
                    _config.min_method_keypoints);
        if (_config.min_method_keypoints > 0 &&
            minKeypoints < _config.min_method_keypoints) {
            return handleViolation("keypoints " + std::to_string(minKeypoints) + " < " +
                                   std::to_string(_config.min_method_keypoints));
        }
        return true;
    }

    if (family == MethodFamily::STRUCTURE) {
        const int minStructures =
            std::min(ctx.result.num_structures_first, ctx.result.num_structures_second);
        const int structureMatches = ctx.result.num_filtered_matches;
        IR_LOG_INFO("Method quality [structures]: first=",
                    ctx.result.num_structures_first,
                    ", second=",
                    ctx.result.num_structures_second,
                    ", matches=",
                    structureMatches,
                    ", min_structures=",
                    _config.min_method_structures,
                    ", min_matches=",
                    _config.min_method_structure_matches);
        if (_config.min_method_structures > 0 &&
            minStructures < _config.min_method_structures) {
            return handleViolation("structures " + std::to_string(minStructures) + " < " +
                                   std::to_string(_config.min_method_structures));
        }
        if (_config.min_method_structure_matches > 0 &&
            structureMatches < _config.min_method_structure_matches) {
            return handleViolation("structure matches " + std::to_string(structureMatches) +
                                   " < " +
                                   std::to_string(_config.min_method_structure_matches));
        }
    }

    return true;
}

bool BasePipeline::validateSharedFinalQuality(RegistrationContext& ctx) {
    // 当前共享最终判定统一落在 warp 质量上，覆盖几何重叠、光度一致性和边缘对齐。
    return validateWarpQuality(ctx);
}

bool BasePipeline::validateMatchQuality(RegistrationContext& ctx) {
    // 只有显式启用了 match_quality 的 pipeline 才做该项检查。
    if (!_config.validate_match_quality) {
        return true;
    }

    // 只有拥有显式匹配/对应点质量判定的方法族进入该逻辑。
    if (_config.methodFamily() != MethodFamily::KEYPOINT &&
        _config.methodFamily() != MethodFamily::LEARNING &&
        _config.methodFamily() != MethodFamily::DIRECT) {
        return true;
    }

    auto handleViolation = [&](const std::string& message) {
        if (_config.fail_on_match_quality) {
            ctx.result.message = "match quality validation failed: " + message;
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
        IR_LOG_WARN("Match quality warning: ", message);
        return true;
    };

    int effectiveInliers = ctx.result.num_inliers;
    double effectiveRatio = ctx.result.inlier_ratio;
    double effectiveReprojError = ctx.result.mean_reproj_error;
    std::vector<cv::Point2f> sourceCoveragePoints;
    std::vector<cv::Point2f> targetCoveragePoints;
    bool hasDiscreteCorrespondences = true;

    if (_config.methodFamily() == MethodFamily::DIRECT) {
        const size_t pointCount =
            std::min(ctx.direct_data.points1.size(), ctx.direct_data.points2.size());
        if (pointCount == 0) {
            IR_LOG_INFO("Match quality skipped for direct method without point correspondences.");
            ctx.result.inlier_spatial_coverage = -1.0;
            return true;
        }

        hasDiscreteCorrespondences = true;
        const bool hasMask = !ctx.direct_data.inlier_mask.empty();
        sourceCoveragePoints.reserve(pointCount);
        targetCoveragePoints.reserve(pointCount);
        effectiveInliers = 0;
        for (size_t i = 0; i < pointCount; ++i) {
            const bool isInlier = !hasMask || (i < ctx.direct_data.inlier_mask.size() &&
                                               ctx.direct_data.inlier_mask[i] != 0);
            if (!isInlier) {
                continue;
            }
            sourceCoveragePoints.push_back(ctx.direct_data.points1[i]);
            targetCoveragePoints.push_back(ctx.direct_data.points2[i]);
            ++effectiveInliers;
        }
        effectiveRatio =
            pointCount == 0 ? 0.0
                            : static_cast<double>(effectiveInliers) /
                                  static_cast<double>(pointCount);
        effectiveReprojError = -1.0;
    }

    IR_LOG_INFO("Match quality: inliers=",
                effectiveInliers,
                ", ratio=",
                effectiveRatio,
                ", reproj=",
                effectiveReprojError);

    // 条件1：最少内点数。是否判失败由 fail_on_violation 控制。
    if (_config.min_match_inliers > 0 && effectiveInliers < _config.min_match_inliers) {
        const std::string message = "inliers " + std::to_string(effectiveInliers) + " < " +
                                    std::to_string(_config.min_match_inliers);
        if (!handleViolation(message)) {
            return false;
        }
    }

    // 条件2：最低内点率。是否判失败由 fail_on_violation 控制。
    if (_config.min_match_inlier_ratio >= 0.0 &&
        effectiveRatio < _config.min_match_inlier_ratio) {
        const std::string message = "inlier ratio " + std::to_string(effectiveRatio) +
                                    " < " + std::to_string(_config.min_match_inlier_ratio);
        if (!handleViolation(message)) {
            return false;
        }
    }

    // 条件3：最大重投影误差。是否判失败由 fail_on_violation 控制。
    if (_config.max_match_reproj_error >= 0.0 &&
        effectiveReprojError >= 0.0 &&
        effectiveReprojError > _config.max_match_reproj_error) {
        const std::string message = "reprojection error " +
                                    std::to_string(effectiveReprojError) + " > " +
                                    std::to_string(_config.max_match_reproj_error);
        if (!handleViolation(message)) {
            return false;
        }
    }

    // 条件4：计算最终内点在前景中的空间覆盖率，写入结果用于后续综合判断。
    // 这里先不提前判失败，避免把 warp 几何和光度都已经对齐的局部包含场景误杀。
    if (_config.min_inlier_spatial_coverage >= 0.0) {
        cv::Mat sourceMask;
        cv::Mat targetMask;
        const int thresholdValue = _config.warp_quality.overlap.foreground_threshold;
        if (!base_pipeline_helpers::buildForegroundMask(ctx.images.first,
                                                        thresholdValue,
                                                        sourceMask) ||
            !base_pipeline_helpers::buildForegroundMask(ctx.images.second,
                                                        thresholdValue,
                                                        targetMask)) {
            if (!handleViolation("cannot build masks for inlier spatial coverage")) {
                return false;
            }
        } else {
            double sourceSpatialCoverage = -1.0;
            double targetSpatialCoverage = -1.0;
            double spatialCoverage = -1.0;
            if (_config.methodFamily() == MethodFamily::DIRECT && hasDiscreteCorrespondences) {
                spatialCoverage = base_pipeline_helpers::computePointSpatialCoverage(
                    sourceCoveragePoints,
                    targetCoveragePoints,
                    sourceMask,
                    targetMask,
                    sourceSpatialCoverage,
                    targetSpatialCoverage);
            } else {
                spatialCoverage = base_pipeline_helpers::computeInlierSpatialCoverage(
                    ctx.keypoint_data.first.keypoints,
                    ctx.keypoint_data.second.keypoints,
                    ctx.keypoint_match_data.inlier_matches,
                    sourceMask,
                    targetMask,
                    sourceSpatialCoverage,
                    targetSpatialCoverage);
            }
            ctx.result.inlier_spatial_coverage = spatialCoverage;
            IR_LOG_INFO("Inlier spatial coverage=",
                        spatialCoverage,
                        ", source=",
                        sourceSpatialCoverage,
                        ", target=",
                        targetSpatialCoverage,
                        ", min=",
                        _config.min_inlier_spatial_coverage);
            if (spatialCoverage < _config.min_inlier_spatial_coverage) {
                IR_LOG_WARN("Match quality warning: inlier spatial coverage ",
                            spatialCoverage,
                            " < ",
                            _config.min_inlier_spatial_coverage,
                            " (recorded only; final acceptance is decided by warp validation)");
            }
        }
    }

    return true;
}

bool BasePipeline::validateDirectQuality(RegistrationContext& ctx) {
    // 只有直接法且启用了 direct_quality 配置时，才进入该分支。
    if (_config.methodFamily() != MethodFamily::DIRECT || !_config.validate_direct_quality) {
        return true;
    }

    auto handleViolation = [&](const std::string& message) {
        if (_config.fail_on_direct_quality) {
            ctx.result.message = "direct quality validation failed: " + message;
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
        IR_LOG_WARN("Direct quality warning: ", message);
        return true;
    };

    const double confidence = ctx.direct_data.score;
    IR_LOG_INFO("Direct quality: confidence=", confidence);

    if (_config.min_direct_confidence >= 0.0 &&
        confidence < _config.min_direct_confidence) {
        const std::string message = "confidence " + std::to_string(confidence) + " < " +
                                    std::to_string(_config.min_direct_confidence);
        if (!handleViolation(message)) {
            return false;
        }
    }

    return true;
}

bool BasePipeline::validateStructureOverlap(RegistrationContext& ctx) {
    // 只有结构法且显式启用了 structure_overlap 的 pipeline 才进入结构重叠验证。
    if (_config.methodFamily() != MethodFamily::STRUCTURE || !_config.validate_structure_overlap) {
        return true;
    }

    // 条件1：必须有结构响应图。
    if (ctx.structure_data.first.response.empty() || ctx.structure_data.second.response.empty()) {
        ctx.result.message = "structure overlap validation failed: missing structure response";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    cv::Mat sourceMask;
    cv::Mat targetMask;
    const int thresholdValue = _config.structure_overlap_foreground_threshold;

    // 条件2：结构响应图要能转成前景 mask。
    if (!base_pipeline_helpers::buildForegroundMask(ctx.structure_data.first.response,
                                                     thresholdValue,
                                                     sourceMask) ||
        !base_pipeline_helpers::buildForegroundMask(ctx.structure_data.second.response,
                                                     thresholdValue,
                                                     targetMask)) {
        ctx.result.message = "structure overlap validation failed: cannot build masks";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    base_pipeline_helpers::dilateMaskIfRequested(sourceMask, _config.structure_overlap_dilate_size);
    base_pipeline_helpers::dilateMaskIfRequested(targetMask, _config.structure_overlap_dilate_size);

    cv::Mat warpedSourceMask;

    // 条件3：source mask 要能 warp 到 target 坐标系。
    if (!base_pipeline_helpers::warpStructureMask(ctx, sourceMask, targetMask.size(), warpedSourceMask)) {
        ctx.result.message = "structure overlap validation failed: cannot warp source mask";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    const double iou = base_pipeline_helpers::computeMaskIou(warpedSourceMask, targetMask);
    ctx.result.structure_overlap_iou = iou;
    if (iou < 0.0) {
        ctx.result.message = "structure overlap IoU failed: empty structure union";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    // 条件4：结构重叠 IoU 要达到阈值。
    IR_LOG_INFO("Structure overlap IoU=", iou, ", min=", _config.min_structure_overlap_iou);
    if (iou < _config.min_structure_overlap_iou) {
        ctx.result.message = "structure overlap IoU below threshold: " +
                             std::to_string(iou) + " < " +
                             std::to_string(_config.min_structure_overlap_iou);
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    return true;
}

bool BasePipeline::validateWarpQuality(RegistrationContext& ctx) {
    ctx.result.final_validation_source.clear();
    ctx.result.warp_overlap_containment = -1.0;
    ctx.result.warp_source_coverage = -1.0;
    ctx.result.warp_target_coverage = -1.0;
    ctx.result.warp_edge_alignment_iou = -1.0;
    ctx.result.warp_photometric_error = -1.0;

    const auto options = warp_quality::makeFinalWarpQualityOptions(_config);

    // 没启用任何 warp 质量验证时，直接视为通过。
    if (!warp_quality::hasEnabledWarpQualityChecks(options)) {
        return true;
    }

    // 条件1：warp 结果必须存在。
    if (!_config.warp || ctx.warped_image.empty()) {
        ctx.result.message = "warp validation failed: warped image is empty";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    // 条件2：warp 和 target 尺寸必须一致。
    if (ctx.images.second.empty() || ctx.warped_image.size() != ctx.images.second.size()) {
        ctx.result.message =
            "warp validation failed: warped image and target have different sizes";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    cv::Mat matrix;
    const bool needsTransformMatrix =
        options.validate_containment;
    if (needsTransformMatrix && !base_pipeline_helpers::activeTransformMatrix(ctx, matrix)) {
        ctx.result.message = "warp mask validation failed: no transform matrix";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    warp_quality::WarpQualityResult quality;
    const bool ok = warp_quality::evaluateWarpQuality(options,
                                                      ctx.images.first,
                                                      ctx.images.second,
                                                      matrix,
                                                      ctx.warped_image,
                                                      quality);
    syncWarpQualityToResult(ctx.result, quality);
    const bool finalOk = resolveDirectFinalValidationReference(_config, ctx, ok, quality);
    if (!finalOk) {
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    IR_LOG_INFO("Warp quality: final_source=",
                ctx.result.final_validation_source,
                ", containment=",
                ctx.result.warp_overlap_containment,
                ", source_coverage=",
                ctx.result.warp_source_coverage,
                ", target_coverage=",
                ctx.result.warp_target_coverage,
                ", edge_iou=",
                ctx.result.warp_edge_alignment_iou,
                ", nmad=",
                ctx.result.warp_photometric_error);
    return true;
}

std::string BasePipeline::buildOutputStem(const RegistrationContext& ctx) const {
    return ctx.image1_path.stem().string() + "_" + ctx.image2_path.stem().string() + "_" + name();
}

bool BasePipeline::saveOutputs(RegistrationContext& ctx) {
    if (!_config.save_visuals || ctx.output_dir.empty()) {
        return true;
    }

    // 1. 创建通用输出目录。
    const fs::path originals_dir = ctx.output_dir / "originals";
    const fs::path warped_dir = ctx.output_dir / "warped";
    const fs::path blend_dir = ctx.output_dir / "blend";
    const fs::path false_color_overlay_dir = ctx.output_dir / "false_color_overlay";
    std::error_code ec;
    fs::create_directories(originals_dir, ec);
    fs::create_directories(warped_dir, ec);
    fs::create_directories(blend_dir, ec);
    fs::create_directories(false_color_overlay_dir, ec);

    const std::string stem = buildOutputStem(ctx);
    const std::string sampleStem = ctx.image1_path.stem().string() + "_" +
                                   ctx.image2_path.stem().string();

    // 2. 按独立开关保存原始 source / target，便于和 warped / blend 对照。
    if (_config.save_originals && !ctx.images.first.empty()) {
        const fs::path out = originals_dir / (sampleStem + "_source_original.png");
        if (cv::imwrite(out.string(), ctx.images.first)) {
            IR_LOG_INFO("Wrote source original image: ", out.string());
        } else {
            IR_LOG_WARN("Failed to write source original image: ", out.string());
        }
    }
    if (_config.save_originals && !ctx.images.second.empty()) {
        const fs::path out = originals_dir / (sampleStem + "_target_original.png");
        if (cv::imwrite(out.string(), ctx.images.second)) {
            IR_LOG_INFO("Wrote target original image: ", out.string());
        } else {
            IR_LOG_WARN("Failed to write target original image: ", out.string());
        }
    }

    // 3. 保存 warped source，并和 target 按同尺寸画布生成 blend。
    if (_config.warp && !ctx.warped_image.empty()) {
        if (_config.save_warped) {
            const fs::path out = warped_dir / (stem + "_warped.png");
            cv::imwrite(out.string(), ctx.warped_image);
            IR_LOG_INFO("Wrote warped image: ", out.string());
        }

        if (_config.save_blend && ctx.warped_image.size() == ctx.images.second.size() &&
            ctx.warped_image.type() == ctx.images.second.type()) {
            cv::Mat blend;
            cv::addWeighted(ctx.warped_image, 0.5, ctx.images.second, 0.5, 0.0, blend);
            const fs::path blend_out = blend_dir / (stem + "_blend.png");
            cv::imwrite(blend_out.string(), blend);
            IR_LOG_INFO("Wrote blend image: ", blend_out.string());
        }

        if (_config.save_false_color_overlay &&
            ctx.warped_image.size() == ctx.images.second.size()) {
            cv::Mat falseColorOverlay;
            if (base_pipeline_helpers::buildFalseColorOverlay(
                    ctx.warped_image,
                    ctx.images.second,
                    _config.warp_quality.overlap.foreground_threshold,
                    falseColorOverlay)) {
                const fs::path false_color_out =
                    false_color_overlay_dir / (stem + "_false_color_overlay.png");
                cv::imwrite(false_color_out.string(), falseColorOverlay);
                IR_LOG_INFO("Wrote false-color overlay image: ", false_color_out.string());
            } else {
                IR_LOG_WARN("Failed to build false-color overlay image for: ", stem);
            }
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

bool BasePipeline::run(RegistrationContext& ctx, const PipelineRunOptions& options) {
    Timer total;

    // 1. 初始化本次运行上下文；批处理优先使用本次传入的路径，单次运行回退到 YAML 配置。
    ctx.reset();
    ctx.image1_path = options.image1_path.empty() ? _config.image1_path : options.image1_path;
    ctx.image2_path = options.image2_path.empty() ? _config.image2_path : options.image2_path;
    ctx.output_dir = options.output_dir.empty() ? _config.output_dir : options.output_dir;

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

    // 几何估计器读取同一份预建快照，避免在各估计器内部重新展开对应点。
    refreshCorrespondenceSnapshot(ctx);
    const bool estimationOk = runEstimation(ctx);

    // 估计过程会写回内点；刷新后评测和可视化共享最终数据，而不重复构建快照。
    refreshCorrespondenceSnapshot(ctx);
    if (!estimationOk) {
        const std::string detail =
            ctx.geometry_data.message.empty()
                ? std::string("estimation failed")
                : std::string("estimation failed: ") + ctx.geometry_data.message;
        return fail(detail);
    }

    if (!runWarp(ctx)) {
        return fail(ctx.result.message.empty() ? "warp failed" : ctx.result.message);
    }
    if (_config.warp && ctx.warped_image.empty()) {
        return fail("warp failed: warped image is empty");
    }
    // 运行评测指标（仅成功时计算）
    if (!_evaluator.metrics().empty()) {
        Sample dummySample;
        _evaluator.evaluate(ctx, dummySample);
        syncEvaluationMetricsToResult(ctx);
    }
    if (!validateRegistrationQuality(ctx)) {
        return fail(ctx.result.message.empty() ? "registration validation failed"
                                              : ctx.result.message);
    }

    // 3. 在写盘前冻结成功路径的总耗时，保证与失败路径采用相同口径。
    ctx.result.success = true;
    ctx.result.t_total_ms = total.elapsedMs();
    ctx.result.message = "OK";
    saveOutputs(ctx);
    return true;
}

} // namespace ir
