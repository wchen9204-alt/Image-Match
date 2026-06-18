#include "pipeline/base_pipeline.h"

#include <filesystem>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/config.h"
#include "pipeline/base_pipeline_helpers.h"
#include "transform/affine_warper.h"
#include "transform/perspective_warper.h"
#include "utils/logger.h"
#include "utils/timer.h"

namespace fs = std::filesystem;

namespace ir {

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

    // 3. 创建通用 warp 组件；默认使用 2x3 仿射 warper，透视 warper 作为扩展保留。
    _warper = std::make_shared<AffineWarper>();

    // 4. 加载评测指标（可选）。
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

    // 3. HOMOGRAPHY 必须保留完整 3x3 透视项；仿射族继续走默认 2x3 warper。
    if (t == GeometryType::HOMOGRAPHY) {
        PerspectiveWarper perspectiveWarper;
        return perspectiveWarper.warp(ctx);
    }
    return _warper->warp(ctx);
}

bool BasePipeline::validateRegistrationQuality(RegistrationContext& ctx) {
    // 先重置可选验证结果。
    ctx.result.structure_overlap_iou = -1.0;

    // 顺序执行各项验证。
    if (!validateMatchQuality(ctx)) {
        return false;
    }
    if (!validateStructureOverlap(ctx)) {
        return false;
    }
    if (!validateWarpQuality(ctx)) {
        return false;
    }
    return true;
}

bool BasePipeline::validateMatchQuality(RegistrationContext& ctx) {
    if (!_config.validate_match_quality) {
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

    // 统一读取结果对象里的匹配统计。
    const auto& r = ctx.result;
    IR_LOG_INFO("Match quality: inliers=",
                r.num_inliers,
                ", ratio=",
                r.inlier_ratio,
                ", reproj=",
                r.mean_reproj_error);

    // 条件1：最少内点数。是否判失败由 fail_on_violation 控制。
    if (_config.min_match_inliers > 0 && r.num_inliers < _config.min_match_inliers) {
        const std::string message = "inliers " + std::to_string(r.num_inliers) + " < " +
                                    std::to_string(_config.min_match_inliers);
        if (!handleViolation(message)) {
            return false;
        }
    }

    // 条件2：最低内点率。是否判失败由 fail_on_violation 控制。
    if (_config.min_match_inlier_ratio >= 0.0 &&
        r.inlier_ratio < _config.min_match_inlier_ratio) {
        const std::string message = "inlier ratio " + std::to_string(r.inlier_ratio) +
                                    " < " + std::to_string(_config.min_match_inlier_ratio);
        if (!handleViolation(message)) {
            return false;
        }
    }

    // 条件3：最大重投影误差。是否判失败由 fail_on_violation 控制。
    if (_config.max_match_reproj_error >= 0.0 &&
        r.mean_reproj_error > _config.max_match_reproj_error) {
        const std::string message = "reprojection error " +
                                    std::to_string(r.mean_reproj_error) + " > " +
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
        const int thresholdValue = _config.warp_overlap_foreground_threshold;
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
            const double spatialCoverage =
                base_pipeline_helpers::computeInlierSpatialCoverage(
                    ctx.keypoint_data.first.keypoints,
                    ctx.keypoint_data.second.keypoints,
                    ctx.keypoint_match_data.inlier_matches,
                    sourceMask,
                    targetMask,
                    sourceSpatialCoverage,
                    targetSpatialCoverage);
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

bool BasePipeline::validateStructureOverlap(RegistrationContext& ctx) {
    if (!_config.validate_structure_overlap) {
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
    ctx.result.warp_overlap_containment = -1.0;
    ctx.result.warp_source_coverage = -1.0;
    ctx.result.warp_target_coverage = -1.0;
    ctx.result.warp_bidirectional_coverage = -1.0;
    ctx.result.warp_photometric_error = -1.0;

    // 没启用任何 warp 质量验证时，直接视为通过。
    if (!_config.validate_warp_containment && !_config.validate_warp_bidirectional_coverage &&
        !_config.validate_warp_photometric) {
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

    // 条件3：两张图都要能提取前景 mask。
    cv::Mat warpedMask;
    cv::Mat targetMask;
    const int thresholdValue = _config.warp_overlap_foreground_threshold;
    if (!base_pipeline_helpers::buildForegroundMask(ctx.warped_image, thresholdValue, warpedMask) ||
        !base_pipeline_helpers::buildForegroundMask(ctx.images.second,
                                                     thresholdValue,
                                                     targetMask)) {
        ctx.result.message = "warp validation failed: cannot build foreground masks";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    cv::Mat matrix;
    cv::Mat inverseMatrix;
    cv::Mat sourceMask;
    cv::Mat reverseWarpedTargetMask;
    cv::Mat warpedSourceMask;
    double sourceCoverage = -1.0;
    double targetCoverage = -1.0;
    double bidirectionalCoverage = -1.0;
    double containment = -1.0;
    // - containmentPassForEither 表示局部包含率这一侧是否达标；
    // - coveragePassForEither 表示双向 coverage 这一侧是否达标。
    bool containmentPassForEither = false;
    bool coveragePassForEither = false;
    if (_config.validate_warp_bidirectional_coverage || _config.validate_warp_containment) {
        if (!base_pipeline_helpers::activeTransformMatrix(ctx, matrix)) {
            ctx.result.message = "warp mask validation failed: no transform matrix";
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
        if (!base_pipeline_helpers::buildForegroundMask(ctx.images.first,
                                                        thresholdValue,
                                                        sourceMask) ||
            !base_pipeline_helpers::warpMaskToTargetSize(sourceMask,
                                                         ctx.images.second.size(),
                                                         matrix,
                                                         warpedSourceMask)) {
            ctx.result.message = "warp mask validation failed: cannot warp source mask";
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
        if (_config.validate_warp_bidirectional_coverage) {
            if (!base_pipeline_helpers::invertTransformMatrix(matrix, inverseMatrix) ||
                !base_pipeline_helpers::warpMaskToTargetSize(targetMask,
                                                             ctx.images.first.size(),
                                                             inverseMatrix,
                                                             reverseWarpedTargetMask)) {
                ctx.result.message = "warp mask validation failed: cannot inverse-warp target mask";
                IR_LOG_WARN(ctx.result.message);
                return false;
            }
        }
    }

    // 条件5：双向 coverage 用于判断局部图是否在某个方向上被完整保留。
    if (_config.validate_warp_bidirectional_coverage) {
        // 1. 正向看 source warp 到 target 画布后保留了多少前景。
        sourceCoverage = base_pipeline_helpers::computeMaskCoverage(sourceMask, warpedSourceMask);
        // 2. 反向看 target inverse-warp 到 source 画布后保留了多少前景。
        targetCoverage =
            base_pipeline_helpers::computeMaskCoverage(targetMask, reverseWarpedTargetMask);
        bidirectionalCoverage = std::max(sourceCoverage, targetCoverage);

        ctx.result.warp_source_coverage = sourceCoverage;
        ctx.result.warp_target_coverage = targetCoverage;
        ctx.result.warp_bidirectional_coverage = bidirectionalCoverage;

        if (sourceCoverage < 0.0 || targetCoverage < 0.0 || bidirectionalCoverage < 0.0) {
            ctx.result.message = "warp bidirectional coverage validation failed: invalid coverage";
            IR_LOG_WARN(ctx.result.message);
            return false;
        }

        IR_LOG_INFO("Warp source coverage=",
                    sourceCoverage,
                    ", target coverage=",
                    targetCoverage,
                    ", bidirectional coverage=",
                    bidirectionalCoverage,
                    ", min=",
                    _config.min_warp_bidirectional_coverage);
        // 严格模式下，双向 coverage 单项不达标就直接失败。
        // either-pass 模式下先只记录结果，等 containment 也算完后统一判断。
        if (!_config.accept_warp_overlap_if_either_passes &&
            bidirectionalCoverage < _config.min_warp_bidirectional_coverage) {
            ctx.result.message = "warp bidirectional coverage below threshold: " +
                                 std::to_string(bidirectionalCoverage) + " < " +
                                 std::to_string(_config.min_warp_bidirectional_coverage);
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
    }

    // 条件6：局部包含率要达到阈值，适配一张图是另一张图局部的场景。
    if (_config.validate_warp_containment) {
        containment = base_pipeline_helpers::computeMaskLocalContainment(sourceMask,
                                                                         warpedSourceMask,
                                                                         targetMask);
        ctx.result.warp_overlap_containment = containment;
        if (containment < 0.0) {
            ctx.result.message = "warp local containment failed: empty foreground";
            IR_LOG_WARN(ctx.result.message);
            return false;
        }

        IR_LOG_INFO("Warp local containment=",
                    containment,
                    ", min=",
                    _config.min_warp_overlap_containment);
        // 严格模式下，局部包含率单项不达标就直接失败。
        // either-pass 模式下改为和 coverage 一起看，允许“局部图完整落入大图”
        // 或“反向覆盖关系成立”中的任意一种成立。
        if (!_config.accept_warp_overlap_if_either_passes &&
            containment < _config.min_warp_overlap_containment) {
            ctx.result.message = "warp local containment below threshold: " +
                                 std::to_string(containment) + " < " +
                                 std::to_string(_config.min_warp_overlap_containment);
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
    }

    // either-pass 模式用于“其中一张图可能只是另一张图局部”的场景。
    // 只要两者至少有一个成立，就允许 warp overlap 这一关先通过。
    if (_config.accept_warp_overlap_if_either_passes &&
        (_config.validate_warp_bidirectional_coverage || _config.validate_warp_containment)) {
        containmentPassForEither =
            !_config.validate_warp_containment ||
            (containment >= 0.0 && containment >= _config.min_warp_overlap_containment);
        coveragePassForEither =
            !_config.validate_warp_bidirectional_coverage ||
            (bidirectionalCoverage >= 0.0 &&
             bidirectionalCoverage >= _config.min_warp_bidirectional_coverage);
        if (!containmentPassForEither && !coveragePassForEither) {
            ctx.result.message =
                "warp overlap validation failed: containment " +
                std::to_string(containment) + " < " +
                std::to_string(_config.min_warp_overlap_containment) +
                " and bidirectional coverage " +
                std::to_string(bidirectionalCoverage) + " < " +
                std::to_string(_config.min_warp_bidirectional_coverage);
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
    }

    // 条件7：重叠区域光度误差不能过大。
    if (_config.validate_warp_photometric) {
        cv::Mat overlapMask;
        cv::bitwise_and(warpedMask, targetMask, overlapMask);
        const double nmad =
            base_pipeline_helpers::computePhotometricError(ctx.warped_image,
                                                            ctx.images.second,
                                                            overlapMask);
        ctx.result.warp_photometric_error = nmad;
        if (nmad < 0.0) {
            ctx.result.message = "warp photometric validation failed: empty overlap";
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
        IR_LOG_INFO("Warp photometric NMAD=",
                    nmad,
                    ", max=",
                    _config.max_warp_photometric_error);
        if (nmad > _config.max_warp_photometric_error) {
            ctx.result.message = "warp photometric error above threshold: " +
                                 std::to_string(nmad) +
                                 " > " + std::to_string(_config.max_warp_photometric_error);
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
        // 当 overlap 只是靠 coverage 通过、而 containment 没通过时，
        // 说明几何关系仍然偏可疑，例如可能只是边缘大面积重叠、
        // 或者发生了“形状能盖住但内容没有真对齐”的误配。
        // 所以这里额外套一个更严格的 photometric 阈值，专门压这类误判。
        if (_config.max_warp_photometric_error_for_coverage_only >= 0.0 &&
            _config.accept_warp_overlap_if_either_passes &&
            coveragePassForEither &&
            !containmentPassForEither &&
            nmad > _config.max_warp_photometric_error_for_coverage_only) {
            ctx.result.message =
                "warp coverage-only photometric error above threshold: " +
                std::to_string(nmad) + " > " +
                std::to_string(_config.max_warp_photometric_error_for_coverage_only);
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
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
    const fs::path false_color_overlay_dir = _config.output_dir / "false_color_overlay";
    std::error_code ec;
    fs::create_directories(originals_dir, ec);
    fs::create_directories(warped_dir, ec);
    fs::create_directories(blend_dir, ec);
    fs::create_directories(false_color_overlay_dir, ec);

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

        if (ctx.warped_image.size() == ctx.images.second.size()) {
            cv::Mat falseColorOverlay;
            if (base_pipeline_helpers::buildFalseColorOverlay(
                    ctx.warped_image,
                    ctx.images.second,
                    _config.warp_overlap_foreground_threshold,
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
    }
    if (!validateRegistrationQuality(ctx)) {
        return fail(ctx.result.message.empty() ? "registration validation failed"
                                              : ctx.result.message);
    }

    saveOutputs(ctx);
    // 3. 所有阶段完成后记录总耗时和成功状态。
    ctx.result.success = true;
    ctx.result.t_total_ms = total.elapsedMs();
    ctx.result.message = "OK";
    return true;
}

} // namespace ir
