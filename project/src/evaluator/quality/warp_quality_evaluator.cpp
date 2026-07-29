#include "evaluator/quality/warp_quality_evaluator.h"

#include <algorithm>
#include <string>

#include <opencv2/imgproc.hpp>

#include "pipeline/base_pipeline_helpers.h"

namespace ir::warp_quality {

namespace {

bool warpImageToTarget(const cv::Mat& source,
                       const cv::Size& targetSize,
                       const cv::Mat& matrix,
                       cv::Mat& warped) {
    warped.release();
    if (source.empty() || matrix.empty()) {
        return false;
    }

    // 1. 根据矩阵类型选择透视或仿射 warp，生成用于外观质量评估的临时图像。
    if (matrix.rows >= 3 && matrix.cols >= 3) {
        cv::warpPerspective(source,
                            warped,
                            matrix,
                            targetSize,
                            cv::INTER_LINEAR,
                            cv::BORDER_CONSTANT,
                            cv::Scalar(0));
        return true;
    }
    if (matrix.rows >= 2 && matrix.cols >= 3) {
        const cv::Mat affine = matrix(cv::Rect(0, 0, 3, 2)).clone();
        cv::warpAffine(source,
                       warped,
                       affine,
                       targetSize,
                       cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(0));
        return true;
    }
    return false;
}

void fail(WarpQualityResult& result, const std::string& message) {
    result.pass = false;
    result.message = message;
}

} // namespace

WarpQualityOptions makeFinalWarpQualityOptions(const PipelineConfig& cfg) {
    const auto& overlap = cfg.warp_quality.overlap;
    const auto& photometric = cfg.warp_quality.photometric;
    const auto& edge = cfg.warp_quality.edge_alignment;

    WarpQualityOptions options;
    options.validate_containment = overlap.containment_enabled;
    options.min_overlap_containment = overlap.min_containment;
    options.foreground_threshold = overlap.foreground_threshold;
    options.validate_photometric = photometric.enabled;
    options.max_photometric_error = photometric.max_nmad;
    options.validate_edge_alignment = edge.enabled;
    options.min_edge_alignment_iou = edge.min_iou;
    options.edge_alignment_canny_low_threshold = edge.canny_low_threshold;
    options.edge_alignment_canny_high_threshold = edge.canny_high_threshold;
    options.edge_alignment_dilate_size = edge.dilate_size;
    options.min_edge_alignment_pixels = edge.min_edge_pixels;
    return options;
}

WarpQualityOptions makeInitializerWarpQualityOptions(const PipelineConfig& cfg) {
    const auto& overlap = cfg.feature_initializer.validation.overlap;
    const auto& photometric = cfg.feature_initializer.validation.photometric;
    const auto& edge = cfg.feature_initializer.validation.edge_alignment;

    WarpQualityOptions options;
    options.validate_containment = overlap.containment_enabled;
    options.min_overlap_containment = overlap.min_containment;
    options.foreground_threshold = overlap.foreground_threshold;
    options.validate_photometric = photometric.enabled;
    options.max_photometric_error = photometric.max_nmad;
    options.validate_edge_alignment = edge.enabled;
    options.min_edge_alignment_iou = edge.min_iou;
    options.edge_alignment_canny_low_threshold = edge.canny_low_threshold;
    options.edge_alignment_canny_high_threshold = edge.canny_high_threshold;
    options.edge_alignment_dilate_size = edge.dilate_size;
    options.min_edge_alignment_pixels = edge.min_edge_pixels;
    return options;
}

bool hasEnabledWarpQualityChecks(const WarpQualityOptions& options) {
    return options.validate_containment ||
           options.validate_photometric || options.validate_edge_alignment;
}

bool evaluateWarpQuality(const WarpQualityOptions& options,
                         const cv::Mat& sourceImage,
                         const cv::Mat& targetImage,
                         const cv::Mat& sourceToTargetMatrix,
                         const cv::Mat& prewarpedSource,
                         WarpQualityResult& result) {
    result = WarpQualityResult{};
    if (!hasEnabledWarpQualityChecks(options)) {
        return true;
    }

    const int thresholdValue = std::clamp(options.foreground_threshold, 0, 255);
    cv::Mat sourceMask;
    cv::Mat targetMask;

    // 1. 先构建 target 前景 mask；后续 overlap、photometric 和 edge 都依赖它。
    if (targetImage.empty() ||
        !base_pipeline_helpers::buildForegroundMask(targetImage, thresholdValue, targetMask)) {
        fail(result, "warp validation failed: cannot build target foreground mask");
        return false;
    }

    cv::Mat warpedSourceMask;
    const bool needOverlapGeometry =
        options.validate_containment;
    if (needOverlapGeometry) {
        // 2. 几何覆盖类指标必须使用原始 source mask 和 source -> target 矩阵计算。
        if (sourceImage.empty() ||
            !base_pipeline_helpers::buildForegroundMask(sourceImage,
                                                        thresholdValue,
                                                        sourceMask) ||
            !base_pipeline_helpers::warpMaskToTargetSize(sourceMask,
                                                         targetImage.size(),
                                                         sourceToTargetMatrix,
                                                         warpedSourceMask)) {
            fail(result, "warp mask validation failed: cannot warp source mask");
            return false;
        }
    }

    if (options.validate_containment) {
        result.overlap_containment =
            base_pipeline_helpers::computeMaskLocalContainment(sourceMask,
                                                               warpedSourceMask,
                                                               targetMask);
        if (result.overlap_containment < 0.0) {
            fail(result, "warp local containment failed: empty foreground");
            return false;
        }
        if (result.overlap_containment < options.min_overlap_containment) {
            fail(result,
                 "warp local containment below threshold: " +
                     std::to_string(result.overlap_containment) + " < " +
                     std::to_string(options.min_overlap_containment));
            return false;
        }
    }

    cv::Mat warped = prewarpedSource;
    const bool needAppearance = options.validate_photometric || options.validate_edge_alignment;
    if (needAppearance) {
        // 5. 外观类指标需要 warped 图像；未传入时用候选矩阵临时生成。
        if (warped.empty() &&
            !warpImageToTarget(sourceImage, targetImage.size(), sourceToTargetMatrix, warped)) {
            fail(result, "warp validation failed: cannot build warped source image");
            return false;
        }
        if (warped.empty() || warped.size() != targetImage.size()) {
            fail(result,
                 "warp validation failed: warped image and target have different sizes");
            return false;
        }
    }

    cv::Mat overlapMask;
    if (needAppearance) {
        cv::Mat warpedMask;
        if (!base_pipeline_helpers::buildForegroundMask(warped, thresholdValue, warpedMask)) {
            fail(result, "warp validation failed: cannot build warped foreground mask");
            return false;
        }
        cv::bitwise_and(warpedMask, targetMask, overlapMask);
    }

    if (options.validate_edge_alignment) {
        result.edge_alignment_iou =
            base_pipeline_helpers::computeEdgeAlignmentIou(
                warped,
                targetImage,
                overlapMask,
                options.edge_alignment_canny_low_threshold,
                options.edge_alignment_canny_high_threshold,
                options.edge_alignment_dilate_size,
                options.min_edge_alignment_pixels);
        if (result.edge_alignment_iou < 0.0) {
            fail(result, "warp edge alignment validation failed: invalid edge overlap");
            return false;
        }
        if (result.edge_alignment_iou < options.min_edge_alignment_iou) {
            fail(result,
                 "warp edge alignment IoU below threshold: " +
                     std::to_string(result.edge_alignment_iou) + " < " +
                     std::to_string(options.min_edge_alignment_iou));
            return false;
        }
    }

    if (options.validate_photometric) {
        result.photometric_error =
            base_pipeline_helpers::computePhotometricError(warped, targetImage, overlapMask);
        if (result.photometric_error < 0.0) {
            fail(result, "warp photometric validation failed: empty overlap");
            return false;
        }
        if (result.photometric_error > options.max_photometric_error) {
            fail(result,
                 "warp photometric error above threshold: " +
                     std::to_string(result.photometric_error) + " > " +
                     std::to_string(options.max_photometric_error));
            return false;
        }
    }

    return true;
}

} // namespace ir::warp_quality
