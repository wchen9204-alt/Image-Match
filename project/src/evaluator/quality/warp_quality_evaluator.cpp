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

void syncHeightDifferenceDiagnostics(
    WarpQualityResult& result,
    const height_difference_evaluator::Statistics& statistics,
    const cv::Mat& targetMask) {
    result.height_diff_valid_count = statistics.raw_abs_diff_count;
    const int targetForegroundCount = targetMask.empty() ? 0 : cv::countNonZero(targetMask);
    result.height_diff_overlap_ratio =
        targetForegroundCount > 0
            ? static_cast<double>(statistics.raw_abs_diff_count) /
                  static_cast<double>(targetForegroundCount)
            : -1.0;
    result.height_diff_mean = statistics.raw_abs_diff_mean;
    result.height_diff_p50 = statistics.raw_abs_diff_p50;
    result.height_diff_p75 = statistics.raw_abs_diff_p75;
    result.height_diff_p90 = statistics.raw_abs_diff_p90;
    result.height_diff_p95 = statistics.raw_abs_diff_p95;
    result.height_diff_max = statistics.raw_abs_diff_max;
    result.height_diff_p90_p75_gap =
        statistics.raw_abs_diff_p90 >= 0.0 && statistics.raw_abs_diff_p75 >= 0.0
            ? statistics.raw_abs_diff_p90 - statistics.raw_abs_diff_p75
            : -1.0;
    if (statistics.compensated_p50 >= 0.0) {
        result.height_diff_global_offset = statistics.height_offset;
        result.height_diff_compensated_mean = statistics.compensated_mean;
        result.height_diff_compensated_p50 = statistics.compensated_p50;
        result.height_diff_compensated_p75 = statistics.compensated_p75;
        result.height_diff_compensated_p90 = statistics.compensated_p90;
        result.height_diff_compensated_p95 = statistics.compensated_p95;
        result.height_diff_compensated_max = statistics.compensated_max;
    }
}

} // namespace

WarpQualityOptions makeFinalWarpQualityOptions(const PipelineConfig& cfg) {
    const auto& overlap = cfg.warp_quality.overlap;
    const auto& heightDifference = cfg.warp_quality.height_difference;
    WarpQualityOptions options;
    options.validate_containment = overlap.containment_enabled;
    options.min_overlap_containment = overlap.min_containment;
    options.foreground_threshold = overlap.foreground_threshold;
    options.containment_tolerance_pixels = overlap.containment_tolerance_pixels;
    options.validate_height_difference = heightDifference.enabled;
    options.compensate_global_height_offset = heightDifference.compensate_global_height_offset;
    options.height_difference_percentile = heightDifference.percentile;
    options.max_height_difference_error = heightDifference.max_abs_error;
    options.allow_local_noise_fallback = heightDifference.local_noise_fallback_enabled;
    options.local_noise_p75_max_abs_error = heightDifference.local_noise_p75_max_abs_error;
    options.local_noise_min_containment = heightDifference.local_noise_min_containment;
    const auto& edgeStructure = cfg.warp_quality.edge_structure_diagnostic;
    options.edge_structure.enabled = edgeStructure.enabled;
    options.edge_structure.visibility_threshold = edgeStructure.visibility_threshold;
    options.edge_structure.min_foreground_elongation_ratio =
        edgeStructure.min_foreground_elongation_ratio;
    options.edge_structure.min_axis_occupancy = edgeStructure.min_axis_occupancy;
    options.edge_structure.max_centerline_deviation_ratio =
        edgeStructure.max_centerline_deviation_ratio;
    options.edge_structure.max_canvas_side_pixels = edgeStructure.max_canvas_side_pixels;
    options.edge_structure.max_canvas_pixels = edgeStructure.max_canvas_pixels;
    options.edge_structure.duplicate_line_normal_tolerance_pixels =
        edgeStructure.duplicate_line_normal_tolerance_pixels;
    options.edge_structure.duplicate_line_min_span_overlap_ratio =
        edgeStructure.duplicate_line_min_span_overlap_ratio;
    options.edge_structure.outer_longitudinal_edge_min_normal_separation_pixels =
        edgeStructure.outer_longitudinal_edge_min_normal_separation_pixels;
    options.edge_structure.outer_longitudinal_edge_max_normal_separation_pixels =
        edgeStructure.outer_longitudinal_edge_max_normal_separation_pixels;
    options.edge_structure.outer_longitudinal_edge_min_span_overlap_ratio =
        edgeStructure.outer_longitudinal_edge_min_span_overlap_ratio;
    options.edge_structure.min_fragment_length_pixels =
        edgeStructure.min_fragment_length_pixels;
    options.edge_structure.group_max_angle_difference_degrees =
        edgeStructure.group_max_angle_difference_degrees;
    options.edge_structure.group_max_normal_distance_pixels =
        edgeStructure.group_max_normal_distance_pixels;
    options.edge_structure.post_fit_group_normal_distance_pixels =
        edgeStructure.post_fit_group_normal_distance_pixels;
    options.edge_structure.min_line_group_actual_length_pixels =
        edgeStructure.min_line_group_actual_length_pixels;
    options.edge_structure.min_line_group_continuity_ratio =
        edgeStructure.min_line_group_continuity_ratio;
    options.edge_structure.max_line_group_gap_ratio = edgeStructure.max_line_group_gap_ratio;
    options.edge_structure.max_fragment_direction_spread_degrees =
        edgeStructure.max_fragment_direction_spread_degrees;
    options.edge_structure.max_line_fit_residual_pixels =
        edgeStructure.max_line_fit_residual_pixels;
    options.edge_structure.min_main_line_actual_length_ratio =
        edgeStructure.min_main_line_actual_length_ratio;
    options.edge_structure.direction_cluster_tolerance_degrees =
        edgeStructure.direction_cluster_tolerance_degrees;
    options.edge_structure.min_main_direction_support_ratio =
        edgeStructure.min_main_direction_support_ratio;
    options.edge_structure.max_main_direction_spread_degrees =
        edgeStructure.max_main_direction_spread_degrees;
    options.edge_structure.min_main_direction_margin = edgeStructure.min_main_direction_margin;
    options.edge_structure.max_main_direction_difference_degrees =
        edgeStructure.max_main_direction_difference_degrees;
    options.edge_structure.max_axis_classification_error_degrees =
        edgeStructure.max_axis_classification_error_degrees;
    options.edge_structure.min_horizontal_actual_length_ratio =
        edgeStructure.min_horizontal_actual_length_ratio;
    options.edge_structure.min_vertical_actual_length_ratio =
        edgeStructure.min_vertical_actual_length_ratio;
    options.edge_structure.max_vertical_unmatched_length_ratio =
        edgeStructure.max_vertical_unmatched_length_ratio;
    options.edge_structure.profile_smoothing_sigma = edgeStructure.profile_smoothing_sigma;
    options.edge_structure.min_peak_prominence = edgeStructure.min_peak_prominence;
    options.edge_structure.candidate_position_tolerance_pixels =
        edgeStructure.candidate_position_tolerance_pixels;
    options.edge_structure.final_position_tolerance_pixels =
        edgeStructure.final_position_tolerance_pixels;
    options.edge_structure.candidate_min_span_overlap_ratio =
        edgeStructure.candidate_min_span_overlap_ratio;
    options.edge_structure.min_shorter_line_overlap_ratio =
        edgeStructure.min_shorter_line_overlap_ratio;
    options.edge_structure.max_line_pair_angle_difference_degrees =
        edgeStructure.max_line_pair_angle_difference_degrees;
    options.edge_structure.match_position_cost_weight = edgeStructure.match_position_cost_weight;
    options.edge_structure.match_overlap_cost_weight = edgeStructure.match_overlap_cost_weight;
    options.edge_structure.match_angle_cost_weight = edgeStructure.match_angle_cost_weight;
    options.edge_structure.match_prominence_cost_weight =
        edgeStructure.match_prominence_cost_weight;
    options.edge_structure.min_strong_line_actual_length_pixels =
        edgeStructure.min_strong_line_actual_length_pixels;
    options.edge_structure.min_strong_peak_prominence = edgeStructure.min_strong_peak_prominence;
    options.edge_structure.ambiguity_score_margin = edgeStructure.ambiguity_score_margin;
    return options;
}

WarpQualityOptions makeInitializerWarpQualityOptions(const PipelineConfig& cfg) {
    const auto& overlap = cfg.feature_initializer.validation.overlap;
    const auto& heightDifference = cfg.feature_initializer.validation.height_difference;
    WarpQualityOptions options;
    options.validate_containment = overlap.containment_enabled;
    options.min_overlap_containment = overlap.min_containment;
    options.foreground_threshold = overlap.foreground_threshold;
    options.containment_tolerance_pixels = overlap.containment_tolerance_pixels;
    options.validate_height_difference = heightDifference.enabled;
    options.compensate_global_height_offset = heightDifference.compensate_global_height_offset;
    options.height_difference_percentile = heightDifference.percentile;
    options.max_height_difference_error = heightDifference.max_abs_error;
    options.allow_local_noise_fallback = heightDifference.local_noise_fallback_enabled;
    options.local_noise_p75_max_abs_error = heightDifference.local_noise_p75_max_abs_error;
    options.local_noise_min_containment = heightDifference.local_noise_min_containment;
    return options;
}

bool hasEnabledWarpQualityChecks(const WarpQualityOptions& options) {
    return options.validate_containment || options.validate_height_difference ||
           options.edge_structure.enabled;
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

    // NGF 使用独立的几何可见域并完整运行。诊断模式仅记录三态结果，不改变旧判定。
    const int thresholdValue = std::clamp(options.foreground_threshold, 0, 255);
    cv::Mat sourceMask;
    cv::Mat targetMask;

    const bool needHeightDifferenceDiagnostics = options.validate_height_difference;
    const bool needAppearance = needHeightDifferenceDiagnostics;
    // 结构诊断也必须比较最终 warp 后的 source 与 target；不能回到原始 source 坐标。
    const bool needWarpedSource = needAppearance || options.edge_structure.enabled;
    const bool needLocalNoiseGeometry =
        options.validate_height_difference && options.allow_local_noise_fallback;
    const bool needTargetMask = options.validate_containment || needAppearance;
    // 1. 只有 overlap/外观检查需要构建 target 前景 mask。
    if (needTargetMask &&
        (targetImage.empty() ||
         !base_pipeline_helpers::buildForegroundMask(targetImage, thresholdValue, targetMask))) {
        fail(result, "warp validation failed: cannot build target foreground mask");
        return false;
    }

    cv::Mat warpedSourceMask;
    const bool needOverlapGeometry = options.validate_containment || needLocalNoiseGeometry;
    std::string containmentFailureMessage;
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
            containmentFailureMessage = "warp mask validation failed: cannot warp source mask";
        }
    }

    if (needOverlapGeometry && containmentFailureMessage.empty()) {
        const double containment = base_pipeline_helpers::computeMaskLocalContainment(
            sourceMask,
            warpedSourceMask,
            targetMask,
            options.containment_tolerance_pixels);
        if (options.validate_containment) {
            result.overlap_containment = containment;
            if (result.overlap_containment < 0.0) {
                fail(result, "warp local containment failed: empty foreground");
                return false;
            }
            if (result.overlap_containment < options.min_overlap_containment) {
                containmentFailureMessage =
                    "warp local containment below threshold: " +
                    std::to_string(result.overlap_containment) + " < " +
                    std::to_string(options.min_overlap_containment);
            }
        }
        if (needLocalNoiseGeometry) {
            result.height_diff_local_noise_containment = containment;
        }
    }

    cv::Mat warped = prewarpedSource;
    if (needWarpedSource) {
        // 5. 外观与结构指标都在 target 坐标系比较；未传入时用候选矩阵临时生成。
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

    // 结构诊断优先给出三态结果：PASS 直接通过，FAIL 直接失败，
    // INSUFFICIENT 才继续使用高度差等通用质量指标。
    // 这里传入最终 warp 图，后续共同参考方向只做双方共有的坐标投影，不再承担几何配准。
    if (options.edge_structure.enabled) {
        edge_structure_diagnostic::evaluate(options.edge_structure,
                                            warped,
                                            targetImage,
                                            result.edge_structure);
        if (result.edge_structure.status == "PASS") {
            result.pass = true;
            result.message = "edge structure validation passed";
            return true;
        }
        if (result.edge_structure.status == "FAIL") {
            fail(result,
                 "edge structure validation failed: " +
                     result.edge_structure.message);
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

    height_difference_evaluator::Result heightDifferenceResult;
    if (needHeightDifferenceDiagnostics) {
        height_difference_evaluator::Options heightDifferenceOptions;
        heightDifferenceOptions.compensate_global_offset =
            options.compensate_global_height_offset;
        heightDifferenceOptions.percentile = options.height_difference_percentile;
        heightDifferenceOptions.max_abs_error = options.max_height_difference_error;
        heightDifferenceOptions.allow_local_noise_fallback =
            options.allow_local_noise_fallback;
        heightDifferenceOptions.local_noise_p75_max_abs_error =
            options.local_noise_p75_max_abs_error;
        heightDifferenceOptions.local_noise_min_containment =
            options.local_noise_min_containment;
        heightDifferenceResult = height_difference_evaluator::evaluate(
            warped,
            targetImage,
            overlapMask,
            result.height_diff_local_noise_containment,
            heightDifferenceOptions);
        syncHeightDifferenceDiagnostics(
            result, heightDifferenceResult.statistics, targetMask);
        result.height_diff_compensation_attempted =
            heightDifferenceResult.compensation_attempted;
        result.height_diff_local_noise_candidate = heightDifferenceResult.local_noise_pass;
    }

    const bool rawHeightPass =
        !options.validate_height_difference || heightDifferenceResult.raw_pass;
    const bool compensatedHeightPass = heightDifferenceResult.compensated_pass;
    const std::string& heightFailureMessage = heightDifferenceResult.failure_message;

    const bool containmentPass =
        !options.validate_containment || containmentFailureMessage.empty();
    const bool heightBranchPass = rawHeightPass ||
                                   result.height_diff_local_noise_candidate ||
                                   compensatedHeightPass;
    const bool legacyPass = containmentPass && heightBranchPass;
    result.pass = legacyPass;
    if (result.pass) {
        result.message = "OK";
        return true;
    }

    if (!containmentPass) {
        result.message = containmentFailureMessage;
    } else if (!heightBranchPass && !heightFailureMessage.empty()) {
        result.message = heightFailureMessage;
    } else {
        result.message = "warp quality validation failed";
    }
    return false;
}

} // namespace ir::warp_quality
