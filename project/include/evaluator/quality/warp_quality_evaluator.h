#pragma once

#include <string>

#include <opencv2/core.hpp>

#include "core/config.h"

namespace ir::warp_quality {

/// Warp 质量验证的独立配置视图。
///
/// 该结构只承载 overlap、photometric 和 edge alignment 相关参数，
/// 便于最终 pipeline 验证和直接法点特征初始化验证共用同一套计算逻辑。
struct WarpQualityOptions {
    bool validate_containment = false;
    double min_overlap_containment = 0.20;

    bool validate_bidirectional_coverage = false;
    double min_bidirectional_coverage = -1.0;

    bool accept_overlap_if_either_passes = false;
    int foreground_threshold = 10;

    bool validate_photometric = false;
    double max_photometric_error = 0.15;
    double max_photometric_error_for_coverage_only = -1.0;

    bool validate_edge_alignment = false;
    double min_edge_alignment_iou = 0.08;
    int edge_alignment_canny_low_threshold = 50;
    int edge_alignment_canny_high_threshold = 150;
    int edge_alignment_dilate_size = 3;
    int min_edge_alignment_pixels = 20;
};

/// Warp 质量验证的计算结果。
///
/// pass/message 表示最终是否通过；其余字段用于写回 summary，
/// 也用于候选初值之间的质量比较。
struct WarpQualityResult {
    bool pass = true;
    std::string message;

    double overlap_containment = -1.0;
    double source_coverage = -1.0;
    double target_coverage = -1.0;
    double bidirectional_coverage = -1.0;
    double edge_alignment_iou = -1.0;
    double photometric_error = -1.0;

    bool containment_pass_for_either = false;
    bool coverage_pass_for_either = false;
};

/// 从最终配准验证配置生成通用 warp 质量选项。
WarpQualityOptions makeFinalWarpQualityOptions(const PipelineConfig& cfg);

/// 从直接法点特征初始化配置生成通用 warp 质量选项。
WarpQualityOptions makeInitializerWarpQualityOptions(const PipelineConfig& cfg);

/// 判断当前选项是否启用了任意 warp 质量验证项。
bool hasEnabledWarpQualityChecks(const WarpQualityOptions& options);

/// 评估 source -> target 变换后的 warp 质量。
///
/// prewarpedSource 可传入已经生成的 warped 图像；为空时，函数会在需要
/// photometric 或 edge alignment 时使用 sourceToTargetMatrix 临时生成 warped 图像。
bool evaluateWarpQuality(const WarpQualityOptions& options,
                         const cv::Mat& sourceImage,
                         const cv::Mat& targetImage,
                         const cv::Mat& sourceToTargetMatrix,
                         const cv::Mat& prewarpedSource,
                         WarpQualityResult& result);

} // namespace ir::warp_quality
