#pragma once

#include <string>
#include <opencv2/core.hpp>

#include "core/config.h"
#include "evaluator/quality/edge_structure_diagnostic.h"
#include "evaluator/quality/height_difference_evaluator.h"
namespace ir::warp_quality {

/// Warp 质量验证的独立配置视图。
///
/// 该结构承载 overlap、高度差与结构法线组判定参数，
/// 便于最终 pipeline 验证和直接法点特征初始化验证共用同一套计算逻辑。
struct WarpQualityOptions {
    bool validate_containment = false;
    double min_overlap_containment = 0.20;
    int foreground_threshold = 10;
    int containment_tolerance_pixels = 0;

    bool validate_height_difference = false;
    /// 原始高度差失败时，是否允许以全局高度偏移补偿作为回退验证。
    bool compensate_global_height_offset = true;
    /// 用于成功判定的高度差分位数；支持 50、75、90、95。
    int height_difference_percentile = 90;
    /// 所选高度差分位数的最大允许值。
    double max_height_difference_error = 0.10;
    /// 原始 P90 超过常规阈值时，是否允许局部噪声尾部例外通过。
    bool allow_local_noise_fallback = false;
    /// 局部噪声例外要求原始 P75 不超过该阈值。
    double local_noise_p75_max_abs_error = 0.10;
    /// 局部噪声例外要求的较小前景包含率。
    double local_noise_min_containment = 0.90;

    edge_structure_diagnostic::Options edge_structure;

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
    edge_structure_diagnostic::Result edge_structure;

    // Raw absolute height-difference diagnostics over the valid overlap.
    int height_diff_valid_count = 0;
    double height_diff_overlap_ratio = -1.0;
    double height_diff_mean = -1.0;
    double height_diff_p50 = -1.0;
    double height_diff_p75 = -1.0;
    double height_diff_p90 = -1.0;
    double height_diff_p95 = -1.0;
    double height_diff_max = -1.0;
    // Compensated height-difference diagnostics. They are populated only when
    // the raw validation fails and the optional compensation fallback runs.
    bool height_diff_compensation_attempted = false;
    double height_diff_global_offset = -1.0;
    double height_diff_compensated_mean = -1.0;
    double height_diff_compensated_p50 = -1.0;
    double height_diff_compensated_p75 = -1.0;
    double height_diff_compensated_p90 = -1.0;
    double height_diff_compensated_p95 = -1.0;
    double height_diff_compensated_max = -1.0;
    bool height_diff_local_noise_candidate = false;
    double height_diff_p90_p75_gap = -1.0;
    double height_diff_local_noise_containment = -1.0;

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
/// height difference、high-gray containment 或结构线组诊断时使用
/// sourceToTargetMatrix 临时生成 warped 图像。结构诊断始终比较 warp 后 source 与 target。
bool evaluateWarpQuality(const WarpQualityOptions& options,
                         const cv::Mat& sourceImage,
                         const cv::Mat& targetImage,
                         const cv::Mat& sourceToTargetMatrix,
                         const cv::Mat& prewarpedSource,
                         WarpQualityResult& result);

} // namespace ir::warp_quality
