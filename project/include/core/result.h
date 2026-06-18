#pragma once

#include <string>

namespace ir {

/// 单次配准运行的汇总结果。
///
/// 这个结构主要用于记录执行状态、匹配数量和各阶段耗时，便于日志、
/// 输出 CSV 和后续统计分析。
struct RegistrationResult {
    bool success = false;
    std::string message;

    /// 各阶段关键数量统计。
    int num_keypoints_first = 0;
    int num_keypoints_second = 0;
    /// 结构法提取到的结构元素数量；点特征法中保持为 0。
    int num_structures_first = 0;
    int num_structures_second = 0;
    int num_raw_matches = 0;
    int num_filtered_matches = 0;
    int num_inliers = 0;

    /// 几何质量统计。
    double inlier_ratio = 0.0;
    double mean_reproj_error = 0.0;
    /// 最终内点在 source / target 前景包围盒中的最大空间覆盖率。
    double inlier_spatial_coverage = -1.0;
    /// warped source 与 target 的局部包含率；适合一张图是另一张图局部的场景。
    double warp_overlap_containment = -1.0;
    /// source 前景 warp 到 target 画布后的保留比例。
    double warp_source_coverage = -1.0;
    /// target 前景反向 warp 到 source 画布后的保留比例。
    double warp_target_coverage = -1.0;
    /// 双向 coverage，取 source / target coverage 的较大值。
    double warp_bidirectional_coverage = -1.0;
    double warp_photometric_error = -1.0;
    double structure_overlap_iou = -1.0;
    /// 各阶段耗时，单位毫秒。
    double t_load_ms = 0.0;
    double t_extract_ms = 0.0;
    double t_match_ms = 0.0;
    double t_filter_ms = 0.0;
    double t_geometry_ms = 0.0;
    double t_warp_ms = 0.0;
    double t_total_ms = 0.0;
};

} // namespace ir
