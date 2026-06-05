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
    double warp_overlap_iou = -1.0;
    double warp_photometric_error = -1.0;

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
