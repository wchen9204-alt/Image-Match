#pragma once

#include <string>

namespace ir {

/// 边缘结构严格线对被拒绝的单项原因统计；各原因之间允许重叠。
struct EdgeStructureStrictRejectionStats {
    int count = 0;
    double source_actual_length = 0.0;
    double target_actual_length = 0.0;
};

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
    /// 直接法算法自身的 confidence / response 分数；非直接法保持为 -1。
    double direct_confidence = -1.0;
    double mean_reproj_error = 0.0;
    /// 最终内点在 source / target 前景包围盒中的最大空间覆盖率。
    double inlier_spatial_coverage = -1.0;
    /// warped source 与 target 的局部包含率；适合一张图是另一张图局部的场景。
    double warp_overlap_containment = -1.0;
    /// source 前景 warp 到 target 画布后的保留比例。
    double warp_source_coverage = -1.0;
    /// target 前景反向 warp 到 source 画布后的保留比例。
    double warp_target_coverage = -1.0;
    /// 长条边缘结构验证；INSUFFICIENT 仅记录，FAIL 参与当前成功判定。
    std::string edge_structure_status = "NOT_RUN";
    std::string edge_structure_message;
    double edge_structure_source_foreground_elongation_ratio = -1.0;
    double edge_structure_target_foreground_elongation_ratio = -1.0;
    double edge_structure_source_axis_occupancy = -1.0;
    double edge_structure_target_axis_occupancy = -1.0;
    double edge_structure_source_centerline_deviation_ratio = -1.0;
    double edge_structure_target_centerline_deviation_ratio = -1.0;
    double edge_structure_source_foreground_long_side = -1.0;
    double edge_structure_target_foreground_long_side = -1.0;
    int edge_structure_common_canvas_width = 0;
    int edge_structure_common_canvas_height = 0;
    double edge_structure_common_canvas_offset_x = 0.0;
    double edge_structure_common_canvas_offset_y = 0.0;
    int edge_structure_source_visibility_pixels = 0;
    int edge_structure_target_visibility_pixels = 0;
    int edge_structure_common_visibility_pixels = 0;
    double edge_structure_visibility_area_ratio = -1.0;
    double edge_structure_visibility_overlap_containment = -1.0;
    /// source/target_visibility_ratio 的历史 ps/pt 别名，保持 CSV 兼容。
    double edge_structure_source_visibility_ratio = -1.0;
    double edge_structure_target_visibility_ratio = -1.0;
    double edge_structure_ps = -1.0;
    double edge_structure_pt = -1.0;
    int edge_structure_source_fragment_count = 0;
    int edge_structure_target_fragment_count = 0;
    int edge_structure_source_line_group_count = 0;
    int edge_structure_target_line_group_count = 0;
    int edge_structure_source_valid_line_group_count = 0;
    int edge_structure_target_valid_line_group_count = 0;
    int edge_structure_source_main_line_group_count = 0;
    int edge_structure_target_main_line_group_count = 0;
    bool edge_structure_source_main_direction_reliable = false;
    bool edge_structure_target_main_direction_reliable = false;
    double edge_structure_source_main_direction_degrees = -1.0;
    double edge_structure_target_main_direction_degrees = -1.0;
    double edge_structure_source_main_direction_support_ratio = -1.0;
    double edge_structure_target_main_direction_support_ratio = -1.0;
    double edge_structure_source_main_direction_spread_degrees = -1.0;
    double edge_structure_target_main_direction_spread_degrees = -1.0;
    double edge_structure_source_main_direction_margin = -1.0;
    double edge_structure_target_main_direction_margin = -1.0;
    double edge_structure_source_main_max_actual_length_ratio = -1.0;
    double edge_structure_target_main_max_actual_length_ratio = -1.0;
    double edge_structure_main_direction_difference_degrees = -1.0;
    double edge_structure_reference_direction_degrees = -1.0;

    // 水平方向线组匹配诊断；source/target_match_ratio 即 Hs/Ht。
    std::string edge_structure_horizontal_status = "NOT_RUN";
    int edge_structure_source_horizontal_eligible_line_groups = 0;
    int edge_structure_target_horizontal_eligible_line_groups = 0;
    int edge_structure_horizontal_candidate_pairs = 0;
    int edge_structure_horizontal_accepted_matches = 0;
    double edge_structure_source_horizontal_match_ratio = -1.0;
    double edge_structure_target_horizontal_match_ratio = -1.0;
    double edge_structure_source_horizontal_matched_actual_length = 0.0;
    double edge_structure_target_horizontal_matched_actual_length = 0.0;
    int edge_structure_horizontal_strong_conflict_count = 0;
    double edge_structure_source_horizontal_strong_conflict_actual_length = 0.0;
    double edge_structure_target_horizontal_strong_conflict_actual_length = 0.0;
    double edge_structure_source_horizontal_strong_conflict_length_ratio = -1.0;
    double edge_structure_target_horizontal_strong_conflict_length_ratio = -1.0;
    EdgeStructureStrictRejectionStats edge_structure_horizontal_position_rejections;
    EdgeStructureStrictRejectionStats edge_structure_horizontal_overlap_rejections;
    EdgeStructureStrictRejectionStats edge_structure_horizontal_angle_rejections;
    double edge_structure_source_horizontal_unmatched_actual_length = 0.0;
    double edge_structure_target_horizontal_unmatched_actual_length = 0.0;
    double edge_structure_source_horizontal_unmatched_length_ratio = -1.0;
    double edge_structure_target_horizontal_unmatched_length_ratio = -1.0;
    int edge_structure_horizontal_ambiguous_match_count = 0;
    double edge_structure_horizontal_ambiguous_actual_length_ratio = -1.0;
    double edge_structure_horizontal_matched_angle_difference_mean_degrees = -1.0;
    double edge_structure_horizontal_matched_angle_difference_max_degrees = -1.0;

    // 竖直方向线组匹配诊断；source/target_match_ratio 即 Vs/Vt。
    std::string edge_structure_vertical_status = "NOT_RUN";
    int edge_structure_source_vertical_eligible_line_groups = 0;
    int edge_structure_target_vertical_eligible_line_groups = 0;
    int edge_structure_vertical_candidate_pairs = 0;
    int edge_structure_vertical_accepted_matches = 0;
    double edge_structure_source_vertical_match_ratio = -1.0;
    double edge_structure_target_vertical_match_ratio = -1.0;
    double edge_structure_source_vertical_matched_actual_length = 0.0;
    double edge_structure_target_vertical_matched_actual_length = 0.0;
    int edge_structure_vertical_strong_conflict_count = 0;
    double edge_structure_source_vertical_strong_conflict_actual_length = 0.0;
    double edge_structure_target_vertical_strong_conflict_actual_length = 0.0;
    double edge_structure_source_vertical_strong_conflict_length_ratio = -1.0;
    double edge_structure_target_vertical_strong_conflict_length_ratio = -1.0;
    EdgeStructureStrictRejectionStats edge_structure_vertical_position_rejections;
    EdgeStructureStrictRejectionStats edge_structure_vertical_overlap_rejections;
    EdgeStructureStrictRejectionStats edge_structure_vertical_angle_rejections;
    double edge_structure_source_vertical_unmatched_actual_length = 0.0;
    double edge_structure_target_vertical_unmatched_actual_length = 0.0;
    double edge_structure_source_vertical_unmatched_length_ratio = -1.0;
    double edge_structure_target_vertical_unmatched_length_ratio = -1.0;
    int edge_structure_vertical_ambiguous_match_count = 0;
    double edge_structure_vertical_ambiguous_actual_length_ratio = -1.0;
    double edge_structure_vertical_matched_angle_difference_mean_degrees = -1.0;
    double edge_structure_vertical_matched_angle_difference_max_degrees = -1.0;
    // 未补偿绝对高度差诊断；当前来自归一化灰度差。
    int warp_height_diff_valid_count = 0;
    double warp_height_diff_overlap_ratio = -1.0;
    double warp_height_diff_mean = -1.0;
    double warp_height_diff_p50 = -1.0;
    double warp_height_diff_p75 = -1.0;
    double warp_height_diff_p90 = -1.0;
    double warp_height_diff_p95 = -1.0;
    double warp_height_diff_max = -1.0;
    // 原始高度差失败后使用全局补偿回退验证时的诊断。
    bool warp_height_diff_compensation_attempted = false;
    double warp_height_diff_global_offset = -1.0;
    double warp_height_diff_compensated_mean = -1.0;
    double warp_height_diff_compensated_p50 = -1.0;
    double warp_height_diff_compensated_p75 = -1.0;
    double warp_height_diff_compensated_p90 = -1.0;
    double warp_height_diff_compensated_p95 = -1.0;
    double warp_height_diff_compensated_max = -1.0;
    bool warp_height_diff_local_noise_candidate = false;
    double warp_height_diff_p90_p75_gap = -1.0;
    double warp_height_diff_local_noise_containment = -1.0;
    double structure_overlap_iou = -1.0;

    /// 直接法点特征初始化是否尝试过。
    bool feature_initializer_attempted = false;
    /// 直接法点特征初始化是否被最终选为输出结果来源。
    bool feature_initializer_used = false;
    /// 直接法最终判定实际采用的结果来源；例如 DIRECT / INITIALIZER。
    std::string final_validation_source;
    /// 被采用的点特征初始化方法名。
    std::string feature_initializer_method;
    /// 被采用初始值的内点数。
    int feature_initializer_inliers = 0;
    /// 被采用初始值的内点率。
    double feature_initializer_inlier_ratio = -1.0;
    /// 被采用初始值的内点空间覆盖率。
    double feature_initializer_spatial_coverage = -1.0;
    // 被采用初始值临时 warp 的未补偿绝对高度差诊断。
    int feature_initializer_warp_height_diff_valid_count = 0;
    double feature_initializer_warp_height_diff_overlap_ratio = -1.0;
    double feature_initializer_warp_height_diff_mean = -1.0;
    double feature_initializer_warp_height_diff_p50 = -1.0;
    double feature_initializer_warp_height_diff_p75 = -1.0;
    double feature_initializer_warp_height_diff_p90 = -1.0;
    double feature_initializer_warp_height_diff_p95 = -1.0;
    double feature_initializer_warp_height_diff_max = -1.0;

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
