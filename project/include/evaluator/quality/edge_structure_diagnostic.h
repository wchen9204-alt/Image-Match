#pragma once

#include <string>

#include <opencv2/core.hpp>

namespace ir::edge_structure_diagnostic {

/// 长条前景的线组结构验证参数。INSUFFICIENT 不阻断配准，明确 FAIL 参与最终判定。
struct Options {
    bool enabled = false;
    int visibility_threshold = 0;
    double min_foreground_elongation_ratio = 4.00;
    // 全部前景沿 PCA 主轴的最低连续占用率。
    double min_axis_occupancy = 0.75;
    // 各主轴截面中心到 PCA 中心线的 P90 偏差 / 前景长边上限。
    double max_centerline_deviation_ratio = 0.03;
    int max_canvas_side_pixels = 16384;
    int max_canvas_pixels = 100000000;

    // 同一物理边缘的重复线组去重容差；距离更大的两条边界仍分别保留。
    double duplicate_line_normal_tolerance_pixels = 2.0;
    double duplicate_line_min_span_overlap_ratio = 0.80;
    // 长条长边的双侧边界中，仅让更靠近前景外侧的一条进入严格匹配。
    // 内侧边仍保留在 filtered/fitted 图中，供诊断和后续厚度分析使用。
    bool prefer_outer_longitudinal_edges = true;
    // 竖直双边界中仅让共同参考系右侧的一条进入严格匹配。
    bool prefer_right_vertical_edges = true;
    double outer_longitudinal_edge_min_normal_separation_pixels = 1.0;
    double outer_longitudinal_edge_max_normal_separation_pixels = 12.0;
    double outer_longitudinal_edge_min_span_overlap_ratio = 0.75;
    // 初始片段固定由 EDLines 在共同可见区域的灰度图上检测。
    double min_fragment_length_pixels = 6.0;

    double group_max_angle_difference_degrees = 12.0;
    double group_max_normal_distance_pixels = 3.0;
    // 首次拟合后，切向不重叠的同轴线组允许重新分配的法向距离。
    double post_fit_group_normal_distance_pixels = 3.0;
    double min_line_group_actual_length_pixels = 6.0;
    double min_line_group_continuity_ratio = 0.10;
    double max_line_group_gap_ratio = 0.45;
    double max_fragment_direction_spread_degrees = 10.0;
    double max_line_fit_residual_pixels = 5.0;

    double min_main_line_actual_length_ratio = 0.30;
    double direction_cluster_tolerance_degrees = 6.0;
    double min_main_direction_support_ratio = 0.60;
    double max_main_direction_spread_degrees = 4.0;
    double min_main_direction_margin = 0.15;
    double max_main_direction_difference_degrees = 5.0;
    double max_axis_classification_error_degrees = 8.0;
    // 按同图、同方向最长有效线组的实际边缘长度过滤短线组。
    double min_horizontal_actual_length_ratio = 0.65;
    double min_vertical_actual_length_ratio = 0.45;
    // 竖直方向已有匹配但双方大段支撑均无法解释时，判为系统性结构冲突。
    double max_vertical_unmatched_length_ratio = 0.70;

    double profile_smoothing_sigma = 1.0;
    double min_peak_prominence = 0.03;
    double candidate_position_tolerance_pixels = 15.0;
    double final_position_tolerance_pixels = 3.0;
    double candidate_min_span_overlap_ratio = 0.05;
    double min_shorter_line_overlap_ratio = 0.90;
    double max_line_pair_angle_difference_degrees = 0.3;
    double match_position_cost_weight = 0.45;
    double match_overlap_cost_weight = 0.30;
    double match_angle_cost_weight = 0.20;
    double match_prominence_cost_weight = 0.05;
    double min_strong_line_actual_length_pixels = 12.0;
    double min_strong_peak_prominence = 0.05;
    double ambiguity_score_margin = 0.05;
};

/// 严格线对被拒绝的单项原因统计；同一线对可同时计入多个原因。
struct StrictRejectionStats {
    int count = 0;
    double source_actual_length = 0.0;
    double target_actual_length = 0.0;
};

/// 单个水平或竖直方向的线组匹配诊断。
struct DirectionResult {
    std::string status = "NOT_RUN";
    int source_eligible_line_groups = 0;
    int target_eligible_line_groups = 0;
    int candidate_pairs = 0;
    int accepted_matches = 0;
    double source_match_ratio = -1.0;
    double target_match_ratio = -1.0;
    double source_matched_actual_length = 0.0;
    double target_matched_actual_length = 0.0;
    int strong_conflict_count = 0;
    double source_strong_conflict_actual_length = 0.0;
    double target_strong_conflict_actual_length = 0.0;
    double source_strong_conflict_length_ratio = -1.0;
    double target_strong_conflict_length_ratio = -1.0;
    StrictRejectionStats strict_position_rejections;
    StrictRejectionStats strict_overlap_rejections;
    StrictRejectionStats strict_angle_rejections;
    double source_unmatched_actual_length = 0.0;
    double target_unmatched_actual_length = 0.0;
    double source_unmatched_length_ratio = -1.0;
    double target_unmatched_length_ratio = -1.0;
    int ambiguous_match_count = 0;
    double ambiguous_actual_length_ratio = -1.0;
    double matched_angle_difference_mean_degrees = -1.0;
    double matched_angle_difference_max_degrees = -1.0;
};

struct Result {
    std::string status = "NOT_RUN";
    std::string message;
    double source_foreground_elongation_ratio = -1.0;
    double target_foreground_elongation_ratio = -1.0;
    double source_axis_occupancy = -1.0;
    double target_axis_occupancy = -1.0;
    double source_centerline_deviation_ratio = -1.0;
    double target_centerline_deviation_ratio = -1.0;
    double source_foreground_long_side = -1.0;
    double target_foreground_long_side = -1.0;
    int common_canvas_width = 0;
    int common_canvas_height = 0;
    double common_canvas_offset_x = 0.0;
    double common_canvas_offset_y = 0.0;

    int source_visibility_pixels = 0;
    int target_visibility_pixels = 0;
    int common_visibility_pixels = 0;
    double visibility_area_ratio = -1.0;
    double visibility_overlap_containment = -1.0;
    double source_visibility_ratio = -1.0;
    double target_visibility_ratio = -1.0;

    int source_fragment_count = 0;
    int target_fragment_count = 0;
    int source_line_group_count = 0;
    int target_line_group_count = 0;
    int source_valid_line_group_count = 0;
    int target_valid_line_group_count = 0;
    int source_main_line_group_count = 0;
    int target_main_line_group_count = 0;

    bool source_main_direction_reliable = false;
    bool target_main_direction_reliable = false;
    double source_main_direction_degrees = -1.0;
    double target_main_direction_degrees = -1.0;
    double source_main_direction_support_ratio = -1.0;
    double target_main_direction_support_ratio = -1.0;
    double source_main_direction_spread_degrees = -1.0;
    double target_main_direction_spread_degrees = -1.0;
    double source_main_direction_margin = -1.0;
    double target_main_direction_margin = -1.0;
    double source_main_max_actual_length_ratio = -1.0;
    double target_main_max_actual_length_ratio = -1.0;
    double main_direction_difference_degrees = -1.0;
    double reference_direction_degrees = -1.0;

    DirectionResult horizontal;
    DirectionResult vertical;

    // 线段检测器直接输出的初始片段，尚未进行线组合并或任何过滤。
    cv::Mat initial_source_line_segments;
    cv::Mat initial_target_line_segments;
    // 最终筛选后保留的真实片段；不会补齐断裂。
    cv::Mat filtered_source_lines;
    cv::Mat filtered_target_lines;
    // 与 filtered 使用相同线组的最终拟合跨度图。
    cv::Mat fitted_source_lines;
    cv::Mat fitted_target_lines;
    cv::Mat matched_line_overlay;
};

bool evaluate(const Options& options,
              const cv::Mat& sourceImage,
              const cv::Mat& targetImage,
              Result& result);

} // namespace ir::edge_structure_diagnostic
