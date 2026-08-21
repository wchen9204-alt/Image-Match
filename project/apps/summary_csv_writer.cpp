#include "summary_csv_writer.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "utils/file_utils.h"

namespace ir::summary_csv {
namespace {

std::vector<std::string> collectMetricColumns(const std::vector<EvaluationData>& evaluations) {
    std::vector<std::string> columns;
    for (const auto& evaluation : evaluations) {
        for (const auto& metric : evaluation.metrics) {
            // 只有当前批次里至少出现过一次有效值的指标，才在 summary.csv 中保留整列。
            if (!metric.valid) {
                continue;
            }
            if (std::find(columns.begin(), columns.end(), metric.name) == columns.end()) {
                columns.push_back(metric.name);
            }
        }
    }
    return columns;
}

std::string metricCsvColumnName(const std::string& metric_name, MethodFamily family) {
    if (family == MethodFamily::DIRECT) {
        return "指标_" + metric_name;
    }
    return "metric_" + metric_name;
}

void appendMetricHeader(std::ostringstream& oss,
                        MethodFamily family,
                        const std::vector<std::string>& metric_columns) {
    for (const auto& metric_name : metric_columns) {
        oss << "," << metricCsvColumnName(metric_name, family);
    }
    oss << "\n";
}

void appendMetricValues(std::ostringstream& oss,
                        const std::vector<std::string>& metric_columns,
                        const EvaluationData* evaluation) {
    for (const auto& metric_name : metric_columns) {
        oss << ",";
        if (!evaluation) {
            continue;
        }
        const MetricResult* metric = evaluation->find(metric_name);
        if (metric && metric->valid) {
            oss << metric->value;
        }
    }
}

void appendCommonCsvPrefix(std::ostringstream& oss,
                           const std::string& sample_name,
                           const RegistrationResult& r) {
    oss << file_utils::csvEscape(sample_name) << "," << (r.success ? "1" : "0") << ","
        << file_utils::csvEscape(r.message) << ",";
}

void appendKeypointCsvHeader(std::ostringstream& oss) {
    oss << "sample_name,success,message,"
        << "num_keypoints_first,num_keypoints_second,"
        << "num_raw_matches,num_filtered_matches,num_inliers,"
        << "inlier_ratio,mean_reproj_error,warp_overlap_containment,"
        << "warp_source_coverage,warp_target_coverage,"
        << "warp_height_diff_valid_count,warp_height_diff_overlap_ratio,warp_height_diff_compensation_attempted,warp_height_diff_global_offset,warp_height_diff_mean,warp_height_diff_compensated_mean,warp_height_diff_p50,warp_height_diff_compensated_p50,warp_height_diff_p75,warp_height_diff_compensated_p75,warp_height_diff_p90,warp_height_diff_compensated_p90,warp_height_diff_p95,warp_height_diff_compensated_p95,warp_height_diff_max,warp_height_diff_compensated_max,"
        << "t_load_ms,t_extract_ms,t_match_ms,t_filter_ms,t_geometry_ms,t_warp_ms,t_total_ms";
}

void appendKeypointCsvRow(std::ostringstream& oss,
                          const std::string& sample_name,
                          const RegistrationResult& r) {
    appendCommonCsvPrefix(oss, sample_name, r);
    oss << r.num_keypoints_first << "," << r.num_keypoints_second << ","
        << r.num_raw_matches << "," << r.num_filtered_matches << "," << r.num_inliers << ","
        << r.inlier_ratio << "," << r.mean_reproj_error << ","
        << r.warp_overlap_containment << ","
        << r.warp_source_coverage << ","
        << r.warp_target_coverage << ","
        << r.warp_height_diff_valid_count << ","
        << r.warp_height_diff_overlap_ratio << ","
        << (r.warp_height_diff_compensation_attempted ? "1" : "0") << ","
        << r.warp_height_diff_global_offset << ","
        << r.warp_height_diff_mean << ","
        << r.warp_height_diff_compensated_mean << ","
        << r.warp_height_diff_p50 << ","
        << r.warp_height_diff_compensated_p50 << ","
        << r.warp_height_diff_p75 << ","
        << r.warp_height_diff_compensated_p75 << ","
        << r.warp_height_diff_p90 << ","
        << r.warp_height_diff_compensated_p90 << ","
        << r.warp_height_diff_p95 << ","
        << r.warp_height_diff_compensated_p95 << ","
        << r.warp_height_diff_max << ","
        << r.warp_height_diff_compensated_max << ","
        << r.t_load_ms << "," << r.t_extract_ms << "," << r.t_match_ms << ","
        << r.t_filter_ms << "," << r.t_geometry_ms << "," << r.t_warp_ms << ","
        << r.t_total_ms;
}

void appendStructureCsvHeader(std::ostringstream& oss) {
    oss << "sample_name,success,message,"
        << "num_structures_first,num_structures_second,"
        << "num_candidate_structure_matches,num_filtered_structure_matches,"
        << "num_inlier_structure_matches,structure_inlier_ratio,"
        << "mean_structure_reproj_error,warp_overlap_containment,"
        << "warp_source_coverage,warp_target_coverage,"
        << "warp_height_diff_valid_count,warp_height_diff_overlap_ratio,warp_height_diff_mean,warp_height_diff_p50,warp_height_diff_p75,warp_height_diff_p90,warp_height_diff_p95,warp_height_diff_max,"
        << "structure_overlap_iou,"
        << "t_load_ms,t_extract_ms,t_associate_ms,t_filter_ms,t_geometry_ms,t_warp_ms,t_total_ms";
}

void appendStructureCsvRow(std::ostringstream& oss,
                           const std::string& sample_name,
                           const RegistrationResult& r) {
    appendCommonCsvPrefix(oss, sample_name, r);
    oss << r.num_structures_first << "," << r.num_structures_second << ","
        << r.num_raw_matches << "," << r.num_filtered_matches << "," << r.num_inliers << ","
        << r.inlier_ratio << "," << r.mean_reproj_error << ","
        << r.warp_overlap_containment << "," << r.warp_source_coverage << ","
        << r.warp_target_coverage << ","
        << r.warp_height_diff_valid_count << ","
        << r.warp_height_diff_overlap_ratio << ","
        << r.warp_height_diff_mean << ","
        << r.warp_height_diff_p50 << ","
        << r.warp_height_diff_p75 << ","
        << r.warp_height_diff_p90 << ","
        << r.warp_height_diff_p95 << ","
        << r.warp_height_diff_max << ","
        << r.structure_overlap_iou << ","
        << r.t_load_ms << "," << r.t_extract_ms << "," << r.t_match_ms << ","
        << r.t_filter_ms << "," << r.t_geometry_ms << "," << r.t_warp_ms << ","
        << r.t_total_ms;
}

void appendDirectCsvHeader(std::ostringstream& oss) {
    // 直接法 summary.csv 只保留真正有判读价值的列：
    oss << "样本名,是否成功,结果说明,"
        << "直接法置信度,最终采用来源,"
        << "初始值内点数,"
        << "初始值内点率,初始值空间覆盖率,"
        << "初始值高度差有效点数,初始值高度差重叠比例,初始值高度差均值,初始值高度差P50,初始值高度差P75,初始值高度差P90,初始值高度差P95,初始值高度差最大值,"
        << "重叠包含率,"
        << "源图覆盖率,目标图覆盖率,"
        << "高度差有效点数,高度差重叠比例,是否尝试高度补偿,全局高度偏移,高度差均值,补偿后高度差均值,高度差P50,补偿后高度差P50,高度差P75,补偿后高度差P75,高度差P90,补偿后高度差P90,高度差P95,补偿后高度差P95,高度差最大值,补偿后高度差最大值,"
        << "加载耗时_ms,几何阶段耗时_ms,变换耗时_ms,总耗时_ms";
}

void appendDirectCsvRow(std::ostringstream& oss,
                        const std::string& sample_name,
                        const RegistrationResult& r) {
    appendCommonCsvPrefix(oss, sample_name, r);
    oss << r.direct_confidence << "," << file_utils::csvEscape(r.final_validation_source) << ","
        << r.feature_initializer_inliers << ","
        << r.feature_initializer_inlier_ratio << ","
        << r.feature_initializer_spatial_coverage << ","
        << r.feature_initializer_warp_height_diff_valid_count << ","
        << r.feature_initializer_warp_height_diff_overlap_ratio << ","
        << r.feature_initializer_warp_height_diff_mean << ","
        << r.feature_initializer_warp_height_diff_p50 << ","
        << r.feature_initializer_warp_height_diff_p75 << ","
        << r.feature_initializer_warp_height_diff_p90 << ","
        << r.feature_initializer_warp_height_diff_p95 << ","
        << r.feature_initializer_warp_height_diff_max << ","
        << r.warp_overlap_containment << ","
        << r.warp_source_coverage << "," << r.warp_target_coverage << ","
        << r.warp_height_diff_valid_count << ","
        << r.warp_height_diff_overlap_ratio << ","
        << (r.warp_height_diff_compensation_attempted ? "1" : "0") << ","
        << r.warp_height_diff_global_offset << ","
        << r.warp_height_diff_mean << ","
        << r.warp_height_diff_compensated_mean << ","
        << r.warp_height_diff_p50 << ","
        << r.warp_height_diff_compensated_p50 << ","
        << r.warp_height_diff_p75 << ","
        << r.warp_height_diff_compensated_p75 << ","
        << r.warp_height_diff_p90 << ","
        << r.warp_height_diff_compensated_p90 << ","
        << r.warp_height_diff_p95 << ","
        << r.warp_height_diff_compensated_p95 << ","
        << r.warp_height_diff_max << ","
        << r.warp_height_diff_compensated_max << ","
        << r.t_load_ms << "," << r.t_geometry_ms << "," << r.t_warp_ms << ","
        << r.t_total_ms;
}

void appendLearningCsvHeader(std::ostringstream& oss) {
    oss << "sample_name,success,message,"
        << "num_learning_points_first,num_learning_points_second,"
        << "num_raw_learning_matches,num_filtered_learning_matches,num_inlier_learning_matches,"
        << "learning_inlier_ratio,mean_reproj_error,warp_overlap_containment,"
        << "warp_source_coverage,warp_target_coverage,"
        << "warp_height_diff_valid_count,warp_height_diff_overlap_ratio,warp_height_diff_compensation_attempted,warp_height_diff_global_offset,warp_height_diff_mean,warp_height_diff_compensated_mean,warp_height_diff_p50,warp_height_diff_compensated_p50,warp_height_diff_p75,warp_height_diff_compensated_p75,warp_height_diff_p90,warp_height_diff_compensated_p90,warp_height_diff_p95,warp_height_diff_compensated_p95,warp_height_diff_max,warp_height_diff_compensated_max,"
        << "t_load_ms,t_extract_ms,t_match_ms,t_filter_ms,t_geometry_ms,t_warp_ms,t_total_ms";
}

void appendLearningCsvRow(std::ostringstream& oss,
                          const std::string& sample_name,
                          const RegistrationResult& r) {
    appendCommonCsvPrefix(oss, sample_name, r);
    oss << r.num_keypoints_first << "," << r.num_keypoints_second << ","
        << r.num_raw_matches << "," << r.num_filtered_matches << "," << r.num_inliers << ","
        << r.inlier_ratio << "," << r.mean_reproj_error << ","
        << r.warp_overlap_containment << ","
        << r.warp_source_coverage << ","
        << r.warp_target_coverage << ","
        << r.warp_height_diff_valid_count << ","
        << r.warp_height_diff_overlap_ratio << ","
        << (r.warp_height_diff_compensation_attempted ? "1" : "0") << ","
        << r.warp_height_diff_global_offset << ","
        << r.warp_height_diff_mean << ","
        << r.warp_height_diff_compensated_mean << ","
        << r.warp_height_diff_p50 << ","
        << r.warp_height_diff_compensated_p50 << ","
        << r.warp_height_diff_p75 << ","
        << r.warp_height_diff_compensated_p75 << ","
        << r.warp_height_diff_p90 << ","
        << r.warp_height_diff_compensated_p90 << ","
        << r.warp_height_diff_p95 << ","
        << r.warp_height_diff_compensated_p95 << ","
        << r.warp_height_diff_max << ","
        << r.warp_height_diff_compensated_max << ","
        << r.t_load_ms << "," << r.t_extract_ms << "," << r.t_match_ms << ","
        << r.t_filter_ms << "," << r.t_geometry_ms << "," << r.t_warp_ms << ","
        << r.t_total_ms;
}

void appendSummaryCsvHeader(std::ostringstream& oss,
                            MethodFamily family,
                            const std::vector<std::string>& metric_columns) {
    switch (family) {
    case MethodFamily::STRUCTURE:
        appendStructureCsvHeader(oss);
        break;
    case MethodFamily::DIRECT:
        appendDirectCsvHeader(oss);
        break;
    case MethodFamily::LEARNING:
        appendLearningCsvHeader(oss);
        break;
    case MethodFamily::KEYPOINT:
    default:
        appendKeypointCsvHeader(oss);
        break;
    }
    appendMetricHeader(oss, family, metric_columns);
}

void appendSummaryCsvRow(std::ostringstream& oss,
                         MethodFamily family,
                         const std::string& sample_name,
                         const RegistrationResult& r) {
    switch (family) {
    case MethodFamily::STRUCTURE:
        appendStructureCsvRow(oss, sample_name, r);
        break;
    case MethodFamily::DIRECT:
        appendDirectCsvRow(oss, sample_name, r);
        break;
    case MethodFamily::LEARNING:
        appendLearningCsvRow(oss, sample_name, r);
        break;
    case MethodFamily::KEYPOINT:
    default:
        appendKeypointCsvRow(oss, sample_name, r);
        break;
    }
}

// 根据当前实际写出的固定表头计算列数，避免新增字段后 AVERAGE 行错位到动态指标列。
size_t fixedSummaryColumnCount(MethodFamily family) {
    std::ostringstream header;
    switch (family) {
    case MethodFamily::STRUCTURE:
        appendStructureCsvHeader(header);
        break;
    case MethodFamily::DIRECT:
        appendDirectCsvHeader(header);
        break;
    case MethodFamily::LEARNING:
        appendLearningCsvHeader(header);
        break;
    case MethodFamily::KEYPOINT:
    default:
        appendKeypointCsvHeader(header);
        break;
    }
    const std::string headerText = header.str();
    return static_cast<size_t>(std::count(headerText.begin(), headerText.end(), ',')) + 1;
}

// 在总耗时列写入所有已写出样本的平均值，其他固定列和动态指标列保持为空。
void appendAverageTotalTimeRow(std::ostringstream& oss,
                               MethodFamily family,
                               const std::vector<std::string>& metric_columns,
                               const std::vector<RegistrationResult>& results,
                               size_t row_count) {
    if (row_count == 0) {
        return;
    }

    double total_time_ms = 0.0;
    for (size_t i = 0; i < row_count; ++i) {
        total_time_ms += results[i].t_total_ms;
    }

    oss << "AVERAGE";
    const size_t fixed_column_count = fixedSummaryColumnCount(family);
    for (size_t column = 2; column < fixed_column_count; ++column) {
        oss << ",";
    }
    oss << "," << std::fixed << std::setprecision(3)
        << total_time_ms / static_cast<double>(row_count);
    appendMetricValues(oss, metric_columns, nullptr);
    oss << "\n";
}

void writeHeightDifferenceSummary(const std::filesystem::path& csvPath,
                                  const std::vector<std::string>& sampleNames,
                                  const std::vector<RegistrationResult>& results) {
    std::ostringstream oss;
    oss << "sample_name,warp_height_diff_p95,warp_height_diff_compensated_p95,"
        << "warp_height_diff_p90,warp_height_diff_compensated_p90,"
        << "warp_height_diff_p75,warp_height_diff_compensated_p75,"
        << "warp_height_diff_p50,warp_height_diff_compensated_p50,"
        << "warp_height_diff_mean,warp_height_diff_compensated_mean,"
        << "warp_height_diff_max,warp_height_diff_compensated_max,"
        << "warp_height_diff_valid_count,warp_height_diff_overlap_ratio,"
        << "warp_height_diff_compensation_attempted,warp_height_diff_global_offset,"
        << "warp_height_diff_local_noise_candidate,warp_height_diff_p90_p75_gap,"
        << "warp_height_diff_local_noise_containment,"
        << "success,warp_overlap_containment,message\n";

    const size_t rowCount = std::min(sampleNames.size(), results.size());
    for (size_t i = 0; i < rowCount; ++i) {
        const auto& result = results[i];
        oss << file_utils::csvEscape(sampleNames[i]) << ","
            << result.warp_height_diff_p95 << ","
            << result.warp_height_diff_compensated_p95 << ","
            << result.warp_height_diff_p90 << ","
            << result.warp_height_diff_compensated_p90 << ","
            << result.warp_height_diff_p75 << ","
            << result.warp_height_diff_compensated_p75 << ","
            << result.warp_height_diff_p50 << ","
            << result.warp_height_diff_compensated_p50 << ","
            << result.warp_height_diff_mean << ","
            << result.warp_height_diff_compensated_mean << ","
            << result.warp_height_diff_max << ","
            << result.warp_height_diff_compensated_max << ","
            << result.warp_height_diff_valid_count << ","
            << result.warp_height_diff_overlap_ratio << ","
            << (result.warp_height_diff_compensation_attempted ? "1" : "0") << ","
            << result.warp_height_diff_global_offset << ","
            << (result.warp_height_diff_local_noise_candidate ? "1" : "0") << ","
            << result.warp_height_diff_p90_p75_gap << ","
            << result.warp_height_diff_local_noise_containment << ","
            << (result.success ? "1" : "0") << ","
            << result.warp_overlap_containment << ","
            << file_utils::csvEscape(result.message) << "\n";
    }

    file_utils::writeWholeFile(csvPath.parent_path() / "height_diff_summary.csv", oss.str());
}

void writeEdgeStructureSummary(const std::filesystem::path& csvPath,
                               const std::vector<std::string>& sampleNames,
                               const std::vector<RegistrationResult>& results) {
    std::ostringstream oss;
    oss << "sample_name,edge_structure_status,edge_structure_message,"
        << "source_foreground_elongation_ratio,target_foreground_elongation_ratio,"
        << "source_axis_occupancy,target_axis_occupancy,"
        << "source_centerline_deviation_ratio,target_centerline_deviation_ratio,"
        << "source_foreground_long_side,target_foreground_long_side,"
        << "common_canvas_width,common_canvas_height,common_canvas_offset_x,common_canvas_offset_y,"
        << "source_visibility_pixels,target_visibility_pixels,common_visibility_pixels,"
        << "visibility_area_ratio,visibility_overlap_containment,"
        << "source_visibility_ratio,target_visibility_ratio,ps,pt,"
        << "source_fragment_count,target_fragment_count,"
        << "source_line_group_count,target_line_group_count,"
        << "source_valid_line_group_count,target_valid_line_group_count,"
        << "source_main_line_group_count,target_main_line_group_count,"
        << "source_main_direction_reliable,target_main_direction_reliable,"
        << "source_main_direction_degrees,target_main_direction_degrees,"
        << "source_main_direction_support_ratio,target_main_direction_support_ratio,"
        << "source_main_direction_spread_degrees,target_main_direction_spread_degrees,"
        << "source_main_direction_margin,target_main_direction_margin,"
        << "source_main_max_actual_length_ratio,target_main_max_actual_length_ratio,"
        << "main_direction_difference_degrees,reference_direction_degrees,"
        << "horizontal_status,source_horizontal_eligible_line_groups,"
        << "target_horizontal_eligible_line_groups,horizontal_candidate_pairs,"
        << "horizontal_accepted_matches,horizontal_Hs,horizontal_Ht,"
        << "source_horizontal_matched_actual_length,target_horizontal_matched_actual_length,"
        << "horizontal_strong_conflict_count,"
        << "source_horizontal_strong_conflict_actual_length,"
        << "target_horizontal_strong_conflict_actual_length,"
        << "source_horizontal_strong_conflict_length_ratio,"
        << "target_horizontal_strong_conflict_length_ratio,"
        << "source_horizontal_unmatched_actual_length,target_horizontal_unmatched_actual_length,"
        << "source_horizontal_unmatched_length_ratio,target_horizontal_unmatched_length_ratio,"
        << "horizontal_ambiguous_match_count,horizontal_ambiguous_actual_length_ratio,"
        << "horizontal_matched_angle_difference_mean_degrees,"
        << "horizontal_matched_angle_difference_max_degrees,"
        << "vertical_status,source_vertical_eligible_line_groups,"
        << "target_vertical_eligible_line_groups,vertical_candidate_pairs,"
        << "vertical_accepted_matches,vertical_Vs,vertical_Vt,"
        << "source_vertical_matched_actual_length,target_vertical_matched_actual_length,"
        << "vertical_strong_conflict_count,"
        << "source_vertical_strong_conflict_actual_length,"
        << "target_vertical_strong_conflict_actual_length,"
        << "source_vertical_strong_conflict_length_ratio,"
        << "target_vertical_strong_conflict_length_ratio,"
        << "source_vertical_unmatched_actual_length,target_vertical_unmatched_actual_length,"
        << "source_vertical_unmatched_length_ratio,target_vertical_unmatched_length_ratio,"
        << "vertical_ambiguous_match_count,vertical_ambiguous_actual_length_ratio,"
        << "vertical_matched_angle_difference_mean_degrees,"
        << "vertical_matched_angle_difference_max_degrees,"
        << "registration_success,registration_message\n";

    const size_t rowCount = std::min(sampleNames.size(), results.size());
    for (size_t i = 0; i < rowCount; ++i) {
        const auto& result = results[i];
        oss << file_utils::csvEscape(sampleNames[i]) << ","
            << file_utils::csvEscape(result.edge_structure_status) << ","
            << file_utils::csvEscape(result.edge_structure_message) << ","
            << result.edge_structure_source_foreground_elongation_ratio << ","
            << result.edge_structure_target_foreground_elongation_ratio << ","
            << result.edge_structure_source_axis_occupancy << ","
            << result.edge_structure_target_axis_occupancy << ","
            << result.edge_structure_source_centerline_deviation_ratio << ","
            << result.edge_structure_target_centerline_deviation_ratio << ","
            << result.edge_structure_source_foreground_long_side << ","
            << result.edge_structure_target_foreground_long_side << ","
            << result.edge_structure_common_canvas_width << ","
            << result.edge_structure_common_canvas_height << ","
            << result.edge_structure_common_canvas_offset_x << ","
            << result.edge_structure_common_canvas_offset_y << ","
            << result.edge_structure_source_visibility_pixels << ","
            << result.edge_structure_target_visibility_pixels << ","
            << result.edge_structure_common_visibility_pixels << ","
            << result.edge_structure_visibility_area_ratio << ","
            << result.edge_structure_visibility_overlap_containment << ","
            << result.edge_structure_source_visibility_ratio << ","
            << result.edge_structure_target_visibility_ratio << ","
            << result.edge_structure_ps << ","
            << result.edge_structure_pt << ","
            << result.edge_structure_source_fragment_count << ","
            << result.edge_structure_target_fragment_count << ","
            << result.edge_structure_source_line_group_count << ","
            << result.edge_structure_target_line_group_count << ","
            << result.edge_structure_source_valid_line_group_count << ","
            << result.edge_structure_target_valid_line_group_count << ","
            << result.edge_structure_source_main_line_group_count << ","
            << result.edge_structure_target_main_line_group_count << ","
            << (result.edge_structure_source_main_direction_reliable ? "1" : "0") << ","
            << (result.edge_structure_target_main_direction_reliable ? "1" : "0") << ","
            << result.edge_structure_source_main_direction_degrees << ","
            << result.edge_structure_target_main_direction_degrees << ","
            << result.edge_structure_source_main_direction_support_ratio << ","
            << result.edge_structure_target_main_direction_support_ratio << ","
            << result.edge_structure_source_main_direction_spread_degrees << ","
            << result.edge_structure_target_main_direction_spread_degrees << ","
            << result.edge_structure_source_main_direction_margin << ","
            << result.edge_structure_target_main_direction_margin << ","
            << result.edge_structure_source_main_max_actual_length_ratio << ","
            << result.edge_structure_target_main_max_actual_length_ratio << ","
            << result.edge_structure_main_direction_difference_degrees << ","
            << result.edge_structure_reference_direction_degrees << ","
            << file_utils::csvEscape(result.edge_structure_horizontal_status) << ","
            << result.edge_structure_source_horizontal_eligible_line_groups << ","
            << result.edge_structure_target_horizontal_eligible_line_groups << ","
            << result.edge_structure_horizontal_candidate_pairs << ","
            << result.edge_structure_horizontal_accepted_matches << ","
            << result.edge_structure_source_horizontal_match_ratio << ","
            << result.edge_structure_target_horizontal_match_ratio << ","
            << result.edge_structure_source_horizontal_matched_actual_length << ","
            << result.edge_structure_target_horizontal_matched_actual_length << ","
            << result.edge_structure_horizontal_strong_conflict_count << ","
            << result.edge_structure_source_horizontal_strong_conflict_actual_length << ","
            << result.edge_structure_target_horizontal_strong_conflict_actual_length << ","
            << result.edge_structure_source_horizontal_strong_conflict_length_ratio << ","
            << result.edge_structure_target_horizontal_strong_conflict_length_ratio << ","
            << result.edge_structure_source_horizontal_unmatched_actual_length << ","
            << result.edge_structure_target_horizontal_unmatched_actual_length << ","
            << result.edge_structure_source_horizontal_unmatched_length_ratio << ","
            << result.edge_structure_target_horizontal_unmatched_length_ratio << ","
            << result.edge_structure_horizontal_ambiguous_match_count << ","
            << result.edge_structure_horizontal_ambiguous_actual_length_ratio << ","
            << result.edge_structure_horizontal_matched_angle_difference_mean_degrees << ","
            << result.edge_structure_horizontal_matched_angle_difference_max_degrees << ","
            << file_utils::csvEscape(result.edge_structure_vertical_status) << ","
            << result.edge_structure_source_vertical_eligible_line_groups << ","
            << result.edge_structure_target_vertical_eligible_line_groups << ","
            << result.edge_structure_vertical_candidate_pairs << ","
            << result.edge_structure_vertical_accepted_matches << ","
            << result.edge_structure_source_vertical_match_ratio << ","
            << result.edge_structure_target_vertical_match_ratio << ","
            << result.edge_structure_source_vertical_matched_actual_length << ","
            << result.edge_structure_target_vertical_matched_actual_length << ","
            << result.edge_structure_vertical_strong_conflict_count << ","
            << result.edge_structure_source_vertical_strong_conflict_actual_length << ","
            << result.edge_structure_target_vertical_strong_conflict_actual_length << ","
            << result.edge_structure_source_vertical_strong_conflict_length_ratio << ","
            << result.edge_structure_target_vertical_strong_conflict_length_ratio << ","
            << result.edge_structure_source_vertical_unmatched_actual_length << ","
            << result.edge_structure_target_vertical_unmatched_actual_length << ","
            << result.edge_structure_source_vertical_unmatched_length_ratio << ","
            << result.edge_structure_target_vertical_unmatched_length_ratio << ","
            << result.edge_structure_vertical_ambiguous_match_count << ","
            << result.edge_structure_vertical_ambiguous_actual_length_ratio << ","
            << result.edge_structure_vertical_matched_angle_difference_mean_degrees << ","
            << result.edge_structure_vertical_matched_angle_difference_max_degrees << ","
            << (result.success ? "1" : "0") << ","
            << file_utils::csvEscape(result.message) << "\n";
    }

    file_utils::writeWholeFile(csvPath.parent_path() / "edge_structure_summary.csv", oss.str());
}

} // namespace

void write(const std::filesystem::path& csv_path,
           MethodFamily family,
           const std::vector<std::string>& sample_names,
           const std::vector<RegistrationResult>& results,
           const std::vector<EvaluationData>& evaluations) {
    const std::vector<std::string> metric_columns = collectMetricColumns(evaluations);
    std::ostringstream oss;
    appendSummaryCsvHeader(oss, family, metric_columns);

    const size_t row_count = std::min(sample_names.size(), results.size());
    for (size_t i = 0; i < row_count; ++i) {
        const auto& r = results[i];
        appendSummaryCsvRow(oss, family, sample_names[i], r);
        appendMetricValues(oss, metric_columns, i < evaluations.size() ? &evaluations[i] : nullptr);
        oss << "\n";
    }

    appendAverageTotalTimeRow(oss, family, metric_columns, results, row_count);

    file_utils::writeWholeFile(csv_path, oss.str());
    writeHeightDifferenceSummary(csv_path, sample_names, results);
    writeEdgeStructureSummary(csv_path, sample_names, results);
}

} // namespace ir::summary_csv
