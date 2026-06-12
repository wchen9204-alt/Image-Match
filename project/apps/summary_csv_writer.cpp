#include "summary_csv_writer.h"

#include <algorithm>
#include <sstream>

#include "utils/file_utils.h"

namespace ir::summary_csv {
namespace {

std::vector<std::string> collectMetricColumns(const std::vector<EvaluationData>& evaluations) {
    std::vector<std::string> columns;
    for (const auto& evaluation : evaluations) {
        for (const auto& metric : evaluation.metrics) {
            if (std::find(columns.begin(), columns.end(), metric.name) == columns.end()) {
                columns.push_back(metric.name);
            }
        }
    }
    return columns;
}

std::string metricCsvColumnName(const std::string& metric_name) {
    return "metric_" + metric_name;
}

void appendMetricHeader(std::ostringstream& oss, const std::vector<std::string>& metric_columns) {
    for (const auto& metric_name : metric_columns) {
        oss << "," << metricCsvColumnName(metric_name);
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
        << "inlier_ratio,mean_reproj_error,warp_overlap_iou,warp_photometric_error,"
        << "t_load_ms,t_extract_ms,t_match_ms,t_filter_ms,t_geometry_ms,t_warp_ms,t_total_ms";
}

void appendKeypointCsvRow(std::ostringstream& oss,
                          const std::string& sample_name,
                          const RegistrationResult& r) {
    appendCommonCsvPrefix(oss, sample_name, r);
    oss << r.num_keypoints_first << "," << r.num_keypoints_second << ","
        << r.num_raw_matches << "," << r.num_filtered_matches << "," << r.num_inliers << ","
        << r.inlier_ratio << "," << r.mean_reproj_error << "," << r.warp_overlap_iou << ","
        << r.warp_photometric_error << ","
        << r.t_load_ms << "," << r.t_extract_ms << "," << r.t_match_ms << ","
        << r.t_filter_ms << "," << r.t_geometry_ms << "," << r.t_warp_ms << ","
        << r.t_total_ms;
}

void appendStructureCsvHeader(std::ostringstream& oss) {
    oss << "sample_name,success,message,"
        << "num_structures_first,num_structures_second,"
        << "num_candidate_structure_matches,num_filtered_structure_matches,"
        << "num_inlier_structure_matches,structure_inlier_ratio,"
        << "mean_structure_reproj_error,warp_overlap_iou,warp_photometric_error,"
        << "structure_overlap_iou,"
        << "t_load_ms,t_extract_ms,t_associate_ms,t_filter_ms,t_geometry_ms,t_warp_ms,t_total_ms";
}

void appendStructureCsvRow(std::ostringstream& oss,
                           const std::string& sample_name,
                           const RegistrationResult& r) {
    appendCommonCsvPrefix(oss, sample_name, r);
    oss << r.num_structures_first << "," << r.num_structures_second << ","
        << r.num_raw_matches << "," << r.num_filtered_matches << "," << r.num_inliers << ","
        << r.inlier_ratio << "," << r.mean_reproj_error << "," << r.warp_overlap_iou << ","
        << r.warp_photometric_error << "," << r.structure_overlap_iou << ","
        << r.t_load_ms << "," << r.t_extract_ms << "," << r.t_match_ms << ","
        << r.t_filter_ms << "," << r.t_geometry_ms << "," << r.t_warp_ms << ","
        << r.t_total_ms;
}

void appendDirectCsvHeader(std::ostringstream& oss) {
    oss << "sample_name,success,message,"
        << "num_correspondences,num_inlier_correspondences,direct_confidence,"
        << "mean_reproj_error,warp_overlap_iou,warp_photometric_error,"
        << "t_load_ms,t_align_ms,t_geometry_ms,t_warp_ms,t_total_ms";
}

void appendDirectCsvRow(std::ostringstream& oss,
                        const std::string& sample_name,
                        const RegistrationResult& r) {
    appendCommonCsvPrefix(oss, sample_name, r);
    oss << r.num_raw_matches << "," << r.num_inliers << "," << r.inlier_ratio << ","
        << r.mean_reproj_error << "," << r.warp_overlap_iou << ","
        << r.warp_photometric_error << ","
        << r.t_load_ms << "," << r.t_match_ms << "," << r.t_geometry_ms << ","
        << r.t_warp_ms << "," << r.t_total_ms;
}

void appendLearningCsvHeader(std::ostringstream& oss) {
    oss << "sample_name,success,message,"
        << "num_learning_points_first,num_learning_points_second,"
        << "num_raw_learning_matches,num_filtered_learning_matches,num_inlier_learning_matches,"
        << "learning_inlier_ratio,mean_reproj_error,warp_overlap_iou,warp_photometric_error,"
        << "t_load_ms,t_extract_ms,t_match_ms,t_filter_ms,t_geometry_ms,t_warp_ms,t_total_ms";
}

void appendLearningCsvRow(std::ostringstream& oss,
                          const std::string& sample_name,
                          const RegistrationResult& r) {
    appendCommonCsvPrefix(oss, sample_name, r);
    oss << r.num_keypoints_first << "," << r.num_keypoints_second << ","
        << r.num_raw_matches << "," << r.num_filtered_matches << "," << r.num_inliers << ","
        << r.inlier_ratio << "," << r.mean_reproj_error << "," << r.warp_overlap_iou << ","
        << r.warp_photometric_error << ","
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
    appendMetricHeader(oss, metric_columns);
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

} // namespace

void write(const std::filesystem::path& csv_path,
           MethodFamily family,
           const std::vector<std::string>& sample_names,
           const std::vector<RegistrationResult>& results,
           const std::vector<EvaluationData>& evaluations) {
    const std::vector<std::string> metric_columns = collectMetricColumns(evaluations);
    std::ostringstream oss;
    appendSummaryCsvHeader(oss, family, metric_columns);

    for (size_t i = 0; i < sample_names.size() && i < results.size(); ++i) {
        const auto& r = results[i];
        appendSummaryCsvRow(oss, family, sample_names[i], r);
        appendMetricValues(oss, metric_columns, i < evaluations.size() ? &evaluations[i] : nullptr);
        oss << "\n";
    }

    file_utils::writeWholeFile(csv_path, oss.str());
}

} // namespace ir::summary_csv
