#include "registration_app_helpers.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include "utils/file_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace fs = std::filesystem;

namespace ir::registration_app_helpers {
namespace {

namespace summary_text {

std::string formatMilliseconds(double v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << v << " ms";
    return oss.str();
}

void appendTiming(std::ostringstream& oss, const RegistrationResult& r) {
    oss << "  -- timings --\n";
    oss << "  load          : " << formatMilliseconds(r.t_load_ms) << "\n";
    oss << "  extract       : " << formatMilliseconds(r.t_extract_ms) << "\n";
    oss << "  match         : " << formatMilliseconds(r.t_match_ms) << "\n";
    oss << "  filter        : " << formatMilliseconds(r.t_filter_ms) << "\n";
    oss << "  geometry      : " << formatMilliseconds(r.t_geometry_ms) << "\n";
    oss << "  warp          : " << formatMilliseconds(r.t_warp_ms) << "\n";
    oss << "  TOTAL         : " << formatMilliseconds(r.t_total_ms) << "\n";
}

void appendEvaluation(std::ostringstream& oss, const EvaluationData& evaluation) {
    if (evaluation.metrics.empty()) {
        return;
    }

    oss << "  -- metrics --\n";
    for (const auto& metric : evaluation.metrics) {
        oss << "  " << metric.name << " : ";
        if (metric.valid) {
            oss << std::fixed << std::setprecision(6) << metric.value;
        } else {
            oss << "N/A";
        }
        if (!metric.note.empty()) {
            oss << " (" << metric.note << ")";
        }
        oss << "\n";
    }
}

void appendDirectDiagnostics(std::ostringstream& oss, const DirectData& direct) {
    if (direct.diagnostics.empty()) {
        return;
    }

    for (const auto& item : direct.diagnostics) {
        if (!std::isfinite(item.value)) {
            continue;
        }
        const std::string label = item.label.empty() ? item.key : item.label;
        oss << "  " << label << " : " << std::fixed << std::setprecision(3)
            << item.value << "\n";
    }
}

void appendValidationQuality(std::ostringstream& oss, const RegistrationResult& r) {
    if (r.inlier_spatial_coverage >= 0.0) {
        oss << "  match spread  : " << std::fixed << std::setprecision(3)
            << r.inlier_spatial_coverage << "\n";
    }
    if (r.structure_overlap_iou >= 0.0) {
        oss << "  structure IoU : " << std::fixed << std::setprecision(3)
            << r.structure_overlap_iou << "\n";
    }
    if (r.edge_structure_status != "NOT_RUN") {
        oss << "  edge structure: " << r.edge_structure_status
            << " (elongation=" << std::fixed << std::setprecision(3)
            << r.edge_structure_source_foreground_elongation_ratio << "/"
            << r.edge_structure_target_foreground_elongation_ratio
            << ", occupancy="
            << r.edge_structure_source_axis_occupancy << "/"
            << r.edge_structure_target_axis_occupancy
            << ", centerline="
            << r.edge_structure_source_centerline_deviation_ratio << "/"
            << r.edge_structure_target_centerline_deviation_ratio
            << ", area="
            << r.edge_structure_visibility_area_ratio
            << ", overlap=" << r.edge_structure_visibility_overlap_containment
            << ", Ps=" << r.edge_structure_ps
            << ", Pt=" << r.edge_structure_pt
            << ")\n";
        oss << "  edge main dir : source="
            << (r.edge_structure_source_main_direction_reliable ? "reliable" : "unreliable")
            << "@" << r.edge_structure_source_main_direction_degrees
            << ", target="
            << (r.edge_structure_target_main_direction_reliable ? "reliable" : "unreliable")
            << "@" << r.edge_structure_target_main_direction_degrees
            << ", diff=" << r.edge_structure_main_direction_difference_degrees << "\n";
        oss << "  edge horizontal: " << r.edge_structure_horizontal_status
            << " (Hs=" << r.edge_structure_source_horizontal_match_ratio
            << ", Ht=" << r.edge_structure_target_horizontal_match_ratio
            << ", accepted=" << r.edge_structure_horizontal_accepted_matches
            << "/" << r.edge_structure_horizontal_candidate_pairs
            << ", conflict="
            << r.edge_structure_source_horizontal_strong_conflict_length_ratio << "/"
            << r.edge_structure_target_horizontal_strong_conflict_length_ratio
            << ", unmatched="
            << r.edge_structure_source_horizontal_unmatched_length_ratio << "/"
            << r.edge_structure_target_horizontal_unmatched_length_ratio << ")\n";
        oss << "  edge vertical : " << r.edge_structure_vertical_status
            << " (Vs=" << r.edge_structure_source_vertical_match_ratio
            << ", Vt=" << r.edge_structure_target_vertical_match_ratio
            << ", accepted=" << r.edge_structure_vertical_accepted_matches
            << "/" << r.edge_structure_vertical_candidate_pairs
            << ", conflict="
            << r.edge_structure_source_vertical_strong_conflict_length_ratio << "/"
            << r.edge_structure_target_vertical_strong_conflict_length_ratio
            << ", unmatched="
            << r.edge_structure_source_vertical_unmatched_length_ratio << "/"
            << r.edge_structure_target_vertical_unmatched_length_ratio << ")\n";
    }
}

void appendHeightDifferenceStats(std::ostringstream& oss,
                                 const std::string& label,
                                 int validCount,
                                 double overlapRatio,
                                 double mean,
                                 double p50,
                                 double p75,
                                 double p90,
                                 double p95,
                                 double maxValue) {
    if (mean < 0.0) {
        return;
    }
    oss << "  " << label << " n   : " << validCount;
    if (overlapRatio >= 0.0) {
        oss << " (overlap=" << std::fixed << std::setprecision(3)
            << overlapRatio << ")";
    }
    oss << "\n";
    oss << "  " << label << " diff: mean=" << std::fixed << std::setprecision(4)
        << mean << ", P50=" << p50 << ", P75=" << p75 << ", P90=" << p90
        << ", P95=" << p95 << ", max=" << maxValue << "\n";
}

void appendWarpHeightDifferenceStats(std::ostringstream& oss, const RegistrationResult& r) {
    appendHeightDifferenceStats(oss,
                                "height",
                                r.warp_height_diff_valid_count,
                                r.warp_height_diff_overlap_ratio,
                                r.warp_height_diff_mean,
                                r.warp_height_diff_p50,
                                r.warp_height_diff_p75,
                                r.warp_height_diff_p90,
                                r.warp_height_diff_p95,
                                r.warp_height_diff_max);
}

void appendInitializerHeightDifferenceStats(std::ostringstream& oss, const RegistrationResult& r) {
    appendHeightDifferenceStats(oss,
                                "init height",
                                r.feature_initializer_warp_height_diff_valid_count,
                                r.feature_initializer_warp_height_diff_overlap_ratio,
                                r.feature_initializer_warp_height_diff_mean,
                                r.feature_initializer_warp_height_diff_p50,
                                r.feature_initializer_warp_height_diff_p75,
                                r.feature_initializer_warp_height_diff_p90,
                                r.feature_initializer_warp_height_diff_p95,
                                r.feature_initializer_warp_height_diff_max);
}
} // namespace summary_text

namespace json_output {

std::string escapeString(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out.push_back(c);
            break;
        }
    }
    return out;
}

void appendDirectDiagnostics(std::ostringstream& oss, const DirectData& direct) {
    oss << "    \"direct_diagnostics\": {";
    bool wroteAny = false;
    for (const auto& item : direct.diagnostics) {
        if (item.key.empty() || !std::isfinite(item.value)) {
            continue;
        }
        oss << (wroteAny ? ",\n" : "\n");
        oss << "      \"" << escapeString(item.key) << "\": " << item.value;
        wroteAny = true;
    }
    if (wroteAny) {
        oss << "\n    }\n";
    } else {
        oss << "}\n";
    }
}

} // namespace json_output

double matAtAsDouble(const cv::Mat& mat, int row, int col) {
    if (mat.empty() || row < 0 || col < 0 || row >= mat.rows || col >= mat.cols) {
        return 0.0;
    }

    cv::Mat value64;
    mat(cv::Rect(col, row, 1, 1)).convertTo(value64, CV_64F);
    return value64.at<double>(0, 0);
}

std::vector<std::string> readPatternCandidates(const YAML::Node& node,
                                               const std::string& listKey,
                                               const std::string& scalarKey,
                                               const std::vector<std::string>& fallback) {
    std::vector<std::string> values = yaml_utils::getVec<std::string>(node, listKey, {});
    if (!values.empty()) {
        return values;
    }

    const std::string scalar = yaml_utils::getString(node, scalarKey);
    if (!scalar.empty()) {
        return {scalar};
    }
    return fallback;
}

void applyVisualizationOverrides(PipelineConfig& pipelineCfg, const YAML::Node& node) {
    Config::applyVisualizationOverrides(pipelineCfg, node);
}

std::string buildKeypointSummaryText(const RegistrationContext& ctx) {
    const auto& r = ctx.result;
    std::ostringstream oss;
    oss << "\n================ Keypoint registration summary ================\n";
    oss << "  status        : " << (r.success ? "OK" : "FAILED") << "\n";
    oss << "  message       : " << r.message << "\n";
    oss << "  keypoints     : " << r.num_keypoints_first << " / " << r.num_keypoints_second
        << "\n";
    oss << "  raw matches   : " << r.num_raw_matches << "\n";
    oss << "  filtered      : " << r.num_filtered_matches << "\n";
    oss << "  inliers       : " << r.num_inliers << " (" << std::fixed << std::setprecision(3)
        << r.inlier_ratio << ")\n";
    if (r.warp_overlap_containment >= 0.0) {
        oss << "  warp contain  : " << std::fixed << std::setprecision(3)
            << r.warp_overlap_containment << "\n";
    }
    if (r.warp_source_coverage >= 0.0) {
        oss << "  warp source   : " << std::fixed << std::setprecision(3)
            << r.warp_source_coverage << "\n";
    }
    if (r.warp_target_coverage >= 0.0) {
        oss << "  warp target   : " << std::fixed << std::setprecision(3)
            << r.warp_target_coverage << "\n";
    }
    summary_text::appendWarpHeightDifferenceStats(oss, r);
    summary_text::appendValidationQuality(oss, r);
    summary_text::appendTiming(oss, r);
    summary_text::appendEvaluation(oss, ctx.evaluation);
    oss << "==============================================================\n";
    return oss.str();
}

std::string buildLearningSummaryText(const RegistrationContext& ctx) {
    const auto& r = ctx.result;
    std::ostringstream oss;
    oss << "\n================ Learning registration summary ================\n";
    oss << "  status        : " << (r.success ? "OK" : "FAILED") << "\n";
    oss << "  message       : " << r.message << "\n";
    oss << "  keypoints     : " << r.num_keypoints_first << " / " << r.num_keypoints_second
        << "\n";
    oss << "  raw matches   : " << r.num_raw_matches << "\n";
    oss << "  filtered      : " << r.num_filtered_matches << "\n";
    oss << "  inliers       : " << r.num_inliers << " (" << std::fixed << std::setprecision(3)
        << r.inlier_ratio << ")\n";
    if (r.warp_overlap_containment >= 0.0) {
        oss << "  warp contain  : " << std::fixed << std::setprecision(3)
            << r.warp_overlap_containment << "\n";
    }
    if (r.warp_source_coverage >= 0.0) {
        oss << "  warp source   : " << std::fixed << std::setprecision(3)
            << r.warp_source_coverage << "\n";
    }
    if (r.warp_target_coverage >= 0.0) {
        oss << "  warp target   : " << std::fixed << std::setprecision(3)
            << r.warp_target_coverage << "\n";
    }
    summary_text::appendWarpHeightDifferenceStats(oss, r);
    summary_text::appendValidationQuality(oss, r);
    summary_text::appendTiming(oss, r);
    summary_text::appendEvaluation(oss, ctx.evaluation);
    oss << "==============================================================\n";
    return oss.str();
}

std::string buildStructureSummaryText(const RegistrationContext& ctx) {
    const auto& r = ctx.result;
    const auto& gd = ctx.geometry_data;
    const auto& smd = ctx.structure_match_data;
    std::ostringstream oss;
    oss << "\n================ Structure registration summary ================\n";
    oss << "  status        : " << (r.success ? "OK" : "FAILED") << "\n";
    oss << "  message       : " << r.message << "\n";
    oss << "  structure type: " << toString(ctx.structure_data.type) << "\n";
    oss << "  structures    : " << r.num_structures_first << " / " << r.num_structures_second
        << "\n";
    if (gd.valid && !gd.A.empty() && gd.A.rows >= 2 && gd.A.cols >= 3) {
        oss << "  translation   : dx=" << std::fixed << std::setprecision(3)
            << gd.A.at<double>(0, 2) << ", dy=" << gd.A.at<double>(1, 2) << "\n";
    } else if (!smd.affine.empty() && smd.affine.rows >= 2 && smd.affine.cols >= 3) {
        oss << "  translation   : dx=" << std::fixed << std::setprecision(3)
            << smd.affine.at<double>(0, 2) << ", dy=" << smd.affine.at<double>(1, 2) << "\n";
    } else {
        oss << "  translation   : dx=" << std::fixed << std::setprecision(3)
            << smd.translation.x << ", dy=" << smd.translation.y << "\n";
    }
    oss << "  response      : " << std::fixed << std::setprecision(3) << r.inlier_ratio << "\n";
    if (r.warp_overlap_containment >= 0.0) {
        oss << "  warp contain  : " << std::fixed << std::setprecision(3)
            << r.warp_overlap_containment << "\n";
    }
    if (r.warp_source_coverage >= 0.0) {
        oss << "  warp source   : " << std::fixed << std::setprecision(3)
            << r.warp_source_coverage << "\n";
    }
    if (r.warp_target_coverage >= 0.0) {
        oss << "  warp target   : " << std::fixed << std::setprecision(3)
            << r.warp_target_coverage << "\n";
    }
    summary_text::appendWarpHeightDifferenceStats(oss, r);
    summary_text::appendValidationQuality(oss, r);
    summary_text::appendTiming(oss, r);
    summary_text::appendEvaluation(oss, ctx.evaluation);
    oss << "================================================================\n";
    return oss.str();
}

std::string buildDirectSummaryText(const RegistrationContext& ctx) {
    const auto& r = ctx.result;
    const auto& gd = ctx.geometry_data;
    const auto& dd = ctx.direct_data;
    std::ostringstream oss;
    oss << "\n================ Direct registration summary ================\n";
    oss << "  status        : " << (r.success ? "OK" : "FAILED") << "\n";
    oss << "  message       : " << r.message << "\n";
    oss << "  method        : " << (dd.method.empty() ? "DIRECT" : dd.method) << "\n";
    oss << "  geometry      : " << toString(gd.type) << "\n";

    if (gd.valid && !gd.A.empty() && gd.A.rows >= 2 && gd.A.cols >= 3) {
        const double a00 = matAtAsDouble(gd.A, 0, 0);
        const double a10 = matAtAsDouble(gd.A, 1, 0);
        const double tx = matAtAsDouble(gd.A, 0, 2);
        const double ty = matAtAsDouble(gd.A, 1, 2);
        const double rotationDeg = std::atan2(a10, a00) * 180.0 / 3.14159265358979323846;
        oss << "  translation   : dx=" << std::fixed << std::setprecision(3) << tx
            << ", dy=" << ty << "\n";
        if (gd.type == GeometryType::RIGID || gd.type == GeometryType::SIMILARITY ||
            gd.type == GeometryType::AFFINE) {
            oss << "  rotation      : " << std::fixed << std::setprecision(3)
                << rotationDeg << " deg\n";
        }
    }

    oss << "  correspondences: " << r.num_raw_matches << "\n";
    oss << "  confidence    : " << std::fixed << std::setprecision(3) << r.direct_confidence
        << "\n";
    if (!r.final_validation_source.empty()) {
        oss << "  final source  : " << r.final_validation_source << "\n";
    }
    if (r.feature_initializer_attempted) {
        oss << "  feature init  : " << (r.feature_initializer_used ? "USED" : "SKIPPED");
        if (!r.feature_initializer_method.empty()) {
            oss << " (" << r.feature_initializer_method << ")";
        }
        oss << "\n";
        if (r.feature_initializer_used) {
            oss << "  init inliers  : " << r.feature_initializer_inliers << " ("
                << std::fixed << std::setprecision(3)
                << r.feature_initializer_inlier_ratio << ")\n";
            if (r.feature_initializer_spatial_coverage >= 0.0) {
                oss << "  init spread   : " << std::fixed << std::setprecision(3)
                    << r.feature_initializer_spatial_coverage << "\n";
            }
            summary_text::appendInitializerHeightDifferenceStats(oss, r);
        }
    }
    summary_text::appendDirectDiagnostics(oss, dd);
    if (dd.photometric_error >= 0.0) {
        oss << "  direct residual MSE: " << std::fixed << std::setprecision(6)
            << dd.photometric_error << "\n";
    }
    if (r.warp_overlap_containment >= 0.0) {
        oss << "  warp contain  : " << std::fixed << std::setprecision(3)
            << r.warp_overlap_containment << "\n";
    }
    if (r.warp_source_coverage >= 0.0) {
        oss << "  warp source   : " << std::fixed << std::setprecision(3)
            << r.warp_source_coverage << "\n";
    }
    if (r.warp_target_coverage >= 0.0) {
        oss << "  warp target   : " << std::fixed << std::setprecision(3)
            << r.warp_target_coverage << "\n";
    }
    summary_text::appendWarpHeightDifferenceStats(oss, r);
    summary_text::appendValidationQuality(oss, r);
    summary_text::appendTiming(oss, r);
    summary_text::appendEvaluation(oss, ctx.evaluation);
    oss << "=============================================================\n";
    return oss.str();
}

std::string buildSummaryText(const RegistrationContext& ctx, MethodFamily family) {
    if (family == MethodFamily::STRUCTURE)
        return buildStructureSummaryText(ctx);
    if (family == MethodFamily::DIRECT)
        return buildDirectSummaryText(ctx);
    if (family == MethodFamily::LEARNING)
        return buildLearningSummaryText(ctx);
    return buildKeypointSummaryText(ctx);
}

std::string buildSummaryJson(const RegistrationContext& ctx,
                             const PipelineConfig& cfg,
                             const std::string& sample_name) {
    const auto& r = ctx.result;
    const auto family = cfg.methodFamily();
    const bool isStructure = (family == MethodFamily::STRUCTURE);
    const bool isDirect = (family == MethodFamily::DIRECT);
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"pipeline_name\": \"" << json_output::escapeString(cfg.name) << "\",\n";
    oss << "  \"method_family\": \"" << methodFamilyDir(family) << "\",\n";
    oss << "  \"sample_name\": \"" << json_output::escapeString(sample_name) << "\",\n";
    oss << "  \"status\": \"" << (r.success ? "OK" : "FAILED") << "\",\n";
    oss << "  \"message\": \"" << json_output::escapeString(r.message) << "\",\n";
    oss << "  \"image1_path\": \"" << json_output::escapeString(ctx.image1_path.string()) << "\",\n";
    oss << "  \"image2_path\": \"" << json_output::escapeString(ctx.image2_path.string()) << "\",\n";
    oss << "  \"counts\": {\n";
    if (isStructure) {
        oss << "    \"num_structures_first\": " << r.num_structures_first << ",\n";
        oss << "    \"num_structures_second\": " << r.num_structures_second << ",\n";
        oss << "    \"num_raw_matches\": " << r.num_raw_matches << ",\n";
        oss << "    \"num_filtered_matches\": " << r.num_filtered_matches << ",\n";
        oss << "    \"num_inliers\": " << r.num_inliers << "\n";
    } else if (isDirect) {
        oss << "    \"num_correspondences\": " << r.num_raw_matches << ",\n";
        oss << "    \"direct_confidence\": " << r.direct_confidence << ",\n";
        oss << "    \"final_validation_source\": \""
            << json_output::escapeString(r.final_validation_source) << "\",\n";
        oss << "    \"feature_initializer_attempted\": "
            << (r.feature_initializer_attempted ? "true" : "false") << ",\n";
        oss << "    \"feature_initializer_used\": "
            << (r.feature_initializer_used ? "true" : "false") << ",\n";
        oss << "    \"feature_initializer_method\": \""
            << json_output::escapeString(r.feature_initializer_method) << "\"\n";
    } else {
        oss << "    \"num_keypoints_first\": " << r.num_keypoints_first << ",\n";
        oss << "    \"num_keypoints_second\": " << r.num_keypoints_second << ",\n";
        oss << "    \"num_raw_matches\": " << r.num_raw_matches << ",\n";
        oss << "    \"num_filtered_matches\": " << r.num_filtered_matches << ",\n";
        oss << "    \"num_inliers\": " << r.num_inliers << "\n";
    }
    oss << "  },\n";
    oss << "  \"quality\": {\n";
    if (isDirect) {
        oss << "    \"feature_initializer_inliers\": "
            << r.feature_initializer_inliers << ",\n";
        oss << "    \"feature_initializer_inlier_ratio\": "
            << r.feature_initializer_inlier_ratio << ",\n";
        oss << "    \"feature_initializer_spatial_coverage\": "
            << r.feature_initializer_spatial_coverage << ",\n";
        oss << "    \"feature_initializer_warp_height_diff_valid_count\": "
            << r.feature_initializer_warp_height_diff_valid_count << ",\n";
        oss << "    \"feature_initializer_warp_height_diff_overlap_ratio\": "
            << r.feature_initializer_warp_height_diff_overlap_ratio << ",\n";
        oss << "    \"feature_initializer_warp_height_diff_mean\": "
            << r.feature_initializer_warp_height_diff_mean << ",\n";
        oss << "    \"feature_initializer_warp_height_diff_p50\": "
            << r.feature_initializer_warp_height_diff_p50 << ",\n";
        oss << "    \"feature_initializer_warp_height_diff_p75\": "
            << r.feature_initializer_warp_height_diff_p75 << ",\n";
        oss << "    \"feature_initializer_warp_height_diff_p90\": "
            << r.feature_initializer_warp_height_diff_p90 << ",\n";
        oss << "    \"feature_initializer_warp_height_diff_p95\": "
            << r.feature_initializer_warp_height_diff_p95 << ",\n";
        oss << "    \"feature_initializer_warp_height_diff_max\": "
            << r.feature_initializer_warp_height_diff_max << ",\n";
    } else {
        oss << "    \"inlier_ratio\": " << r.inlier_ratio << ",\n";
    }
    oss << "    \"mean_reproj_error\": " << r.mean_reproj_error << ",\n";
    oss << "    \"inlier_spatial_coverage\": " << r.inlier_spatial_coverage << ",\n";
    oss << "    \"warp_overlap_containment\": " << r.warp_overlap_containment << ",\n";
    oss << "    \"warp_source_coverage\": " << r.warp_source_coverage << ",\n";
    oss << "    \"warp_target_coverage\": " << r.warp_target_coverage << ",\n";
    oss << "    \"edge_structure_status\": \""
        << json_output::escapeString(r.edge_structure_status) << "\",\n";
    oss << "    \"edge_structure_message\": \""
        << json_output::escapeString(r.edge_structure_message) << "\",\n";
    oss << "    \"edge_structure_visibility_area_ratio\": "
        << r.edge_structure_visibility_area_ratio << ",\n";
    oss << "    \"edge_structure_visibility_overlap_containment\": "
        << r.edge_structure_visibility_overlap_containment << ",\n";
    oss << "    \"edge_structure_horizontal\": {\n";
    oss << "      \"status\": \""
        << json_output::escapeString(r.edge_structure_horizontal_status) << "\",\n";
    oss << "      \"accepted_matches\": "
        << r.edge_structure_horizontal_accepted_matches << ",\n";
    oss << "      \"Hs\": " << r.edge_structure_source_horizontal_match_ratio << ",\n";
    oss << "      \"Ht\": " << r.edge_structure_target_horizontal_match_ratio << "\n";
    oss << "    },\n";
    oss << "    \"edge_structure_vertical\": {\n";
    oss << "      \"status\": \""
        << json_output::escapeString(r.edge_structure_vertical_status) << "\",\n";
    oss << "      \"accepted_matches\": "
        << r.edge_structure_vertical_accepted_matches << ",\n";
    oss << "      \"Vs\": " << r.edge_structure_source_vertical_match_ratio << ",\n";
    oss << "      \"Vt\": " << r.edge_structure_target_vertical_match_ratio << "\n";
    oss << "    },\n";
    oss << "    \"warp_height_diff_valid_count\": " << r.warp_height_diff_valid_count << ",\n";
    oss << "    \"warp_height_diff_overlap_ratio\": " << r.warp_height_diff_overlap_ratio << ",\n";
    oss << "    \"warp_height_diff_mean\": " << r.warp_height_diff_mean << ",\n";
    oss << "    \"warp_height_diff_p50\": " << r.warp_height_diff_p50 << ",\n";
    oss << "    \"warp_height_diff_p75\": " << r.warp_height_diff_p75 << ",\n";
    oss << "    \"warp_height_diff_p90\": " << r.warp_height_diff_p90 << ",\n";
    oss << "    \"warp_height_diff_p95\": " << r.warp_height_diff_p95 << ",\n";
    oss << "    \"warp_height_diff_max\": " << r.warp_height_diff_max << ",\n";
    oss << "    \"warp_height_diff_compensation_attempted\": "
        << (r.warp_height_diff_compensation_attempted ? "true" : "false") << ",\n";
    oss << "    \"warp_height_diff_global_offset\": "
        << r.warp_height_diff_global_offset << ",\n";
    oss << "    \"warp_height_diff_compensated_mean\": "
        << r.warp_height_diff_compensated_mean << ",\n";
    oss << "    \"warp_height_diff_compensated_p50\": "
        << r.warp_height_diff_compensated_p50 << ",\n";
    oss << "    \"warp_height_diff_compensated_p75\": "
        << r.warp_height_diff_compensated_p75 << ",\n";
    oss << "    \"warp_height_diff_compensated_p90\": "
        << r.warp_height_diff_compensated_p90 << ",\n";
    oss << "    \"warp_height_diff_compensated_p95\": "
        << r.warp_height_diff_compensated_p95 << ",\n";
    oss << "    \"warp_height_diff_compensated_max\": "
        << r.warp_height_diff_compensated_max << ",\n";
    oss << "    \"structure_overlap_iou\": " << r.structure_overlap_iou;
    if (isDirect) {
        oss << ",\n";
        json_output::appendDirectDiagnostics(oss, ctx.direct_data);
    } else {
        oss << "\n";
    }
    oss << "  },\n";
    if (isStructure && ctx.geometry_data.valid &&
        !ctx.geometry_data.A.empty() && ctx.geometry_data.A.rows >= 2 &&
        ctx.geometry_data.A.cols >= 3) {
        oss << "  \"translation\": {\n";
        oss << "    \"dx\": " << ctx.geometry_data.A.at<double>(0, 2) << ",\n";
        oss << "    \"dy\": " << ctx.geometry_data.A.at<double>(1, 2) << "\n";
        oss << "  },\n";
    } else if (isStructure) {
        oss << "  \"translation\": {\n";
        oss << "    \"dx\": " << ctx.structure_match_data.translation.x << ",\n";
        oss << "    \"dy\": " << ctx.structure_match_data.translation.y << "\n";
        oss << "  },\n";
    }
    if (ctx.geometry_data.type == GeometryType::RIGID) {
        const auto& gd = ctx.geometry_data;
        oss << "  \"rigid_candidate_fallback\": {\n";
        oss << "    \"baseline_valid\": "
            << (gd.baseline_valid ? "true" : "false") << ",\n";
        oss << "    \"baseline_num_inliers\": " << gd.baseline_num_inliers << ",\n";
        oss << "    \"baseline_mean_reproj_error\": "
            << gd.baseline_mean_reproj_error << ",\n";
        oss << "    \"candidate_fallback_attempted\": "
            << (gd.candidate_fallback_attempted ? "true" : "false") << ",\n";
        oss << "    \"candidate_trigger_reason\": \""
            << json_output::escapeString(gd.candidate_fallback_trigger_reason) << "\"\n";
        oss << "  },\n";
    }
    oss << "  \"timings_ms\": {\n";
    oss << "    \"load\": " << r.t_load_ms << ",\n";
    oss << "    \"extract\": " << r.t_extract_ms << ",\n";
    oss << "    \"match\": " << r.t_match_ms << ",\n";
    oss << "    \"filter\": " << r.t_filter_ms << ",\n";
    oss << "    \"geometry\": " << r.t_geometry_ms << ",\n";
    oss << "    \"warp\": " << r.t_warp_ms << ",\n";
    oss << "    \"total\": " << r.t_total_ms << "\n";
    oss << "  },\n";
    oss << "  \"metrics\": {\n";
    for (size_t i = 0; i < ctx.evaluation.metrics.size(); ++i) {
        const auto& metric = ctx.evaluation.metrics[i];
        oss << "    \"" << json_output::escapeString(metric.name) << "\": ";
        if (metric.valid) {
            oss << metric.value;
        } else {
            oss << "null";
        }
        if (i + 1 < ctx.evaluation.metrics.size()) {
            oss << ",";
        }
        oss << "\n";
    }
    oss << "  }\n";
    oss << "}\n";
    return oss.str();
}

} // namespace

void loadDatasetNamingOptions(const YAML::Node& ds, DatasetLoader::Options& options) {
    options.pattern_sources =
        readPatternCandidates(ds, "pattern_sources", "pattern_source", {"source", "moving"});
    options.pattern_targets =
        readPatternCandidates(ds, "pattern_targets", "pattern_target", {"target", "reference"});
}

void printSummary(const RegistrationContext& ctx, MethodFamily family) {
    IR_LOG_INFO(buildSummaryText(ctx, family));
}

std::string sampleStemFromPaths(const fs::path& image1, const fs::path& image2) {
    return image1.stem().string() + "__" + image2.stem().string();
}

fs::path normalizeOutputBaseRoot(fs::path root) {
    if (root.empty()) {
        return root;
    }

    const std::string leaf = root.filename().string();
    if (leaf == "single" || leaf == "batch") {
        root = root.parent_path();
    }
    return root;
}

void applyCompareOverrides(PipelineConfig& pipeline_cfg,
                           const YAML::Node& compare_cfg,
                           const YAML::Node& combination,
                           const fs::path& compare_yaml_dir,
                           const fs::path& output_root,
                           const std::string& label) {
    pipeline_cfg.name = yaml_utils::getString(combination, "name", label);

    const std::string output_dir = yaml_utils::getString(combination, "output_dir");
    pipeline_cfg.output_dir =
        output_dir.empty() ? output_root / pipeline_cfg.name
                           : Config::resolvePath(compare_yaml_dir, output_dir);

    // compare YAML 自己决定可视化，不继承组合 pipeline 的 visualization 配置。
    Config::resetVisualization(pipeline_cfg);
    const YAML::Node output = compare_cfg["output"];
    pipeline_cfg.save_visuals = yaml_utils::getBool(output, "save_visuals", true);
    applyVisualizationOverrides(pipeline_cfg, compare_cfg["visualization"]);
    applyVisualizationOverrides(pipeline_cfg, combination["visualization"]);

    // direct 对比可在 compare YAML 顶层统一选择点特征初始器，组合项可按需覆盖。
    const YAML::Node initializer = combination["feature_initializer"]
                                       ? combination["feature_initializer"]
                                       : compare_cfg["feature_initializer"];
    if (pipeline_cfg.methodFamily() == MethodFamily::DIRECT && initializer && initializer.IsMap()) {
        const std::string keypoint = yaml_utils::getString(initializer, "keypoint");
        if (!keypoint.empty()) {
            pipeline_cfg.feature_initializer.candidate.keypoint_path =
                Config::resolvePath(compare_yaml_dir, keypoint);
            pipeline_cfg.feature_initializer.candidate.name =
                yaml_utils::getString(initializer,
                                      "name",
                                      pipeline_cfg.feature_initializer.candidate.name);
        }
    }

    // compare 暂不显示窗口，避免批量组合运行被交互界面阻塞。
    pipeline_cfg.show_source_window = false;
    pipeline_cfg.show_target_window = false;
    pipeline_cfg.show_warped_window = false;
}

void writeRunSummaryFiles(const RegistrationContext& ctx,
                          const PipelineConfig& cfg,
                          const std::string& sample_name) {
    if (ctx.output_dir.empty()) {
        return;
    }

    const auto family = cfg.methodFamily();
    const std::string summary_text = buildSummaryText(ctx, family);
    const std::string summary_json = buildSummaryJson(ctx, cfg, sample_name);

    const fs::path txt_path = ctx.output_dir / "run_summary.txt";
    const fs::path json_path = ctx.output_dir / "run_summary.json";
    if (file_utils::writeWholeFile(txt_path, summary_text)) {
        IR_LOG_INFO("Wrote run summary text: ", txt_path.string());
    }
    if (file_utils::writeWholeFile(json_path, summary_json)) {
        IR_LOG_INFO("Wrote run summary json: ", json_path.string());
    }
}


namespace {

std::string htmlEscape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        switch (character) {
        case '&': escaped += "&amp;"; break;
        case '<': escaped += "&lt;"; break;
        case '>': escaped += "&gt;"; break;
        case '"': escaped += "&quot;"; break;
        case '\'': escaped += "&#39;"; break;
        default: escaped.push_back(character); break;
        }
    }
    return escaped;
}

fs::path findImageWithSuffix(const fs::path& directory,
                             const std::string& suffix,
                             const std::string& preferredToken = {}) {
    if (!fs::is_directory(directory)) {
        return {};
    }
    std::vector<fs::path> images;
    std::error_code ec;
    std::string wantedSuffix = suffix;
    std::transform(wantedSuffix.begin(), wantedSuffix.end(), wantedSuffix.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (const auto& entry : fs::directory_iterator(directory, ec)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string extension = entry.path().extension().string();
        std::string filename = entry.path().filename().string();
        std::string lowerExtension = extension;
        std::transform(lowerExtension.begin(), lowerExtension.end(), lowerExtension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        std::transform(filename.begin(), filename.end(), filename.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if ((lowerExtension == ".png" || lowerExtension == ".jpg" || lowerExtension == ".jpeg") &&
            filename.size() >= wantedSuffix.size() &&
            filename.compare(filename.size() - wantedSuffix.size(), wantedSuffix.size(), wantedSuffix) == 0) {
            images.push_back(entry.path());
        }
    }
    if (images.empty()) {
        return {};
    }
    if (!preferredToken.empty()) {
        std::string token = preferredToken;
        std::transform(token.begin(), token.end(), token.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const auto preferred = std::find_if(images.begin(), images.end(), [&](const fs::path& path) {
            std::string filename = path.filename().string();
            std::transform(filename.begin(), filename.end(), filename.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return filename.find(token) != std::string::npos;
        });
        if (preferred != images.end()) {
            return *preferred;
        }
    }
    std::sort(images.begin(), images.end());
    return images.front();
}

} // namespace

void writeBatchHtmlReport(const std::filesystem::path& pipeline_root,
                          const std::vector<std::string>& sample_names,
                          const std::vector<RegistrationResult>& results,
                          const std::filesystem::path& assets_root) {
    if (pipeline_root.empty() || sample_names.empty()) {
        return;
    }

    const int total = static_cast<int>(sample_names.size());
    const double totalTimeMs = std::accumulate(
        results.begin(), results.end(), 0.0,
        [](double sum, const RegistrationResult& result) { return sum + result.t_total_ms; });
    const int fastCount = static_cast<int>(std::count_if(
        results.begin(), results.end(), [](const RegistrationResult& result) {
            return result.registration_strategy == "FAST";
        }));
    const int multilayerCount = static_cast<int>(std::count_if(
        results.begin(), results.end(), [](const RegistrationResult& result) {
            return result.registration_strategy == "MULTILAYER";
        }));

    std::ostringstream html;
    html << "<!doctype html><html><head><meta charset=\"utf-8\"><title>"
         << htmlEscape(pipeline_root.filename().string())
         << " false color report</title><style>"
         << "*{box-sizing:border-box}body{font-family:Arial,sans-serif;color:#14213d;background:#f7f9fc;"
            "margin:0;padding:18px}h1{font-size:22px;margin:0 0 16px}.summary{display:grid;"
             "grid-template-columns:repeat(4,minmax(140px,1fr));gap:12px;margin-bottom:22px}.card{"
            "background:#fff;border:1px solid #d5ddea;border-radius:8px;padding:16px}.value{font-size:24px;"
            "font-weight:700}.label{color:#52627a;font-size:12px;margin-top:8px}.case{background:#fff;"
            "border:1px solid #d5ddea;border-radius:8px;margin:0 0 12px;overflow:hidden}.case-head{"
             "display:flex;align-items:center;padding:14px 18px;"
             "border-bottom:1px solid #d5ddea}.case h2{font-size:17px;margin:0}.case-body{padding:12px 18px}.meta{color:#52627a;"
             "font-size:13px;margin-bottom:12px}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.image-cell{min-width:0}.image-label{font-size:12px;font-weight:700;color:#52627a;margin-bottom:6px}.case img{display:block;width:100%;height:260px;object-fit:contain;background:#000}.notice{color:#b3261e;"
             "padding:12px 0;font-size:13px}@media(max-width:800px){.summary{grid-template-columns:repeat(2,1fr)}.images{grid-template-columns:1fr}}"
             "</style></head><body>";
    html << "<h1>False-color registration report</h1><div class=\"summary\">";
    html << "<div class=\"card\"><div class=\"value\">" << total
         << "</div><div class=\"label\">total</div></div>";
    html << "<div class=\"card\"><div class=\"value\">" << std::fixed << std::setprecision(1)
         << totalTimeMs << " ms</div><div class=\"label\">total time</div></div>";
    html << "<div class=\"card\"><div class=\"value\">" << fastCount
         << "</div><div class=\"label\">FAST</div></div>";
    html << "<div class=\"card\"><div class=\"value\">" << multilayerCount
         << "</div><div class=\"label\">MULTILAYER</div></div></div>";

    for (size_t index = 0; index < sample_names.size(); ++index) {
        const std::string name = htmlEscape(sample_names[index]);
        const RegistrationResult* result = index < results.size() ? &results[index] : nullptr;
        const std::string strategy = result && !result->registration_strategy.empty()
            ? result->registration_strategy : "UNKNOWN";
        const double elapsedMs = result ? result->t_total_ms : 0.0;
        html << "<section class=\"case\"><div class=\"case-head\"><h2>" << name
             << "</h2></div><div class=\"case-body\"><div class=\"meta\">方案："
             << htmlEscape(strategy) << "　耗时：" << std::fixed << std::setprecision(2)
             << elapsedMs << " ms</div><div class=\"images\">";
        const fs::path imageRoot = assets_root.empty() ? pipeline_root : assets_root;
        const fs::path caseRoot = imageRoot / sample_names[index];
        const fs::path source = findImageWithSuffix(caseRoot / "originals", "_source_original.png");
        const fs::path target = findImageWithSuffix(caseRoot / "originals", "_target_original.png");
        const fs::path falseColor = findImageWithSuffix(
            caseRoot / "overlay", "_false_color_overlay.png",
            strategy == "MULTILAYER" ? "DARK_MULTILAYER" : "");
        const auto writeImage = [&](const char* label, const fs::path& image) {
            html << "<div class=\"image-cell\"><div class=\"image-label\">" << label << "</div>";
            if (!image.empty()) {
                const fs::path relative = fs::relative(image, pipeline_root);
                html << "<img src=\"" << htmlEscape(relative.generic_string()) << "\" alt=\""
                     << name << " " << label << "\">";
            } else {
                html << "<div class=\"notice\">未生成</div>";
            }
            html << "</div>";
        };
        writeImage("source", source);
        writeImage("target", target);
        writeImage("false-color", falseColor);
        html << "</div>";
        html << "</div></section>";
    }
    html << "</body></html>";

    const fs::path reportPath = pipeline_root / "false_color_report.html";
    // 让报告源文件也保持可读，便于直接检查生成的统计和用例条目。
    const std::string compactHtml = html.str();
    std::string formattedHtml;
    formattedHtml.reserve(compactHtml.size() + compactHtml.size() / 20);
    for (const char character : compactHtml) {
        formattedHtml.push_back(character);
        if (character == '>') {
            formattedHtml.push_back('\n');
        }
    }
    if (file_utils::writeWholeFile(reportPath, formattedHtml)) {
        IR_LOG_INFO("Wrote batch HTML report: ", reportPath.string());
    }
}

} // namespace ir::registration_app_helpers
