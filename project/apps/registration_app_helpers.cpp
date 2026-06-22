#include "registration_app_helpers.h"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
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
    if (r.warp_edge_alignment_iou >= 0.0) {
        oss << "  edge align    : " << std::fixed << std::setprecision(3)
            << r.warp_edge_alignment_iou << "\n";
    }
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
    if (!node || !node.IsMap()) {
        return;
    }
    if (node["draw_keypoints"])
        pipelineCfg.draw_keypoints =
            yaml_utils::getBool(node, "draw_keypoints", pipelineCfg.draw_keypoints);
    if (node["draw_matches"])
        pipelineCfg.draw_matches =
            yaml_utils::getBool(node, "draw_matches", pipelineCfg.draw_matches);
    if (node["draw_inliers_only"])
        pipelineCfg.draw_inliers_only =
            yaml_utils::getBool(node, "draw_inliers_only", pipelineCfg.draw_inliers_only);
    if (node["max_matches_drawn"])
        pipelineCfg.max_matches_drawn =
            yaml_utils::getInt(node, "max_matches_drawn", pipelineCfg.max_matches_drawn);
    if (node["warp"])
        pipelineCfg.warp = yaml_utils::getBool(node, "warp", pipelineCfg.warp);
    if (node["show_source_window"])
        pipelineCfg.show_source_window =
            yaml_utils::getBool(node, "show_source_window", pipelineCfg.show_source_window);
    if (node["show_target_window"])
        pipelineCfg.show_target_window =
            yaml_utils::getBool(node, "show_target_window", pipelineCfg.show_target_window);
    if (node["show_warped_window"])
        pipelineCfg.show_warped_window =
            yaml_utils::getBool(node, "show_warped_window", pipelineCfg.show_warped_window);
    if (node["wait_key"])
        pipelineCfg.wait_key = yaml_utils::getInt(node, "wait_key", pipelineCfg.wait_key);
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
    if (r.warp_bidirectional_coverage >= 0.0) {
        oss << "  warp bi-cov   : " << std::fixed << std::setprecision(3)
            << r.warp_bidirectional_coverage << "\n";
    }
    if (r.warp_photometric_error >= 0.0) {
        oss << "  warp NMAD      : " << std::fixed << std::setprecision(4)
            << r.warp_photometric_error << "\n";
    }
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
    if (r.warp_bidirectional_coverage >= 0.0) {
        oss << "  warp bi-cov   : " << std::fixed << std::setprecision(3)
            << r.warp_bidirectional_coverage << "\n";
    }
    if (r.warp_photometric_error >= 0.0) {
        oss << "  warp NMAD      : " << std::fixed << std::setprecision(4)
            << r.warp_photometric_error << "\n";
    }
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
    if (r.warp_bidirectional_coverage >= 0.0) {
        oss << "  warp bi-cov   : " << std::fixed << std::setprecision(3)
            << r.warp_bidirectional_coverage << "\n";
    }
    if (r.warp_photometric_error >= 0.0) {
        oss << "  warp NMAD      : " << std::fixed << std::setprecision(4)
            << r.warp_photometric_error << "\n";
    }
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
            if (r.feature_initializer_warp_photometric_error >= 0.0) {
                oss << "  init NMAD     : " << std::fixed << std::setprecision(4)
                    << r.feature_initializer_warp_photometric_error << "\n";
            }
            if (r.feature_initializer_warp_edge_alignment_iou >= 0.0) {
                oss << "  init edge     : " << std::fixed << std::setprecision(3)
                    << r.feature_initializer_warp_edge_alignment_iou << "\n";
            }
        }
    }
    summary_text::appendDirectDiagnostics(oss, dd);
    if (dd.photometric_error >= 0.0) {
        oss << "  photometric MSE: " << std::fixed << std::setprecision(6)
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
    if (r.warp_bidirectional_coverage >= 0.0) {
        oss << "  warp bi-cov   : " << std::fixed << std::setprecision(3)
            << r.warp_bidirectional_coverage << "\n";
    }
    if (r.warp_photometric_error >= 0.0) {
        oss << "  warp NMAD      : " << std::fixed << std::setprecision(4)
            << r.warp_photometric_error << "\n";
    }
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
        oss << "    \"feature_initializer_warp_photometric_error\": "
            << r.feature_initializer_warp_photometric_error << ",\n";
        oss << "    \"feature_initializer_warp_edge_alignment_iou\": "
            << r.feature_initializer_warp_edge_alignment_iou << ",\n";
    } else {
        oss << "    \"inlier_ratio\": " << r.inlier_ratio << ",\n";
    }
    oss << "    \"mean_reproj_error\": " << r.mean_reproj_error << ",\n";
    oss << "    \"inlier_spatial_coverage\": " << r.inlier_spatial_coverage << ",\n";
    oss << "    \"warp_overlap_containment\": " << r.warp_overlap_containment << ",\n";
    oss << "    \"warp_source_coverage\": " << r.warp_source_coverage << ",\n";
    oss << "    \"warp_target_coverage\": " << r.warp_target_coverage << ",\n";
    oss << "    \"warp_bidirectional_coverage\": " << r.warp_bidirectional_coverage << ",\n";
    oss << "    \"warp_edge_alignment_iou\": " << r.warp_edge_alignment_iou << ",\n";
    oss << "    \"warp_photometric_error\": " << r.warp_photometric_error << ",\n";
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
    std::cout << buildSummaryText(ctx, family);
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

    const YAML::Node output = compare_cfg["output"];
    if (output && output.IsMap() && output["save_visuals"]) {
        pipeline_cfg.draw_matches =
            yaml_utils::getBool(output, "save_visuals", pipeline_cfg.draw_matches);
    }
    applyVisualizationOverrides(pipeline_cfg, compare_cfg["visualization"]);
    applyVisualizationOverrides(pipeline_cfg, combination["visualization"]);
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

} // namespace ir::registration_app_helpers

