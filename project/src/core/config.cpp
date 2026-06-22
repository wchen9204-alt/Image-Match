#include "core/config.h"

#include <fstream>
#include <stdexcept>
#include <vector>

#include "utils/logger.h"
#include "utils/string_utils.h"
#include "utils/yaml_utils.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

std::vector<MatchView> parseMatchViews(const YAML::Node& vis,
                                       const std::vector<MatchView>& fallback) {
    if (!vis || !vis.IsMap() || !vis["match_views"] || !vis["match_views"].IsSequence()) {
        return fallback;
    }

    std::vector<MatchView> out;
    for (const auto& item : vis["match_views"]) {
        const std::string key = string_utils::toUpperAscii(item.as<std::string>());
        if (key == "RAW" || key == "ALL") {
            out.push_back(MatchView::RAW);
        } else if (key == "FILTERED" || key == "FILTER" || key == "FILTER_MATCH") {
            out.push_back(MatchView::FILTERED);
        } else if (key == "INLIERS" || key == "INLIER" || key == "INLIER_MATCH") {
            out.push_back(MatchView::INLIERS);
        } else {
            IR_LOG_WARN("Unknown visualization.match_views entry ignored: ", key);
        }
    }
    return out.empty() ? fallback : out;
}

std::string pathStemUpper(const fs::path& path) {
    return string_utils::toUpperAscii(path.stem().string());
}

DirectValidationReferenceMode parseDirectValidationReferenceMode(const std::string& raw) {
    const std::string key = string_utils::toUpperAscii(raw);
    if (key == "BEST_OF_DIRECT_AND_INITIALIZER" || key == "BEST_OF_BOTH" ||
        key == "BEST_OF_DIRECT_AND_INIT") {
        return DirectValidationReferenceMode::BEST_OF_DIRECT_AND_INITIALIZER;
    }
    return DirectValidationReferenceMode::DIRECT_ONLY;
}

PipelineConfig::FeatureInitializerCandidateConfig
parseFeatureInitializerCandidate(const YAML::Node& item, const fs::path& base) {
    PipelineConfig::FeatureInitializerCandidateConfig candidate;

    // 兼容两种写法：字符串表示 keypoint 配置路径；map 可额外指定显示名。
    if (item.IsScalar()) {
        candidate.keypoint_path = Config::resolvePath(base, item.as<std::string>());
        candidate.name = pathStemUpper(candidate.keypoint_path);
    } else if (item.IsMap()) {
        const std::string keypointEntry =
            yaml_utils::getString(item, "keypoint", yaml_utils::getString(item, "path"));
        candidate.keypoint_path = Config::resolvePath(base, keypointEntry);
        candidate.name = yaml_utils::getString(item, "name", pathStemUpper(candidate.keypoint_path));
    }

    candidate.name = string_utils::toUpperAscii(candidate.name);
    return candidate;
}

void parseFeatureInitializer(const YAML::Node& node, const fs::path& base, PipelineConfig& cfg) {
    if (!node["feature_initializer"] || !node["feature_initializer"].IsMap()) {
        return;
    }

    const auto& initializer = node["feature_initializer"];
    cfg.feature_initializer.enabled =
        yaml_utils::getBool(initializer, "enabled", cfg.feature_initializer.enabled);
    cfg.feature_initializer.final_validation_reference =
        parseDirectValidationReferenceMode(yaml_utils::getString(
            initializer,
            "final_validation_reference",
            "DIRECT_ONLY"));

    cfg.feature_initializer.candidate = {};
    if (initializer["candidate"]) {
        cfg.feature_initializer.candidate =
            parseFeatureInitializerCandidate(initializer["candidate"], base);
    } else if (initializer["candidates"] && initializer["candidates"].IsSequence()) {
        // 兼容旧写法；只读取第一项有效候选。
        for (const auto& item : initializer["candidates"]) {
            auto candidate = parseFeatureInitializerCandidate(item, base);
            if (!candidate.keypoint_path.empty()) {
                cfg.feature_initializer.candidate = candidate;
                break;
            }
        }
    } else {
        const std::string keypointEntry = yaml_utils::getString(initializer, "keypoint");
        if (!keypointEntry.empty()) {
            PipelineConfig::FeatureInitializerCandidateConfig candidate;
            candidate.keypoint_path = Config::resolvePath(base, keypointEntry);
            candidate.name =
                string_utils::toUpperAscii(yaml_utils::getString(initializer,
                                                                 "name",
                                                                 pathStemUpper(candidate.keypoint_path)));
            cfg.feature_initializer.candidate = candidate;
        }
    }

    cfg.feature_initializer.matcher_path =
        Config::resolvePath(base, yaml_utils::getString(initializer, "matcher"));

    cfg.feature_initializer.filter_paths.clear();
    if (initializer["filters"] && initializer["filters"].IsSequence()) {
        for (const auto& f : initializer["filters"]) {
            cfg.feature_initializer.filter_paths.push_back(
                Config::resolvePath(base, f.as<std::string>()));
        }
    }

    cfg.feature_initializer.geometry_path =
        Config::resolvePath(base, yaml_utils::getString(initializer, "geometry"));

    if (initializer["acceptance"] && initializer["acceptance"].IsMap()) {
        const auto& acceptance = initializer["acceptance"];
        auto& accept = cfg.feature_initializer.acceptance;
        accept.min_inliers =
            yaml_utils::getInt(acceptance, "min_inliers", accept.min_inliers);
        accept.min_inlier_ratio =
            yaml_utils::getDouble(acceptance,
                                  "min_inlier_ratio",
                                  accept.min_inlier_ratio);
        accept.min_inlier_spatial_coverage =
            yaml_utils::getDouble(acceptance,
                                  "min_inlier_spatial_coverage",
                                  accept.min_inlier_spatial_coverage);
    }

    if (initializer["validation"] && initializer["validation"].IsMap()) {
        const auto& validation = initializer["validation"];
        if (validation["warp_overlap"] && validation["warp_overlap"].IsMap()) {
            const auto& overlap = validation["warp_overlap"];
            auto& overlapCfg = cfg.feature_initializer.validation.overlap;
            const bool enabled =
                yaml_utils::getBool(overlap,
                                    "enabled",
                                    overlapCfg.containment_enabled ||
                                        overlapCfg.bidirectional_coverage_enabled);
            overlapCfg.containment_enabled = enabled;
            overlapCfg.bidirectional_coverage_enabled = enabled;
            overlapCfg.min_containment =
                yaml_utils::getDouble(overlap,
                                      "min_containment",
                                      overlapCfg.min_containment);
            overlapCfg.min_bidirectional_coverage =
                yaml_utils::getDouble(overlap,
                                      "min_bidirectional_coverage",
                                      overlapCfg.min_bidirectional_coverage);
            overlapCfg.accept_if_either_passes =
                yaml_utils::getBool(overlap,
                                    "accept_if_either_passes",
                                    overlapCfg.accept_if_either_passes);
            overlapCfg.foreground_threshold =
                yaml_utils::getInt(overlap,
                                   "foreground_threshold",
                                   overlapCfg.foreground_threshold);
        }
        if (validation["photometric"] && validation["photometric"].IsMap()) {
            const auto& photometric = validation["photometric"];
            auto& photoCfg = cfg.feature_initializer.validation.photometric;
            photoCfg.enabled =
                yaml_utils::getBool(photometric,
                                    "enabled",
                                    photoCfg.enabled);
            photoCfg.max_nmad =
                yaml_utils::getDouble(photometric,
                                      "max_nmad",
                                      photoCfg.max_nmad);
            photoCfg.max_nmad_for_coverage_only =
                yaml_utils::getDouble(photometric,
                                      "max_nmad_for_coverage_only",
                                      photoCfg.max_nmad_for_coverage_only);
        }
        if (validation["edge_alignment"] && validation["edge_alignment"].IsMap()) {
            const auto& edgeAlignment = validation["edge_alignment"];
            auto& edgeCfg = cfg.feature_initializer.validation.edge_alignment;
            edgeCfg.enabled =
                yaml_utils::getBool(edgeAlignment,
                                    "enabled",
                                    edgeCfg.enabled);
            edgeCfg.min_iou =
                yaml_utils::getDouble(edgeAlignment,
                                      "min_iou",
                                      edgeCfg.min_iou);
            edgeCfg.canny_low_threshold =
                yaml_utils::getInt(edgeAlignment,
                                   "canny_low_threshold",
                                   edgeCfg.canny_low_threshold);
            edgeCfg.canny_high_threshold =
                yaml_utils::getInt(edgeAlignment,
                                   "canny_high_threshold",
                                   edgeCfg.canny_high_threshold);
            edgeCfg.dilate_size =
                yaml_utils::getInt(edgeAlignment,
                                   "dilate_size",
                                   edgeCfg.dilate_size);
            edgeCfg.min_edge_pixels =
                yaml_utils::getInt(edgeAlignment,
                                   "min_edge_pixels",
                                   edgeCfg.min_edge_pixels);
        }
    }
}

} // namespace

YAML::Node Config::load(const fs::path& path) {
    if (!fs::exists(path)) {
        throw std::runtime_error("Config::load - file not found: " + path.string());
    }
    try {
        return YAML::LoadFile(path.string());
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Config::load - YAML parse error in " + path.string() + ": " +
                                 e.what());
    }
}

fs::path Config::resolvePath(const fs::path& base_dir, const std::string& relative_or_absolute) {
    if (relative_or_absolute.empty())
        return {};

    fs::path p(relative_or_absolute);
    if (p.is_absolute() && fs::exists(p))
        return p;

    if (!base_dir.empty()) {
        fs::path c1 = base_dir / p;
        if (fs::exists(c1))
            return fs::weakly_canonical(c1);
    }

    fs::path c2 = fs::current_path() / p;
    if (fs::exists(c2))
        return fs::weakly_canonical(c2);

    fs::path walk = base_dir;
    for (int i = 0; i < 4 && !walk.empty(); ++i) {
        fs::path c3 = walk / p;
        if (fs::exists(c3))
            return fs::weakly_canonical(c3);
        if (walk.has_parent_path())
            walk = walk.parent_path();
        else
            break;
    }

    if (!base_dir.empty())
        return fs::weakly_canonical(base_dir / p);
    return fs::weakly_canonical(c2);
}

PipelineConfig Config::loadPipeline(const fs::path& path) {
    YAML::Node node = load(path);
    const fs::path base = path.parent_path();

    PipelineConfig cfg;
    cfg.name = yaml_utils::getString(node, "name", path.stem().string());

    // 方法族：显式声明优先，否则从配置路径推断（兼容旧 YAML）
    const std::string familyStr = yaml_utils::getString(node, "method_family");
    if (!familyStr.empty()) {
        const std::string key = familyStr;
        if (key == "keypoint" || key == "KEYPOINT")       cfg.method_family = MethodFamily::KEYPOINT;
        else if (key == "structure" || key == "STRUCTURE") cfg.method_family = MethodFamily::STRUCTURE;
        else if (key == "direct" || key == "DIRECT")       cfg.method_family = MethodFamily::DIRECT;
        else if (key == "learning" || key == "LEARNING" ||
                 key == "deep" || key == "DEEP")           cfg.method_family = MethodFamily::LEARNING;
    } else {
        // 向后兼容：无 method_family 时从路径推断
        if (!yaml_utils::getString(node, "structure").empty())
            cfg.method_family = MethodFamily::STRUCTURE;
    }

    // 各方法族配置文件路径
    const std::string keypoint_entry = yaml_utils::getString(
        node, "keypoint", yaml_utils::getString(node, "feature"));
    cfg.keypoint_path = resolvePath(base, keypoint_entry);
    cfg.structure_path = resolvePath(base, yaml_utils::getString(node, "structure"));
    cfg.direct_path   = resolvePath(base, yaml_utils::getString(node, "direct"));
    cfg.learning_path = resolvePath(base, yaml_utils::getString(node, "learning"));
    cfg.matcher_path  = resolvePath(base, yaml_utils::getString(node, "matcher"));
    cfg.geometry_path = resolvePath(base, yaml_utils::getString(node, "geometry"));

    cfg.filter_paths.clear();
    if (node["filters"] && node["filters"].IsSequence()) {
        for (const auto& f : node["filters"]) {
            cfg.filter_paths.push_back(resolvePath(base, f.as<std::string>()));
        }
    }

    cfg.evaluator_path = resolvePath(base, yaml_utils::getString(node, "evaluator"));

    if (node["io"] && node["io"].IsMap()) {
        const auto& io = node["io"];
        cfg.image1_path = resolvePath(base, yaml_utils::getString(io, "image1"));
        cfg.image2_path = resolvePath(base, yaml_utils::getString(io, "image2"));
        cfg.output_dir = resolvePath(base, yaml_utils::getString(io, "output_dir", "outputs"));
    }

    if (node["visualization"] && node["visualization"].IsMap()) {
        const auto& vis = node["visualization"];
        cfg.draw_keypoints = yaml_utils::getBool(vis, "draw_keypoints", false);
        cfg.draw_matches = yaml_utils::getBool(vis, "draw_matches", true);
        cfg.draw_inliers_only = yaml_utils::getBool(vis, "draw_inliers_only", false);
        cfg.match_views = parseMatchViews(vis, cfg.match_views);
        cfg.max_matches_drawn = yaml_utils::getInt(vis, "max_matches_drawn", 100);
        cfg.warp = yaml_utils::getBool(vis, "warp", true);
        cfg.show_source_window = yaml_utils::getBool(vis, "show_source_window", false);
        cfg.show_target_window = yaml_utils::getBool(vis, "show_target_window", false);
        cfg.show_warped_window = yaml_utils::getBool(vis, "show_warped_window", false);
        cfg.wait_key = yaml_utils::getInt(vis, "wait_key", 0);
    }

    if (node["validation"] && node["validation"].IsMap()) {
        const auto& validation = node["validation"];
        if (validation["warp_overlap"] && validation["warp_overlap"].IsMap()) {
            const auto& overlap = validation["warp_overlap"];
            auto& overlapCfg = cfg.warp_quality.overlap;
            const bool warpOverlapEnabled = yaml_utils::getBool(overlap, "enabled", false);
            const bool hasContainmentThreshold = static_cast<bool>(overlap["min_containment"]);
            overlapCfg.min_containment =
                yaml_utils::getDouble(overlap, "min_containment", overlapCfg.min_containment);
            overlapCfg.containment_enabled =
                warpOverlapEnabled &&
                yaml_utils::getBool(overlap,
                                    "containment_enabled",
                                    hasContainmentThreshold);
            const bool hasBidirectionalCoverageThreshold =
                static_cast<bool>(overlap["min_bidirectional_coverage"]);
            const bool hasLegacySourceCoverageThreshold =
                static_cast<bool>(overlap["min_source_coverage"]);
            overlapCfg.min_bidirectional_coverage =
                hasBidirectionalCoverageThreshold
                    ? yaml_utils::getDouble(overlap,
                                            "min_bidirectional_coverage",
                                            overlapCfg.min_bidirectional_coverage)
                    : yaml_utils::getDouble(overlap,
                                            "min_source_coverage",
                                            overlapCfg.min_bidirectional_coverage);
            overlapCfg.bidirectional_coverage_enabled =
                yaml_utils::getBool(overlap,
                                    "bidirectional_coverage_enabled",
                                    hasBidirectionalCoverageThreshold ||
                                        hasLegacySourceCoverageThreshold);
            overlapCfg.accept_if_either_passes =
                yaml_utils::getBool(overlap,
                                    "accept_if_either_passes",
                                    overlapCfg.accept_if_either_passes);
            overlapCfg.foreground_threshold =
                yaml_utils::getInt(overlap, "foreground_threshold", overlapCfg.foreground_threshold);
        }
        if (validation["photometric"] && validation["photometric"].IsMap()) {
            const auto& photometric = validation["photometric"];
            auto& photoCfg = cfg.warp_quality.photometric;
            photoCfg.enabled =
                yaml_utils::getBool(photometric, "enabled", photoCfg.enabled);
            photoCfg.max_nmad =
                yaml_utils::getDouble(photometric, "max_nmad", photoCfg.max_nmad);
            photoCfg.max_nmad_for_coverage_only =
                yaml_utils::getDouble(photometric,
                                      "max_nmad_for_coverage_only",
                                      photoCfg.max_nmad_for_coverage_only);
        }
        if (validation["edge_alignment"] && validation["edge_alignment"].IsMap()) {
            const auto& edgeAlignment = validation["edge_alignment"];
            auto& edgeCfg = cfg.warp_quality.edge_alignment;
            edgeCfg.enabled =
                yaml_utils::getBool(edgeAlignment, "enabled", edgeCfg.enabled);
            edgeCfg.min_iou =
                yaml_utils::getDouble(edgeAlignment, "min_iou", edgeCfg.min_iou);
            edgeCfg.canny_low_threshold =
                yaml_utils::getInt(edgeAlignment,
                                   "canny_low_threshold",
                                   edgeCfg.canny_low_threshold);
            edgeCfg.canny_high_threshold =
                yaml_utils::getInt(edgeAlignment,
                                   "canny_high_threshold",
                                   edgeCfg.canny_high_threshold);
            edgeCfg.dilate_size =
                yaml_utils::getInt(edgeAlignment, "dilate_size", edgeCfg.dilate_size);
            edgeCfg.min_edge_pixels =
                yaml_utils::getInt(edgeAlignment, "min_edge_pixels", edgeCfg.min_edge_pixels);
        }
        if (validation["match_quality"] && validation["match_quality"].IsMap()) {
            const auto& matchQuality = validation["match_quality"];
            cfg.validate_match_quality = yaml_utils::getBool(matchQuality, "enabled", false);
            cfg.fail_on_match_quality =
                yaml_utils::getBool(matchQuality, "fail_on_violation", false);
            cfg.min_match_inliers = yaml_utils::getInt(matchQuality, "min_inliers", 0);
            cfg.min_match_inlier_ratio =
                yaml_utils::getDouble(matchQuality, "min_inlier_ratio", -1.0);
            cfg.max_match_reproj_error =
                yaml_utils::getDouble(matchQuality, "max_reproj_error", -1.0);
            cfg.min_inlier_spatial_coverage =
                yaml_utils::getDouble(matchQuality, "min_inlier_spatial_coverage", -1.0);
        }
        if (validation["direct_quality"] && validation["direct_quality"].IsMap()) {
            const auto& directQuality = validation["direct_quality"];
            cfg.validate_direct_quality = yaml_utils::getBool(directQuality, "enabled", false);
            cfg.fail_on_direct_quality =
                yaml_utils::getBool(directQuality, "fail_on_violation", true);
            cfg.min_direct_confidence =
                yaml_utils::getDouble(directQuality, "min_confidence", -1.0);
        }
        if (validation["structure_overlap"] && validation["structure_overlap"].IsMap()) {
            const auto& overlap = validation["structure_overlap"];
            cfg.validate_structure_overlap = yaml_utils::getBool(overlap, "enabled", false);
            cfg.min_structure_overlap_iou = yaml_utils::getDouble(overlap, "min_iou", 0.20);
            cfg.structure_overlap_foreground_threshold =
                yaml_utils::getInt(overlap, "foreground_threshold", 0);
            cfg.structure_overlap_dilate_size =
                yaml_utils::getInt(overlap, "dilate_size", 3);
        }
    }

    parseFeatureInitializer(node, base, cfg);

    IR_LOG_INFO("Pipeline '", cfg.name, "' loaded from ", path.string());
    IR_LOG_INFO("  keypoint : ", cfg.keypoint_path.string());
    IR_LOG_INFO("  structure: ", cfg.structure_path.string());
    IR_LOG_INFO("  learning : ", cfg.learning_path.string());
    IR_LOG_INFO("  matcher  : ", cfg.matcher_path.string());
    for (const auto& f : cfg.filter_paths) {
        IR_LOG_INFO("  filter   : ", f.string());
    }
    IR_LOG_INFO("  geometry : ", cfg.geometry_path.string());
    if (cfg.feature_initializer.enabled) {
        IR_LOG_INFO("  direct initializer candidate configured: ",
                    cfg.feature_initializer.candidate.keypoint_path.empty() ? 0 : 1);
    }
    IR_LOG_INFO("  image1   : ", cfg.image1_path.string());
    IR_LOG_INFO("  image2   : ", cfg.image2_path.string());
    IR_LOG_INFO("  output   : ", cfg.output_dir.string());

    return cfg;
}

} // namespace ir

