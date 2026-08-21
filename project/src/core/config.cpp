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
    return out;
}

StructureMatchSource parseStructureMatchSource(const YAML::Node& vis,
                                                StructureMatchSource fallback) {
    if (!vis || !vis.IsMap() || !vis["structure_match_source"]) {
        return fallback;
    }
    const std::string key =
        string_utils::toUpperAscii(vis["structure_match_source"].as<std::string>());
    if (key == "RAW") {
        return StructureMatchSource::RAW;
    }
    if (key == "INLIERS" || key == "INLIER") {
        return StructureMatchSource::INLIERS;
    }
    IR_LOG_WARN("Unknown visualization.structure_match_source ignored: ", key);
    return fallback;
}

std::string pathStemUpper(const fs::path& path) {
    return string_utils::toUpperAscii(path.stem().string());
}

YAML::Node mergeYamlMaps(const YAML::Node& base, const YAML::Node& overrides) {
    YAML::Node merged(YAML::NodeType::Map);
    if (base && base.IsMap()) {
        merged = YAML::Clone(base);
    }
    if (!overrides || !overrides.IsMap()) {
        return merged;
    }

    for (const auto& item : overrides) {
        const std::string key = item.first.as<std::string>();
        if (key == "profile" || key == "preset") {
            continue;
        }
        if (merged[key] && merged[key].IsMap() && item.second.IsMap()) {
            merged[key] = mergeYamlMaps(merged[key], item.second);
        } else {
            merged[key] = YAML::Clone(item.second);
        }
    }
    return merged;
}

YAML::Node resolveValidationProfile(const YAML::Node& validation, const fs::path& base) {
    if (!validation || !validation.IsMap() || !validation["profile"]) {
        return validation;
    }

    const std::string profileEntry = yaml_utils::getString(validation, "profile");
    const std::string preset = yaml_utils::getString(validation, "preset");
    if (profileEntry.empty() || preset.empty()) {
        throw std::runtime_error(
            "validation.profile requires both a profile path and a preset name");
    }

    const fs::path profilePath = Config::resolvePath(base, profileEntry);
    const YAML::Node profileRoot = Config::load(profilePath);
    const YAML::Node profiles = profileRoot["profiles"];
    const YAML::Node selected = profiles ? profiles[preset] : YAML::Node();
    if (!selected || !selected.IsMap()) {
        throw std::runtime_error("Validation preset '" + preset + "' not found in " +
                                 profilePath.string());
    }
    const YAML::Node shared = profileRoot["shared"];
    return mergeYamlMaps(mergeYamlMaps(shared, selected), validation);
}

DirectValidationReferenceMode parseDirectValidationReferenceMode(const std::string& raw) {
    const std::string key = string_utils::toUpperAscii(raw);
    if (key == "BEST_OF_DIRECT_AND_INITIALIZER" || key == "BEST_OF_BOTH" ||
        key == "BEST_OF_DIRECT_AND_INIT") {
        return DirectValidationReferenceMode::BEST_OF_DIRECT_AND_INITIALIZER;
    }
    return DirectValidationReferenceMode::DIRECT_ONLY;
}

FeatureInitializerSeedMode parseFeatureInitializerSeedMode(const std::string& raw) {
    const std::string key = string_utils::toUpperAscii(raw);
    if (key == "ESTIMATED_WHEN_AVAILABLE" || key == "ALWAYS_IF_ESTIMATED") {
        return FeatureInitializerSeedMode::ESTIMATED_WHEN_AVAILABLE;
    }
    return FeatureInitializerSeedMode::ACCEPTED_ONLY;
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
    cfg.feature_initializer.seed_mode = parseFeatureInitializerSeedMode(
        yaml_utils::getString(initializer, "seed_mode", "ACCEPTED_ONLY"));
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
            overlapCfg.containment_enabled =
                yaml_utils::getBool(overlap, "enabled", overlapCfg.containment_enabled);
            overlapCfg.min_containment =
                yaml_utils::getDouble(overlap,
                                      "min_containment",
                                      overlapCfg.min_containment);
            overlapCfg.foreground_threshold =
                yaml_utils::getInt(overlap,
                                   "foreground_threshold",
                                   overlapCfg.foreground_threshold);
            overlapCfg.containment_tolerance_pixels =
                yaml_utils::getInt(overlap,
                                   "containment_tolerance_pixels",
                                   overlapCfg.containment_tolerance_pixels);
        }
        if (validation["height_difference"] && validation["height_difference"].IsMap()) {
            const auto& heightDifference = validation["height_difference"];
            auto& heightCfg = cfg.feature_initializer.validation.height_difference;
            heightCfg.enabled = yaml_utils::getBool(heightDifference, "enabled", heightCfg.enabled);
            heightCfg.compensate_global_height_offset = yaml_utils::getBool(
                heightDifference,
                "compensate_global_height_offset",
                heightCfg.compensate_global_height_offset);
            heightCfg.percentile = yaml_utils::getInt(heightDifference, "percentile", heightCfg.percentile);
            heightCfg.max_abs_error = yaml_utils::getDouble(
                heightDifference, "max_abs_error", heightCfg.max_abs_error);
            if (heightDifference["local_noise_fallback"] &&
                heightDifference["local_noise_fallback"].IsMap()) {
                const auto& localNoise = heightDifference["local_noise_fallback"];
                heightCfg.local_noise_fallback_enabled = yaml_utils::getBool(
                    localNoise, "enabled", heightCfg.local_noise_fallback_enabled);
                heightCfg.local_noise_p75_max_abs_error = yaml_utils::getDouble(
                    localNoise, "p75_max_abs_error", heightCfg.local_noise_p75_max_abs_error);
                heightCfg.local_noise_min_containment = yaml_utils::getDouble(
                    localNoise, "min_containment", heightCfg.local_noise_min_containment);
            }
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

void Config::resetVisualization(PipelineConfig& cfg) {
    cfg.save_visuals = true;
    cfg.draw_keypoints = false;
    cfg.draw_matches = false;
    cfg.match_views.clear();
    cfg.max_matches_drawn = 100;
    cfg.save_originals = true;
    cfg.save_warped = true;
    cfg.save_blend = true;
    cfg.save_false_color_overlay = true;
    cfg.false_color_foreground_threshold = 0;
    cfg.save_foreground_masks = true;
    cfg.save_structure_responses = true;
    cfg.structure_match_source = StructureMatchSource::RAW;
    cfg.warp = true;
    cfg.show_source_window = false;
    cfg.show_target_window = false;
    cfg.show_warped_window = false;
    cfg.wait_key = 0;
}

void Config::applyVisualizationOverrides(PipelineConfig& cfg, const YAML::Node& vis) {
    if (!vis || !vis.IsMap()) {
        return;
    }
    if (vis["draw_keypoints"])
        cfg.draw_keypoints = yaml_utils::getBool(vis, "draw_keypoints", cfg.draw_keypoints);
    if (vis["draw_matches"])
        cfg.draw_matches = yaml_utils::getBool(vis, "draw_matches", cfg.draw_matches);
    if (vis["match_views"])
        cfg.match_views = parseMatchViews(vis, cfg.match_views);
    if (vis["max_matches_drawn"])
        cfg.max_matches_drawn = yaml_utils::getInt(vis, "max_matches_drawn", cfg.max_matches_drawn);
    if (vis["save_originals"])
        cfg.save_originals = yaml_utils::getBool(vis, "save_originals", cfg.save_originals);
    if (vis["save_warped"])
        cfg.save_warped = yaml_utils::getBool(vis, "save_warped", cfg.save_warped);
    if (vis["save_blend"])
        cfg.save_blend = yaml_utils::getBool(vis, "save_blend", cfg.save_blend);
    if (vis["save_false_color_overlay"])
        cfg.save_false_color_overlay = yaml_utils::getBool(
            vis, "save_false_color_overlay", cfg.save_false_color_overlay);
    if (vis["false_color_foreground_threshold"])
        cfg.false_color_foreground_threshold = yaml_utils::getInt(
            vis, "false_color_foreground_threshold", cfg.false_color_foreground_threshold);
    if (vis["save_foreground_masks"])
        cfg.save_foreground_masks = yaml_utils::getBool(
            vis, "save_foreground_masks", cfg.save_foreground_masks);
    if (vis["save_structure_responses"])
        cfg.save_structure_responses = yaml_utils::getBool(
            vis, "save_structure_responses", cfg.save_structure_responses);
    cfg.structure_match_source = parseStructureMatchSource(vis, cfg.structure_match_source);
    if (vis["warp"])
        cfg.warp = yaml_utils::getBool(vis, "warp", cfg.warp);
    if (vis["show_source_window"])
        cfg.show_source_window = yaml_utils::getBool(
            vis, "show_source_window", cfg.show_source_window);
    if (vis["show_target_window"])
        cfg.show_target_window = yaml_utils::getBool(
            vis, "show_target_window", cfg.show_target_window);
    if (vis["show_warped_window"])
        cfg.show_warped_window = yaml_utils::getBool(
            vis, "show_warped_window", cfg.show_warped_window);
    if (vis["wait_key"])
        cfg.wait_key = yaml_utils::getInt(vis, "wait_key", cfg.wait_key);
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
        applyVisualizationOverrides(cfg, node["visualization"]);
    }

    if (node["validation"] && node["validation"].IsMap()) {
        const YAML::Node validation = resolveValidationProfile(node["validation"], base);
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
            overlapCfg.foreground_threshold =
                yaml_utils::getInt(overlap, "foreground_threshold", overlapCfg.foreground_threshold);
            overlapCfg.containment_tolerance_pixels =
                yaml_utils::getInt(overlap,
                                   "containment_tolerance_pixels",
                                   overlapCfg.containment_tolerance_pixels);
        }
        if (validation["height_difference"] && validation["height_difference"].IsMap()) {
            const auto& heightDifference = validation["height_difference"];
            auto& heightCfg = cfg.warp_quality.height_difference;
            heightCfg.enabled = yaml_utils::getBool(heightDifference, "enabled", heightCfg.enabled);
            heightCfg.compensate_global_height_offset = yaml_utils::getBool(
                heightDifference,
                "compensate_global_height_offset",
                heightCfg.compensate_global_height_offset);
            heightCfg.percentile = yaml_utils::getInt(heightDifference, "percentile", heightCfg.percentile);
            heightCfg.max_abs_error = yaml_utils::getDouble(
                heightDifference, "max_abs_error", heightCfg.max_abs_error);
            if (heightDifference["local_noise_fallback"] &&
                heightDifference["local_noise_fallback"].IsMap()) {
                const auto& localNoise = heightDifference["local_noise_fallback"];
                heightCfg.local_noise_fallback_enabled = yaml_utils::getBool(
                    localNoise, "enabled", heightCfg.local_noise_fallback_enabled);
                heightCfg.local_noise_p75_max_abs_error = yaml_utils::getDouble(
                    localNoise, "p75_max_abs_error", heightCfg.local_noise_p75_max_abs_error);
                heightCfg.local_noise_min_containment = yaml_utils::getDouble(
                    localNoise, "min_containment", heightCfg.local_noise_min_containment);
            }
        }

        if (validation["edge_structure_diagnostic"] &&
            validation["edge_structure_diagnostic"].IsMap()) {
            const auto& diagnostic = validation["edge_structure_diagnostic"];
            auto& edgeCfg = cfg.warp_quality.edge_structure_diagnostic;
            edgeCfg.enabled = yaml_utils::getBool(diagnostic, "enabled", edgeCfg.enabled);
            edgeCfg.visibility_threshold = yaml_utils::getInt(
                diagnostic, "visibility_threshold", edgeCfg.visibility_threshold);
            edgeCfg.min_foreground_elongation_ratio = yaml_utils::getDouble(
                diagnostic,
                "min_foreground_elongation_ratio",
                edgeCfg.min_foreground_elongation_ratio);
            edgeCfg.min_axis_occupancy = yaml_utils::getDouble(
                diagnostic, "min_axis_occupancy", edgeCfg.min_axis_occupancy);
            edgeCfg.max_centerline_deviation_ratio = yaml_utils::getDouble(
                diagnostic,
                "max_centerline_deviation_ratio",
                edgeCfg.max_centerline_deviation_ratio);
            edgeCfg.max_canvas_side_pixels = yaml_utils::getInt(
                diagnostic, "max_canvas_side_pixels", edgeCfg.max_canvas_side_pixels);
            edgeCfg.max_canvas_pixels = yaml_utils::getInt(
                diagnostic, "max_canvas_pixels", edgeCfg.max_canvas_pixels);
            edgeCfg.duplicate_line_normal_tolerance_pixels = yaml_utils::getDouble(
                diagnostic,
                "duplicate_line_normal_tolerance_pixels",
                edgeCfg.duplicate_line_normal_tolerance_pixels);
            edgeCfg.duplicate_line_min_span_overlap_ratio = yaml_utils::getDouble(
                diagnostic,
                "duplicate_line_min_span_overlap_ratio",
                edgeCfg.duplicate_line_min_span_overlap_ratio);
            edgeCfg.outer_longitudinal_edge_min_normal_separation_pixels =
                yaml_utils::getDouble(
                    diagnostic,
                    "outer_longitudinal_edge_min_normal_separation_pixels",
                    edgeCfg.outer_longitudinal_edge_min_normal_separation_pixels);
            edgeCfg.outer_longitudinal_edge_max_normal_separation_pixels =
                yaml_utils::getDouble(
                    diagnostic,
                    "outer_longitudinal_edge_max_normal_separation_pixels",
                    edgeCfg.outer_longitudinal_edge_max_normal_separation_pixels);
            edgeCfg.outer_longitudinal_edge_min_span_overlap_ratio = yaml_utils::getDouble(
                diagnostic,
                "outer_longitudinal_edge_min_span_overlap_ratio",
                edgeCfg.outer_longitudinal_edge_min_span_overlap_ratio);
            edgeCfg.min_fragment_length_pixels = yaml_utils::getDouble(
                diagnostic,
                "min_fragment_length_pixels",
                edgeCfg.min_fragment_length_pixels);
            edgeCfg.group_max_angle_difference_degrees = yaml_utils::getDouble(
                diagnostic,
                "group_max_angle_difference_degrees",
                edgeCfg.group_max_angle_difference_degrees);
            edgeCfg.group_max_normal_distance_pixels = yaml_utils::getDouble(
                diagnostic,
                "group_max_normal_distance_pixels",
                edgeCfg.group_max_normal_distance_pixels);
            edgeCfg.post_fit_group_normal_distance_pixels = yaml_utils::getDouble(
                diagnostic,
                "post_fit_group_normal_distance_pixels",
                edgeCfg.post_fit_group_normal_distance_pixels);
            edgeCfg.min_line_group_actual_length_pixels = yaml_utils::getDouble(
                diagnostic,
                "min_line_group_actual_length_pixels",
                edgeCfg.min_line_group_actual_length_pixels);
            edgeCfg.min_line_group_continuity_ratio = yaml_utils::getDouble(
                diagnostic,
                "min_line_group_continuity_ratio",
                edgeCfg.min_line_group_continuity_ratio);
            edgeCfg.max_line_group_gap_ratio = yaml_utils::getDouble(
                diagnostic,
                "max_line_group_gap_ratio",
                edgeCfg.max_line_group_gap_ratio);
            edgeCfg.max_fragment_direction_spread_degrees = yaml_utils::getDouble(
                diagnostic,
                "max_fragment_direction_spread_degrees",
                edgeCfg.max_fragment_direction_spread_degrees);
            edgeCfg.max_line_fit_residual_pixels = yaml_utils::getDouble(
                diagnostic,
                "max_line_fit_residual_pixels",
                edgeCfg.max_line_fit_residual_pixels);

            edgeCfg.min_main_line_actual_length_ratio = yaml_utils::getDouble(
                diagnostic,
                "min_main_line_actual_length_ratio",
                edgeCfg.min_main_line_actual_length_ratio);
            edgeCfg.direction_cluster_tolerance_degrees = yaml_utils::getDouble(
                diagnostic,
                "direction_cluster_tolerance_degrees",
                edgeCfg.direction_cluster_tolerance_degrees);
            edgeCfg.min_main_direction_support_ratio = yaml_utils::getDouble(
                diagnostic,
                "min_main_direction_support_ratio",
                edgeCfg.min_main_direction_support_ratio);
            edgeCfg.max_main_direction_spread_degrees = yaml_utils::getDouble(
                diagnostic,
                "max_main_direction_spread_degrees",
                edgeCfg.max_main_direction_spread_degrees);
            edgeCfg.min_main_direction_margin = yaml_utils::getDouble(
                diagnostic,
                "min_main_direction_margin",
                edgeCfg.min_main_direction_margin);
            edgeCfg.max_main_direction_difference_degrees = yaml_utils::getDouble(
                diagnostic,
                "max_main_direction_difference_degrees",
                edgeCfg.max_main_direction_difference_degrees);
            edgeCfg.max_axis_classification_error_degrees = yaml_utils::getDouble(
                diagnostic,
                "max_axis_classification_error_degrees",
                edgeCfg.max_axis_classification_error_degrees);
            edgeCfg.min_horizontal_actual_length_ratio = yaml_utils::getDouble(
                diagnostic,
                "min_horizontal_actual_length_ratio",
                edgeCfg.min_horizontal_actual_length_ratio);
            edgeCfg.min_vertical_actual_length_ratio = yaml_utils::getDouble(
                diagnostic,
                "min_vertical_actual_length_ratio",
                edgeCfg.min_vertical_actual_length_ratio);
            edgeCfg.max_vertical_unmatched_length_ratio = yaml_utils::getDouble(
                diagnostic,
                "max_vertical_unmatched_length_ratio",
                edgeCfg.max_vertical_unmatched_length_ratio);

            edgeCfg.profile_smoothing_sigma = yaml_utils::getDouble(
                diagnostic, "profile_smoothing_sigma", edgeCfg.profile_smoothing_sigma);
            edgeCfg.min_peak_prominence = yaml_utils::getDouble(
                diagnostic, "min_peak_prominence", edgeCfg.min_peak_prominence);
            edgeCfg.candidate_position_tolerance_pixels = yaml_utils::getDouble(
                diagnostic,
                "candidate_position_tolerance_pixels",
                edgeCfg.candidate_position_tolerance_pixels);
            edgeCfg.final_position_tolerance_pixels = yaml_utils::getDouble(
                diagnostic,
                "final_position_tolerance_pixels",
                edgeCfg.final_position_tolerance_pixels);
            edgeCfg.candidate_min_span_overlap_ratio = yaml_utils::getDouble(
                diagnostic,
                "candidate_min_span_overlap_ratio",
                edgeCfg.candidate_min_span_overlap_ratio);
            edgeCfg.min_shorter_line_overlap_ratio = yaml_utils::getDouble(
                diagnostic,
                "min_shorter_line_overlap_ratio",
                edgeCfg.min_shorter_line_overlap_ratio);
            edgeCfg.max_line_pair_angle_difference_degrees = yaml_utils::getDouble(
                diagnostic,
                "max_line_pair_angle_difference_degrees",
                edgeCfg.max_line_pair_angle_difference_degrees);
            edgeCfg.match_position_cost_weight = yaml_utils::getDouble(
                diagnostic, "match_position_cost_weight", edgeCfg.match_position_cost_weight);
            edgeCfg.match_overlap_cost_weight = yaml_utils::getDouble(
                diagnostic, "match_overlap_cost_weight", edgeCfg.match_overlap_cost_weight);
            edgeCfg.match_angle_cost_weight = yaml_utils::getDouble(
                diagnostic, "match_angle_cost_weight", edgeCfg.match_angle_cost_weight);
            edgeCfg.match_prominence_cost_weight = yaml_utils::getDouble(
                diagnostic,
                "match_prominence_cost_weight",
                edgeCfg.match_prominence_cost_weight);
            edgeCfg.min_strong_line_actual_length_pixels = yaml_utils::getDouble(
                diagnostic,
                "min_strong_line_actual_length_pixels",
                edgeCfg.min_strong_line_actual_length_pixels);
            edgeCfg.min_strong_peak_prominence = yaml_utils::getDouble(
                diagnostic,
                "min_strong_peak_prominence",
                edgeCfg.min_strong_peak_prominence);
            edgeCfg.ambiguity_score_margin = yaml_utils::getDouble(
                diagnostic, "ambiguity_score_margin", edgeCfg.ambiguity_score_margin);
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
        if (validation["method_quality"] && validation["method_quality"].IsMap()) {
            const auto& methodQuality = validation["method_quality"];
            cfg.validate_method_quality = yaml_utils::getBool(methodQuality, "enabled", false);
            cfg.fail_on_method_quality =
                yaml_utils::getBool(methodQuality, "fail_on_violation", true);
            cfg.min_method_keypoints =
                yaml_utils::getInt(methodQuality, "min_keypoints", 0);
            cfg.min_method_structures =
                yaml_utils::getInt(methodQuality, "min_structures", 0);
            cfg.min_method_structure_matches =
                yaml_utils::getInt(methodQuality, "min_structure_matches", 0);
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
