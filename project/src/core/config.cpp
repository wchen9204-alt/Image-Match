#include "core/config.h"

#include <fstream>
#include <stdexcept>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace fs = std::filesystem;

namespace ir {

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
            cfg.validate_warp_overlap = yaml_utils::getBool(overlap, "enabled", false);
            cfg.min_warp_overlap_iou = yaml_utils::getDouble(overlap, "min_iou", 0.20);
            cfg.warp_overlap_foreground_threshold =
                yaml_utils::getInt(overlap, "foreground_threshold", 10);
        }
        if (validation["photometric"] && validation["photometric"].IsMap()) {
            const auto& photometric = validation["photometric"];
            cfg.validate_warp_photometric =
                yaml_utils::getBool(photometric, "enabled", false);
            cfg.max_warp_photometric_error =
                yaml_utils::getDouble(photometric, "max_nmad", 0.15);
        }
    }

    IR_LOG_INFO("Pipeline '", cfg.name, "' loaded from ", path.string());
    IR_LOG_INFO("  keypoint : ", cfg.keypoint_path.string());
    IR_LOG_INFO("  structure: ", cfg.structure_path.string());
    IR_LOG_INFO("  matcher  : ", cfg.matcher_path.string());
    for (const auto& f : cfg.filter_paths) {
        IR_LOG_INFO("  filter   : ", f.string());
    }
    IR_LOG_INFO("  geometry : ", cfg.geometry_path.string());
    IR_LOG_INFO("  image1   : ", cfg.image1_path.string());
    IR_LOG_INFO("  image2   : ", cfg.image2_path.string());
    IR_LOG_INFO("  output   : ", cfg.output_dir.string());

    return cfg;
}

} // namespace ir
