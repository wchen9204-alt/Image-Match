#include "pipeline/keypoint_pipeline.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "core/config.h"
#include "core/factory.h"
#include "core/types.h"
#include "utils/logger.h"
#include "utils/string_utils.h"
#include "utils/timer.h"
#include "utils/visualization/draw_inliers.h"
#include "utils/visualization/draw_matches.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

// 将匹配器类型别名归一到稳定标签，用于输出命名和日志统计。
std::string normalizeMatcherType(const std::string& raw_type) {
    const std::string type = string_utils::toUpperAscii(raw_type);
    if (type == "BF" || type == "BFMATCHER" || type == "BRUTE_FORCE") {
        return "BF";
    }
    if (type == "FLANN" || type == "FLANNBASED" || type == "FLANNMATCHER") {
        return "FLANN";
    }
    return type.empty() ? "MATCHER" : type;
}

// 将匹配接口名称归一化，避免配置别名导致输出目录标签分裂。
std::string normalizeMatchMethod(const std::string& raw_method) {
    const std::string method = string_utils::toUpperAscii(raw_method);
    if (method == "MATCH" || method == "TOP1" || method == "NN") {
        return "MATCH";
    }
    if (method == "KNN" || method == "KNNMATCH") {
        return "KNN";
    }
    if (method == "RADIUS" || method == "RADIUSMATCH") {
        return "RADIUS";
    }
    return method.empty() ? "MATCH" : method;
}

// 从匹配器 YAML 中提炼稳定的可读标签，便于批量实验结果按算法组合区分。
std::string buildMatcherLabel(const fs::path& matcher_path) {
    if (matcher_path.empty()) {
        return "MATCHER";
    }

    try {
        const YAML::Node cfg = Config::load(matcher_path);
        const std::string raw_type = cfg["type"] ? cfg["type"].as<std::string>() : "";
        const YAML::Node params = cfg["params"];
        std::string raw_method;
        if (params && params["method"]) {
            raw_method = params["method"].as<std::string>();
        } else {
            const std::string type = normalizeMatcherType(raw_type);
            raw_method = (type == "FLANN") ? "KNN" : "MATCH";
        }

        return normalizeMatcherType(raw_type) + "_" + normalizeMatchMethod(raw_method);
    } catch (const std::exception&) {
        return string_utils::toUpperAscii(matcher_path.stem().string());
    }
}

void removeStaleKeypointVariants(const fs::path& dir,
                                 const std::string& keypoint_stem,
                                 const std::string& suffix,
                                 const fs::path& keep) {
    if (dir.empty()) {
        return;
    }

    std::error_code ec;
    if (!fs::exists(dir, ec)) {
        return;
    }

    const std::string prefix = keypoint_stem + "_";
    const fs::path keep_path = fs::weakly_canonical(keep, ec);
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec || !entry.is_regular_file()) {
            continue;
        }

        const std::string name = entry.path().filename().string();
        const bool has_prefix = name.rfind(prefix, 0) == 0;
        const bool has_suffix =
            name.size() >= suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
        if (!has_prefix || !has_suffix) {
            continue;
        }

        std::error_code path_ec;
        const fs::path candidate = fs::weakly_canonical(entry.path(), path_ec);
        if (!path_ec && !keep_path.empty() && candidate == keep_path) {
            continue;
        }

        std::error_code remove_ec;
        fs::remove(entry.path(), remove_ec);
        if (remove_ec) {
            IR_LOG_WARN("Failed to remove stale keypoints visualization: ",
                        entry.path().string());
        } else {
            IR_LOG_INFO("Removed stale keypoints visualization: ", entry.path().string());
        }
    }
}

struct MatchViewOutput {
    MatchView view;
    fs::path dir;
    std::string suffix;
    std::string log_label;
};

} // namespace

void KeypointPipeline::resetStages() {
    _extractor.reset();
    _matcher.reset();
    _filters.clear();
    _geometry.reset();
}

bool KeypointPipeline::configureStages(const PipelineConfig& cfg) {
    // 这里仍沿用 KeypointPipeline / keypoint_path 命名，但目录语义已经切到 keypoint。
    // 1. 从 keypoint YAML 创建关键点提取器。
    _extractor = Factory::createKeypointExtractor(Config::load(cfg.keypoint_path));
    // 2. 从 matcher YAML 创建描述子匹配器。
    _matcher = Factory::createMatcher(Config::load(cfg.matcher_path));

    // 3. 按 pipeline YAML 顺序创建匹配过滤器链。
    for (const auto& fp : cfg.filter_paths) {
        _filters.push_back(Factory::createFilter(Config::load(fp)));
    }

    // 4. 创建几何估计器，后续由 runEstimation 调用。
    _geometry = Factory::createGeometryEstimator(Config::load(cfg.geometry_path));

    IR_LOG_INFO("KeypointPipeline stages configured: extractor=",
                _extractor->name(),
                ", matcher=",
                _matcher->name(),
                ", filters=",
                static_cast<int>(_filters.size()),
                ", geometry=",
                _geometry->name());
    return true;
}

bool KeypointPipeline::runExtraction(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_extract_ms);

    // 1. 检查提取器是否已经由 configureStages 创建。
    if (!_extractor) {
        IR_LOG_ERROR("runExtraction: no keypoint extractor configured.");
        return false;
    }

    // 2. 执行关键点检测和描述子计算，结果写入 ctx.keypoint_data。
    const bool ok = _extractor->extract(ctx);
    // 3. 将关键点数量回填到运行摘要，便于输出和批量统计。
    ctx.result.num_keypoints_first = static_cast<int>(ctx.keypoint_data.first.keypoints.size());
    ctx.result.num_keypoints_second = static_cast<int>(ctx.keypoint_data.second.keypoints.size());
    return ok;
}

bool KeypointPipeline::runAssociation(RegistrationContext& ctx) {
    // 1. 先执行描述子匹配，生成 neighbour_matches_by_query 或初始 filtered 匹配。
    if (!runMatch(ctx)) {
        return false;
    }

    // 2. 再按配置执行过滤器链；过滤器软失败时保留告警但不中断整体流程。
    if (!runFilters(ctx)) {
        IR_LOG_WARN("Some filter stage reported a soft failure.");
    }
    // 3. 过滤后的匹配会进入 runEstimation 做几何模型估计。
    return true;
}

bool KeypointPipeline::runMatch(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_match_ms);

    // 1. 检查匹配器是否已经由 configureStages 创建。
    if (!_matcher) {
        IR_LOG_ERROR("runMatch: no matcher configured.");
        return false;
    }

    // 2. 调用匹配器生成原始匹配结果。
    const bool ok = _matcher->match(ctx);

    // 3. 原始统计使用每个 query 的最佳候选，三种匹配方法语义一致。
    ctx.result.num_raw_matches = static_cast<int>(ctx.keypoint_match_data.raw_matches.size());    return ok;
}

bool KeypointPipeline::runFilters(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_filter_ms);
    auto& md = ctx.keypoint_match_data;

    // 1. 过滤链从匹配器给出的原始一对一候选开始。
    md.seedFilteredMatchesFromRaw();

    // 2. 按 YAML 顺序执行过滤器，每个过滤器都基于当前 filtered 继续筛选。
    bool ok = true;
    for (const auto& f : _filters) {
        if (!f) {
            continue;
        }
        if (!f->apply(ctx)) {
            IR_LOG_WARN("Filter '", f->name(), "' returned false.");
            ok = false;
        }
    }

    // 3. 所有过滤器都未保留候选时，回退到 raw，避免空集合直接进入几何阶段。
    if (md.filtered_matches.empty() && !md.raw_matches.empty()) {
        const size_t restored_count = md.raw_matches.size();
        md.restoreFilteredMatchesFromRaw();
        IR_LOG_WARN("All filters rejected candidates; restored ", restored_count, " raw matches.");
    }
    // 4. 记录最终进入几何估计的匹配数量。
    ctx.result.num_filtered_matches = static_cast<int>(md.filtered_matches.size());
    return ok;
}

bool KeypointPipeline::runEstimation(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_geometry_ms);

    // 1. 检查几何估计器是否已经由 configureStages 创建。
    if (!_geometry) {
        IR_LOG_ERROR("runEstimation: no geometry estimator configured.");
        ctx.geometry_data.message = "no geometry estimator configured";
        return false;
    }

    ctx.correspondence_source = "KEYPOINT";
    // 2. 使用过滤后的匹配估计几何模型，结果写入 ctx.geometry_data。
    const bool ok = _geometry->estimate(ctx);
    // 3. 将内点统计同步到通用运行摘要。
    ctx.result.num_inliers = ctx.geometry_data.num_inliers;
    ctx.result.inlier_ratio = ctx.geometry_data.inlier_ratio;
    return ok;
}

std::string KeypointPipeline::buildOutputStem(const RegistrationContext& ctx) const {
    const std::string matcher_label = buildMatcherLabel(_config.matcher_path);

    return ctx.image1_path.stem().string() + "_" + ctx.image2_path.stem().string() + "_" +
           (_extractor ? _extractor->name() : std::string("UNK")) + "_" +
           (_geometry ? toString(_geometry->type()) : std::string("UNK")) + "_" + matcher_label;
}

bool KeypointPipeline::saveOutputs(RegistrationContext& ctx) {
    if (_config.output_dir.empty()) {
        return true;
    }

    // 1. 准备点特征专属输出目录。
    const fs::path keypoints_dir = _config.output_dir / "keypoints";
    const fs::path all_match_dir = _config.output_dir / "all_match";
    const fs::path filter_match_dir = _config.output_dir / "filter_match";
    const fs::path inlier_match_dir = _config.output_dir / "inlier_match";
    std::error_code ec;
    fs::create_directories(keypoints_dir, ec);
    fs::create_directories(all_match_dir, ec);
    fs::create_directories(filter_match_dir, ec);
    fs::create_directories(inlier_match_dir, ec);

    const std::string stem = buildOutputStem(ctx);
    const std::string keypoint_stem =
        ctx.image1_path.stem().string() + "_" + ctx.image2_path.stem().string() + "_" +
        (_extractor ? _extractor->name() : std::string("UNK"));

    // 2. 按配置保存 source / target 关键点可视化。
    if (_config.draw_keypoints) {
        cv::Mat src_vis;
        cv::Mat dst_vis;
        cv::drawKeypoints(ctx.images.first,
                          ctx.keypoint_data.first.keypoints,
                          src_vis,
                          cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::drawKeypoints(ctx.images.second,
                          ctx.keypoint_data.second.keypoints,
                          dst_vis,
                          cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        if (!src_vis.empty()) {
            const fs::path out = keypoints_dir / (keypoint_stem + "_source_keypoints.png");
            if (cv::imwrite(out.string(), src_vis)) {
                IR_LOG_INFO("Wrote source keypoints visualization: ", out.string());
                removeStaleKeypointVariants(
                    keypoints_dir, keypoint_stem, "_source_keypoints.png", out);
            } else {
                IR_LOG_WARN("Failed to write source keypoints visualization: ", out.string());
            }
        }
        if (!dst_vis.empty()) {
            const fs::path out = keypoints_dir / (keypoint_stem + "_target_keypoints.png");
            if (cv::imwrite(out.string(), dst_vis)) {
                IR_LOG_INFO("Wrote target keypoints visualization: ", out.string());
                removeStaleKeypointVariants(
                    keypoints_dir, keypoint_stem, "_target_keypoints.png", out);
            } else {
                IR_LOG_WARN("Failed to write target keypoints visualization: ", out.string());
            }
        }
    }

    // 3. 按配置同时保存全部匹配和内点匹配两类连线图，便于对比过滤与几何估计效果。
    if (_config.draw_matches) {
        auto removeStale = [](const fs::path& out) {
            std::error_code remove_ec;
            if (fs::exists(out, remove_ec)) {
                fs::remove(out, remove_ec);
                if (remove_ec) {
                    IR_LOG_WARN("Failed to remove stale matches visualization: ", out.string());
                } else {
                    IR_LOG_INFO("Removed stale matches visualization: ", out.string());
                }
            }
        };

        const std::vector<MatchViewOutput> outputs = {
            {MatchView::RAW, all_match_dir, "_all_match.png", "all matches"},
            {MatchView::FILTERED, filter_match_dir, "_filter_match.png", "filtered matches"},
            {MatchView::INLIERS, inlier_match_dir, "_inlier_match.png", "inlier matches"}
        };

        for (const MatchView view : _config.match_views) {
            const auto it = std::find_if(outputs.begin(), outputs.end(), [&](const MatchViewOutput& output) {
                return output.view == view;
            });
            if (it == outputs.end()) {
                continue;
            }

            const fs::path out = it->dir / (stem + it->suffix);
            cv::Mat vis;
            if (view == MatchView::INLIERS) {
                DrawInliers::Options inlierOpt;
                inlierOpt.max_inliers = _config.max_matches_drawn;
                vis = DrawInliers::render(ctx, inlierOpt);
            } else {
                DrawMatches::Options matchOpt;
                matchOpt.draw_raw_matches = view == MatchView::RAW;
                matchOpt.max_matches = _config.max_matches_drawn;
                vis = DrawMatches::render(ctx, matchOpt);
            }

            if (!vis.empty()) {
                cv::imwrite(out.string(), vis);
                IR_LOG_INFO("Wrote ", it->log_label, " visualization: ", out.string());
            } else {
                removeStale(out);
            }
        }

        const fs::path old_matches_dir = _config.output_dir / "matches";
        removeStale(old_matches_dir / (stem + "_matches.png"));
        removeStale(old_matches_dir / (stem + "_matches_all.png"));
        removeStale(old_matches_dir / (stem + "_matches_inliers.png"));
    }

    // 4. 委托基类保存 originals / warped / blend 等通用输出。
    return BasePipeline::saveOutputs(ctx);
}

} // namespace ir



