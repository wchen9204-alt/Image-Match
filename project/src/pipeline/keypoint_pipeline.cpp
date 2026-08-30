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
#include "evaluator/quality/warp_quality_evaluator.h"
#include "geometry/multilayer_dark_rigid_estimator.h"
#include "matcher/keypoint/multilayer_dark_bf_matcher.h"
#include "keypoint/multilayer_dark_keypoint_extractor.h"
#include "pipeline/base_pipeline_helpers.h"
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

bool KeypointPipeline::run(RegistrationContext& ctx, const PipelineRunOptions& options) {
    // 每个新样本先使用 YAML 指定的首选点特征；本次回退不会影响下一样本。
    _active_extractor = _primary_extractor;
    _active_matcher = _matcher;
    _active_geometry = _geometry;
    return BasePipeline::run(ctx, options);
}

void KeypointPipeline::resetStages() {
    _primary_extractor.reset();
    _multilayer_dark_extractor.reset();
    _active_extractor.reset();
    _matcher.reset();
    _multilayer_dark_matcher.reset();
    _active_matcher.reset();
    _filters.clear();
    _geometry.reset();
    _multilayer_dark_geometry.reset();
    _active_geometry.reset();
}

bool KeypointPipeline::configureStages(const PipelineConfig& cfg) {
    // 这里仍沿用 KeypointPipeline / keypoint_path 命名，但目录语义已经切到 keypoint。
    // 1. 从 keypoint YAML 创建首选点特征提取器。
    const YAML::Node keypoint_cfg = Config::load(cfg.keypoint_path);
    _primary_extractor = Factory::createKeypointExtractor(keypoint_cfg);
    _active_extractor = _primary_extractor;

    // 2. 多层暗部复用同一份点特征 YAML，所有已支持的点特征类型均可使用。
    if (cfg.multilayer_dark_fallback.enabled) {
        _multilayer_dark_extractor = std::make_shared<MultilayerDarkKeypointExtractor>(
            keypoint_cfg, cfg.multilayer_dark_fallback.layer_count);
    }

    // 3. 从 matcher YAML 创建描述子匹配器。
    const YAML::Node matcher_cfg = Config::load(cfg.matcher_path);
    _matcher = Factory::createMatcher(matcher_cfg);
    _active_matcher = _matcher;
    if (cfg.multilayer_dark_fallback.enabled) {
        _multilayer_dark_matcher = std::make_shared<MultilayerDarkBfMatcher>(matcher_cfg);
    }

    // 4. 按 pipeline YAML 顺序创建匹配过滤器链。
    for (const auto& fp : cfg.filter_paths) {
        _filters.push_back(Factory::createFilter(Config::load(fp)));
    }

    // 5. 创建几何估计器，后续由 runEstimation 调用。
    const YAML::Node geometry_cfg = Config::load(cfg.geometry_path);
    _geometry = Factory::createGeometryEstimator(geometry_cfg);
    _active_geometry = _geometry;
    if (cfg.multilayer_dark_fallback.enabled) {
        _multilayer_dark_geometry = std::make_shared<MultilayerDarkRigidEstimator>(geometry_cfg);
    }

    IR_LOG_INFO("KeypointPipeline stages configured: extractor=",
                _primary_extractor->name(),
                ", matcher=",
                _matcher->name(),
                ", filters=",
                static_cast<int>(_filters.size()),
                ", geometry=",
                _geometry->name(),
                ", multilayer_dark_fallback=",
                _multilayer_dark_extractor ? "enabled" : "disabled");
    return true;
}

bool KeypointPipeline::runExtraction(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_extract_ms);

    // 1. 检查提取器是否已经由 configureStages 创建。
    if (!_active_extractor) {
        IR_LOG_ERROR("runExtraction: no keypoint extractor configured.");
        return false;
    }

    // 2. 执行关键点检测和描述子计算，结果写入 ctx.keypoint_data。
    const bool ok = _active_extractor->extract(ctx);
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
    if (!_active_matcher) {
        IR_LOG_ERROR("runMatch: no matcher configured.");
        return false;
    }

    // 2. 调用匹配器生成原始匹配结果。
    const bool ok = _active_matcher->match(ctx);

    // 3. 原始统计使用每个 query 的最佳候选，三种匹配方法语义一致。
    ctx.result.num_raw_matches = static_cast<int>(ctx.keypoint_match_data.raw_matches.size());
    return ok;
}

bool KeypointPipeline::runFilters(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_filter_ms);
    auto& md = ctx.keypoint_match_data;

    // 多层暗部匹配器已经按参考流程生成扩展候选，不再经过普通过滤器链。
    if (_active_matcher != _matcher) {
        ctx.result.num_filtered_matches = static_cast<int>(md.filtered_matches.size());
        return !md.filtered_matches.empty();
    }

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
    if (!_active_geometry) {
        IR_LOG_ERROR("runEstimation: no geometry estimator configured.");
        ctx.geometry_data.message = "no geometry estimator configured";
        return false;
    }

    ctx.correspondence_source = "KEYPOINT";
    // 2. 使用过滤后的匹配估计几何模型，结果写入 ctx.geometry_data。
    const bool ok = _active_geometry->estimate(ctx);
    // 3. 将内点统计同步到通用运行摘要。
    ctx.result.num_inliers = ctx.geometry_data.num_inliers;
    ctx.result.inlier_ratio = ctx.geometry_data.inlier_ratio;
    if (_active_geometry != _geometry) {
        ctx.result.num_filtered_matches =
            static_cast<int>(ctx.keypoint_match_data.filtered_matches.size());
    }
    return ok;
}

bool KeypointPipeline::shouldUseMultilayerDarkFallback(
    const RegistrationAttemptFailure failure) const {
    // 1. 首选 FAST 方案，提取、匹配、估计、变换或质量验证失败后都可由多层暗部重试。
    //    读图失败发生在尝试流程外，两套方案都没有输入，因此不在这里处理。
    switch (failure) {
    case RegistrationAttemptFailure::EXTRACTION:
    case RegistrationAttemptFailure::ASSOCIATION:
    case RegistrationAttemptFailure::ESTIMATION:
    case RegistrationAttemptFailure::WARP:
    case RegistrationAttemptFailure::QUALITY:
        break;
    default:
        return false;
    }
    if (!_config.multilayer_dark_fallback.enabled ||
        !_multilayer_dark_extractor || !_multilayer_dark_matcher ||
        !_multilayer_dark_geometry ||
        _active_extractor != _primary_extractor) {
        return false;
    }
    return true;
}

bool KeypointPipeline::passMultilayerDarkFallbackGate(
    const RegistrationContext& ctx,
    std::string& failure_message) const {
    const auto& gate = _config.multilayer_dark_fallback;

    // 1. 先检查几何估计得到的内点率。
    if (ctx.result.inlier_ratio < gate.min_inlier_ratio) {
        failure_message = "multilayer fallback gate: inlier ratio " +
                           std::to_string(ctx.result.inlier_ratio) + " < " +
                           std::to_string(gate.min_inlier_ratio);
        return false;
    }

    // 2. 读取当前首选几何结果，使用同一个 source -> target 变换评估图像质量。
    cv::Mat matrix;
    if (!base_pipeline_helpers::activeTransformMatrix(ctx, matrix)) {
        failure_message = "multilayer fallback gate: no transform matrix";
        return false;
    }

    // 3. 按参考 FAST 流程分别计算 source / target 前景重合率；仅两者都不足时拒绝。
    cv::Mat source_mask;
    cv::Mat target_mask;
    cv::Mat warped_source_mask;
    if (!base_pipeline_helpers::buildForegroundMask(ctx.images.first, 0, source_mask) ||
        !base_pipeline_helpers::buildForegroundMask(ctx.images.second, 0, target_mask) ||
        !base_pipeline_helpers::warpMaskToTargetSize(
            source_mask, target_mask.size(), matrix, warped_source_mask)) {
        failure_message = "multilayer fallback gate: cannot build foreground overlap masks";
        return false;
    }

    cv::Mat overlap_mask;
    cv::bitwise_and(warped_source_mask, target_mask, overlap_mask);
    const int source_foreground = cv::countNonZero(source_mask);
    const int target_foreground = cv::countNonZero(target_mask);
    if (source_foreground <= 0 || target_foreground <= 0) {
        failure_message = "multilayer fallback gate: empty foreground";
        return false;
    }

    const int overlap_foreground = cv::countNonZero(overlap_mask);
    const double source_overlap =
        static_cast<double>(overlap_foreground) / static_cast<double>(source_foreground);
    const double target_overlap =
        static_cast<double>(overlap_foreground) / static_cast<double>(target_foreground);
    if (source_overlap < gate.min_foreground_overlap_ratio &&
        target_overlap < gate.min_foreground_overlap_ratio) {
        failure_message = "multilayer fallback gate: source overlap " +
                          std::to_string(source_overlap) + ", target overlap " +
                          std::to_string(target_overlap) + " < " +
                          std::to_string(gate.min_foreground_overlap_ratio);
        return false;
    }

    // 4. 高度差门控使用独立阈值。
    warp_quality::WarpQualityOptions options;
    options.validate_containment = false;
    options.foreground_threshold = 0;
    options.validate_height_difference = true;
    options.compensate_global_height_offset = false;
    options.height_difference_percentile = 90;
    options.max_height_difference_error = gate.max_height_difference;
    options.allow_local_noise_fallback = false;

    // 5. 高度差只在 source 与 target 的前景重合区域内计算，并直接使用原始 P90。
    warp_quality::WarpQualityResult quality;
    const bool evaluated = warp_quality::evaluateWarpQuality(
        options,
        ctx.images.first,
        ctx.images.second,
        matrix,
        ctx.warped_image,
        quality);
    if (!evaluated || !quality.pass) {
        failure_message = quality.message.empty()
            ? "multilayer fallback gate: gray difference failed"
            : "multilayer fallback gate: " + quality.message;
        return false;
    }
    return true;
}

bool KeypointPipeline::runPreQualityGate(RegistrationContext& ctx,
                                         std::string& failure_message) {
    // 1. 只对启用 fallback 的首选方案执行门控，多层暗部重试直接进入正常评估。
    if (!_config.multilayer_dark_fallback.enabled ||
        _active_extractor != _primary_extractor) {
        failure_message.clear();
        return true;
    }
    // 2. 门控失败会在最终质量验证前返回，避免保存首选方案的中间输出。
    return passMultilayerDarkFallbackGate(ctx, failure_message);
}

void KeypointPipeline::clearAttemptArtifacts(RegistrationContext& ctx) const {
    // 1. 图像和运行路径可供重试复用，只清理与首选提取结果绑定的中间数据。
    const double load_time_ms = ctx.result.t_load_ms;
    ctx.keypoint_data.clear();
    ctx.keypoint_match_data.clear();
    ctx.correspondence_source.clear();
    ctx.correspondence_snapshot.reset();
    ctx.geometry_data.clear();
    ctx.transform_data.clear();
    ctx.evaluation.clear();
    ctx.warped_image.release();
    // 2. 最终摘要只记录多层暗部这次结果，读图耗时仍归属于整个样本。
    ctx.result = RegistrationResult{};
    ctx.result.t_load_ms = load_time_ms;
}

bool KeypointPipeline::activateFallback(RegistrationContext& ctx,
                                        const RegistrationAttemptFailure failure,
                                        const std::string& failure_message) {
    if (!shouldUseMultilayerDarkFallback(failure)) {
        return false;
    }

    IR_LOG_WARN("首选点特征方案失败（",
                failure_message,
                "），切换到多层暗部提取。");
    clearAttemptArtifacts(ctx);
    _active_extractor = _multilayer_dark_extractor;
    _active_matcher = _multilayer_dark_matcher;
    _active_geometry = _multilayer_dark_geometry;
    return true;
}

std::string KeypointPipeline::buildOutputStem(const RegistrationContext& ctx) const {
    (void)ctx;
    return _active_extractor ? _active_extractor->name() : std::string("KEYPOINT");
}

bool KeypointPipeline::saveOutputs(RegistrationContext& ctx) {
    // 保存前记录最终 active 方案，供批处理报告展示。
    ctx.result.registration_strategy =
        _active_extractor == _multilayer_dark_extractor ? "MULTILAYER" : "FAST";
    if (!_config.save_visuals || ctx.output_dir.empty()) {
        return true;
    }

    // 1. 准备点特征专属输出目录。
    const fs::path keypoints_dir = ctx.output_dir / "keypoints";
    const fs::path match_dir = ctx.output_dir / "match";
    std::error_code ec;
    fs::create_directories(keypoints_dir, ec);
    fs::create_directories(match_dir, ec);

    const std::string stem = buildOutputStem(ctx);
    const std::string keypoint_stem = stem;

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
            {MatchView::RAW, match_dir, "_all_match.png", "all matches"},
            {MatchView::FILTERED, match_dir, "_filter_match.png", "filtered matches"},
            {MatchView::INLIERS, match_dir, "_inlier_match.png", "inlier matches"}
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

        const fs::path old_matches_dir = ctx.output_dir / "matches";
        removeStale(old_matches_dir / (stem + "_matches.png"));
        removeStale(old_matches_dir / (stem + "_matches_all.png"));
        removeStale(old_matches_dir / (stem + "_matches_inliers.png"));
    }

    // 4. 委托基类保存 originals / warped / blend 等通用输出。
    return BasePipeline::saveOutputs(ctx);
}

} // namespace ir



