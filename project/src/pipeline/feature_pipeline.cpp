#include "pipeline/feature_pipeline.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "core/config.h"
#include "core/factory.h"
#include "core/types.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/visualization/draw_matches.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

// 统一将配置中的枚举样式字符串折叠为大写 ASCII，降低 YAML 写法差异对分支逻辑的影响。
std::string toUpperAscii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

// 将匹配器类型别名归一到稳定标签，用于输出命名和日志统计。
std::string normalizeMatcherType(const std::string& rawType) {
    const std::string type = toUpperAscii(rawType);
    if (type == "BF" || type == "BFMATCHER" || type == "BRUTE_FORCE") {
        return "BF";
    }
    if (type == "FLANN" || type == "FLANNBASED" || type == "FLANNMATCHER") {
        return "FLANN";
    }
    return type.empty() ? "MATCHER" : type;
}

// 将匹配接口名称归一化，避免配置别名导致输出目录标签分裂。
std::string normalizeMatchMethod(const std::string& rawMethod) {
    const std::string method = toUpperAscii(rawMethod);
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
std::string buildMatcherLabel(const fs::path& matcherPath) {
    if (matcherPath.empty()) {
        return "MATCHER";
    }

    try {
        const YAML::Node cfg = Config::load(matcherPath);
        const std::string rawType = cfg["type"] ? cfg["type"].as<std::string>() : "";
        const YAML::Node params = cfg["params"];
        std::string rawMethod;
        if (params && params["method"]) {
            rawMethod = params["method"].as<std::string>();
        } else {
            const std::string type = normalizeMatcherType(rawType);
            rawMethod = (type == "FLANN") ? "KNN" : "MATCH";
        }

        return normalizeMatcherType(rawType) + "_" + normalizeMatchMethod(rawMethod);
    } catch (const std::exception&) {
        return toUpperAscii(matcherPath.stem().string());
    }
}

} // namespace

void FeaturePipeline::resetStages() {
    // 1. 释放单例式阶段组件，避免下一次 configure 复用旧配置。
    _extractor.reset();
    _matcher.reset();

    // 2. 清空过滤器链和几何估计器，使流水线回到未装配状态。
    _filters.clear();
    _geometry.reset();
}

bool FeaturePipeline::configureStages(const PipelineConfig& cfg) {
    // 1. 按 feature/matcher 子配置创建点特征提取器和描述子匹配器。
    _extractor = Factory::createFeatureExtractor(Config::load(cfg.feature_path));
    _matcher = Factory::createMatcher(Config::load(cfg.matcher_path));

    // 2. 按 pipeline 中声明的顺序创建过滤器链。
    for (const auto& fp : cfg.filter_paths) {
        _filters.push_back(Factory::createFilter(Config::load(fp)));
    }

    // 3. 创建几何估计器，后续从过滤后的匹配中求解空间模型。
    _geometry = Factory::createGeometryEstimator(Config::load(cfg.geometry_path));

    IR_LOG_INFO("FeaturePipeline stages configured: extractor=",
                _extractor->name(),
                ", matcher=",
                _matcher->name(),
                ", filters=",
                static_cast<int>(_filters.size()),
                ", geometry=",
                _geometry->name());
    return true;
}

bool FeaturePipeline::runExtraction(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_extract_ms);

    // 1. 检查特征提取器是否已由 configureStages 成功装配。
    if (!_extractor) {
        IR_LOG_ERROR("runExtraction: no feature extractor configured.");
        return false;
    }

    // 2. 调用具体特征算子，结果写入 ctx.feature_data。
    const bool ok = _extractor->extract(ctx);

    // 3. 统一回填摘要和汇总表使用的关键点数量。
    ctx.result.num_keypoints_first = static_cast<int>(ctx.feature_data.first.keypoints.size());
    ctx.result.num_keypoints_second = static_cast<int>(ctx.feature_data.second.keypoints.size());
    return ok;
}

bool FeaturePipeline::runAssociation(RegistrationContext& ctx) {
    // 1. 先生成原始描述子匹配结果。
    if (!runMatch(ctx)) {
        return false;
    }

    // 2. 再按配置执行过滤器链；过滤软失败时保留已有中间结果继续尝试几何估计。
    if (!runFilters(ctx)) {
        IR_LOG_WARN("Some filter stage reported a soft failure.");
    }
    return true;
}

bool FeaturePipeline::runMatch(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_match_ms);

    // 1. 检查匹配器是否已由 configureStages 成功装配。
    if (!_matcher) {
        IR_LOG_ERROR("runMatch: no matcher configured.");
        return false;
    }

    // 2. 执行具体匹配器，可能产生一对一匹配或 k 近邻匹配。
    const bool ok = _matcher->match(ctx);

    // 3. 统计原始候选规模，便于终端摘要和汇总表横向比较。
    int rawMatchCount = 0;
    for (const auto& neighbours : ctx.match_data.raw_knn) {
        rawMatchCount += static_cast<int>(neighbours.size());
    }
    ctx.result.num_raw_matches = rawMatchCount;
    return ok;
}

bool FeaturePipeline::runFilters(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_filter_ms);
    auto& md = ctx.match_data;

    // 1. 若匹配器只产生 raw_knn，则先取 top-1 作为一对一候选，供后续过滤器消费。
    if (!md.raw_knn.empty() && md.filtered.empty()) {
        md.filtered.reserve(md.raw_knn.size());
        for (const auto& nb : md.raw_knn) {
            if (!nb.empty())
                md.filtered.push_back(nb.front());
        }
        IR_LOG_INFO("Seeded filtered matches from raw_knn top-1: ",
                    md.filtered.size(),
                    " / ",
                    md.raw_knn.size());
    }

    // 2. 按配置顺序执行过滤器链，逐步收缩候选匹配集合。
    bool ok = true;
    for (const auto& f : _filters) {
        if (!f)
            continue;
        if (!f->apply(ctx)) {
            IR_LOG_WARN("Filter '", f->name(), "' returned false.");
            ok = false;
        }
    }

    // 3. 回填过滤后的候选数量。
    ctx.result.num_filtered_matches = static_cast<int>(md.filtered.size());
    return ok;
}

bool FeaturePipeline::runEstimation(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_geometry_ms);

    // 1. 检查几何估计器是否已由 configureStages 成功装配。
    if (!_geometry) {
        IR_LOG_ERROR("runEstimation: no geometry estimator configured.");
        ctx.geometry_data.message = "no geometry estimator configured";
        return false;
    }

    // 2. 使用过滤后的匹配估计几何模型，结果写入 ctx.geometry_data。
    const bool ok = _geometry->estimate(ctx);

    // 3. 将几何质量指标同步到总结果，供摘要、汇总表和评测模块使用。
    ctx.result.num_inliers = ctx.geometry_data.num_inliers;
    ctx.result.inlier_ratio = ctx.geometry_data.inlier_ratio;
    return ok;
}

std::string FeaturePipeline::buildOutputStem(const RegistrationContext& ctx) const {
    // 1. 从 matcher 配置中提炼稳定标签，避免输出文件名受 YAML 别名影响。
    const std::string matcherLabel = buildMatcherLabel(_config.matcher_path);

    // 2. 组合输入图像、特征类型、几何模型和匹配器标签，形成可追溯文件名前缀。
    return ctx.image1_path.stem().string() + "_" + ctx.image2_path.stem().string() + "_" +
           (_extractor ? _extractor->name() : std::string("UNK")) + "_" +
           (_geometry ? toString(_geometry->type()) : std::string("UNK")) + "_" + matcherLabel;
}

bool FeaturePipeline::saveOutputs(RegistrationContext& ctx) {
    // 1. 没有配置输出目录时直接跳过可视化落盘。
    if (_config.output_dir.empty())
        return true;

    // 2. 准备点特征专属输出目录。
    const fs::path keypoints_dir = _config.output_dir / "keypoints";
    const fs::path matches_dir = _config.output_dir / "matches";
    std::error_code ec;
    fs::create_directories(keypoints_dir, ec);
    fs::create_directories(matches_dir, ec);

    const std::string stem = buildOutputStem(ctx);

    // 3. 按需保存关键点可视化图。
    if (_config.draw_keypoints) {
        cv::Mat src_vis;
        cv::Mat dst_vis;
        cv::drawKeypoints(ctx.feature_data.first.image,
                          ctx.feature_data.first.keypoints,
                          src_vis,
                          cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::drawKeypoints(ctx.feature_data.second.image,
                          ctx.feature_data.second.keypoints,
                          dst_vis,
                          cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        if (!src_vis.empty()) {
            const fs::path out = keypoints_dir / (stem + "_source_keypoints.png");
            if (cv::imwrite(out.string(), src_vis)) {
                IR_LOG_INFO("Wrote source keypoints visualization: ", out.string());
            } else {
                IR_LOG_WARN("Failed to write source keypoints visualization: ", out.string());
            }
        }
        if (!dst_vis.empty()) {
            const fs::path out = keypoints_dir / (stem + "_target_keypoints.png");
            if (cv::imwrite(out.string(), dst_vis)) {
                IR_LOG_INFO("Wrote target keypoints visualization: ", out.string());
            } else {
                IR_LOG_WARN("Failed to write target keypoints visualization: ", out.string());
            }
        }
    }

    // 4. 按需保存匹配关系可视化图。
    if (_config.draw_matches) {
        DrawMatches::Options opt;
        opt.draw_inliers_only = _config.draw_inliers_only;
        opt.max_matches = _config.max_matches_drawn;
        cv::Mat vis = DrawMatches::render(ctx, opt);
        if (!vis.empty()) {
            const fs::path out = matches_dir / (stem + "_matches.png");
            cv::imwrite(out.string(), vis);
            IR_LOG_INFO("Wrote matches visualization: ", out.string());
        }
    }

    // 5. 委托基类保存 warp 和 blend 等通用输出。
    return BasePipeline::saveOutputs(ctx);
}

} // namespace ir
