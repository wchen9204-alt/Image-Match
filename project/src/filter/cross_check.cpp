#include "filter/cross_check.h"

#include <opencv2/features2d.hpp>
#include <unordered_map>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

// 读取点特征法专用交叉验证过滤器的启用开关。
CrossCheckFilter::CrossCheckFilter(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    _enabled = yaml_utils::getBool(params, "enabled", true);
    IR_LOG_INFO("CrossCheckFilter [keypoint-only] enabled=", _enabled);
}

bool CrossCheckFilter::apply(RegistrationContext& ctx) {
    // 结构法路径：交叉验证依赖可反向验证的描述子匹配，
    // 当前结构匹配结果没有统一的反向匹配输入，因此直接跳过。
    if (!ctx.structure_match_data.raw_matches_knn.empty() ||
        !ctx.structure_match_data.filtered_matches.empty()) {
        IR_LOG_WARN(
            "CrossCheckFilter [keypoint-only]: structure pipeline is not supported, pass-through.");
        return true;
    }

    auto& fd = ctx.keypoint_data;
    auto& md = ctx.keypoint_match_data;

    if (!_enabled) {
        IR_LOG_INFO("CrossCheckFilter [keypoint-only] disabled - pass-through.");
        return true;
    }

    if (md.filtered.empty()) {
        IR_LOG_WARN("CrossCheckFilter [keypoint-only]: no matches available to verify.");
        return false;
    }

    const std::vector<cv::DMatch> forward = md.filtered;
    if (forward.empty()) {
        IR_LOG_WARN("CrossCheckFilter [keypoint-only]: forward set is empty.");
        md.filtered.clear();
        return false;
    }

    // 步骤一：根据描述子类型选择反向最近邻匹配的距离度量。
    NormType norm = fd.norm_type;
    if (norm == NormType::UNKNOWN) {
        norm = (fd.first.descriptors.type() == CV_8U) ? NormType::HAMMING : NormType::L2;
    }

    cv::Ptr<cv::BFMatcher> rev = cv::BFMatcher::create(toCvNorm(norm), false);

    std::vector<cv::DMatch> reverse;
    rev->match(fd.second.descriptors, fd.first.descriptors, reverse);

    // 步骤二：建立反向最佳匹配表，将第二张图索引映射回第一张图索引。
    std::unordered_map<int, int> reverse_best;
    reverse_best.reserve(reverse.size());
    for (const auto& r : reverse) {
        reverse_best[r.queryIdx] = r.trainIdx;
    }

    std::vector<cv::DMatch> kept;
    kept.reserve(forward.size());
    // 步骤三：仅保留正反向都互相指回的双向一致匹配。
    for (const auto& m : forward) {
        const auto it = reverse_best.find(m.trainIdx);
        if (it != reverse_best.end() && it->second == m.queryIdx) {
            kept.push_back(m);
        }
    }

    IR_LOG_INFO("CrossCheckFilter [keypoint-only] kept ",
                kept.size(),
                " / ",
                forward.size(),
                " matches");
    // 步骤四：用通过交叉验证的结果覆盖当前筛选结果集合。
    md.filtered = std::move(kept);
    return true;
}

}
