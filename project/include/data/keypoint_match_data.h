#pragma once

#include <algorithm>

#include <opencv2/core.hpp>

#include "core/types.h"

#include <vector>

namespace ir {

/// 点特征匹配阶段的原始、中间和最终匹配结果。
struct KeypointMatchData {
    /// 本次匹配器实际使用的方法，供依赖候选邻居的过滤器判断适用性。
    MatchMethod match_method = MatchMethod::UNKNOWN;

    /// 每个 query 的原始最佳候选，尚未经过过滤器链。
    /// MATCH 直接写入；KNN / RADIUS 从每行邻居中选取距离最小者写入。
    std::vector<cv::DMatch> raw_matches;

    /// KNN / RADIUS 的完整邻居候选，仅供需要多邻居信息的过滤器使用。
    std::vector<std::vector<cv::DMatch>> neighbour_matches_by_query;

    /// 过滤链之后、进入几何估计之前的匹配结果。
    std::vector<cv::DMatch> filtered_matches;

    /// 几何估计阶段给出的内点掩码，与 filtered_matches 按索引一一对应。
    std::vector<unsigned char> inlier_mask;

    /// 根据几何估计最终确认并写回的内点匹配列表。
    /// 点特征法中由 estimateAffinePartial2D + 可选 rigid refine 的最终 mask 提升得到。
    std::vector<cv::DMatch> inlier_matches;

    /// 从 KNN / RADIUS 邻居中提取每个 query 距离最小的原始候选。
    void buildRawMatchesFromNeighbours() {
        raw_matches.clear();
        raw_matches.reserve(neighbour_matches_by_query.size());
        for (const auto& neighbours : neighbour_matches_by_query) {
            const auto best = std::min_element(neighbours.begin(), neighbours.end(),
                                               [](const cv::DMatch& lhs, const cv::DMatch& rhs) {
                                                   return lhs.distance < rhs.distance;
                                               });
            if (best != neighbours.end()) {
                raw_matches.push_back(*best);
            }
        }
    }

    /// 在过滤链开始前以 raw 初始化工作集合；已有过滤结果时不覆盖。
    void seedFilteredMatchesFromRaw() {
        if (filtered_matches.empty() && !raw_matches.empty()) {
            filtered_matches = raw_matches;
        }
    }

    /// 过滤链没有保留候选时，恢复到匹配器原始候选。
    void restoreFilteredMatchesFromRaw() {
        if (!raw_matches.empty()) {
            filtered_matches = raw_matches;
        }
    }

    void clear() {
        match_method = MatchMethod::UNKNOWN;
        raw_matches.clear();
        neighbour_matches_by_query.clear();
        filtered_matches.clear();
        inlier_mask.clear();
        inlier_matches.clear();
    }
};

} // namespace ir