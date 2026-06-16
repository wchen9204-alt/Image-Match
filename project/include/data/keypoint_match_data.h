#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace ir {

/// 点特征匹配阶段的原始、中间和最终匹配结果。
struct KeypointMatchData {
    /// 匹配器原始输出的候选匹配，按 query 行分组保存。
    /// MATCH 模式下每行通常只有 1 个候选；KNN / RADIUS 模式下可保留多个候选。
    std::vector<std::vector<cv::DMatch>> raw_matches_by_query;

    /// 过滤链之后、进入几何估计之前的匹配结果。
    /// 若匹配器没有直接给出一对一结果，pipeline 会先从 raw_matches_by_query 的每行 top-1 生成它。
    std::vector<cv::DMatch> filtered_matches;

    /// 几何估计阶段给出的内点掩码，与 filtered_matches 按索引一一对应。
    std::vector<unsigned char> inlier_mask;

    /// 根据几何估计最终确认并写回的内点匹配列表。
    /// 点特征法中由 estimateAffinePartial2D + 可选 rigid refine 的最终 mask 提升得到。
    std::vector<cv::DMatch> inlier_matches;

    void clear() {
        raw_matches_by_query.clear();
        filtered_matches.clear();
        inlier_mask.clear();
        inlier_matches.clear();
    }
};

} // namespace ir

