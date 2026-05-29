#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace ir {

/// 匹配阶段的原始、中间和最终匹配结果。
struct MatchData {
    /// KNN 原始匹配结果。
    std::vector<std::vector<cv::DMatch>> raw_knn;
    /// 过滤后的匹配结果。
    std::vector<cv::DMatch> filtered;
    /// 内点掩码。
    std::vector<unsigned char> inlier_mask;
    /// 最终内点匹配。
    std::vector<cv::DMatch> inliers;

    /// 清空所有匹配数据。
    void clear() {
        raw_knn.clear();
        filtered.clear();
        inlier_mask.clear();
        inliers.clear();
    }
};

} // namespace ir
