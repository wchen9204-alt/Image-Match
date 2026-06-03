#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace ir {

/// 点特征匹配阶段的原始、中间和最终匹配结果。
struct KeypointMatchData {
    std::vector<std::vector<cv::DMatch>> raw_knn;
    std::vector<cv::DMatch> filtered;
    std::vector<unsigned char> inlier_mask;
    std::vector<cv::DMatch> inliers;

    void clear() {
        raw_knn.clear();
        filtered.clear();
        inlier_mask.clear();
        inliers.clear();
    }
};

} // namespace ir
