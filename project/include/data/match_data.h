#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace ir {

// ---------------------------------------------------------------------------
// MatchData：保存原始匹配、过滤后匹配以及几何估计得到的内点信息。
// ---------------------------------------------------------------------------
struct MatchData {
    std::vector<std::vector<cv::DMatch>> raw_knn;
    std::vector<cv::DMatch>              filtered;
    std::vector<unsigned char>           inlier_mask;
    std::vector<cv::DMatch>              inliers;

    void clear() {
        raw_knn.clear();
        filtered.clear();
        inlier_mask.clear();
        inliers.clear();
    }
};

} // namespace ir
