#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace ir {

// 轮廓的基础几何特征缓存，供提取过滤和后续匹配阶段复用。
struct ContourFeature {
    int index = -1;
    double area = 0.0;
    double perimeter = 0.0;
    cv::Point2d centroid{0.0, 0.0};
    cv::Rect bbox;
    double bboxArea = 0.0;
    double extent = 0.0;
    double aspectRatio = 0.0;
    bool valid = false;
};

// 为单条轮廓计算基础几何特征。
ContourFeature buildContourFeature(const std::vector<cv::Point>& contour, int index = -1);

// 批量计算轮廓特征，返回顺序与输入轮廓一致。
std::vector<ContourFeature> buildContourFeatures(
    const std::vector<std::vector<cv::Point>>& contours);

} // namespace ir
