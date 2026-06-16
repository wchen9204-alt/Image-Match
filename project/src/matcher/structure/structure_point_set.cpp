#include "matcher/structure/structure_point_set.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace ir {
namespace structure_points {

// 将结构响应图统一成单通道 CV_32F，供点集型匹配方法按非零响应采样。
cv::Mat toGray32F(const cv::Mat& response) {
    cv::Mat gray;
    if (response.channels() == 1) {
        gray = response;
    } else {
        cv::cvtColor(response, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat out;
    if (gray.depth() == CV_32F) {
        out = gray;
    } else {
        gray.convertTo(out, CV_32F, 1.0 / 255.0);
    }
    return out;
}

// 将结构响应图统一成 8 位二值掩膜，供距离变换和可视化复用。
cv::Mat toBinaryMask(const cv::Mat& response) {
    cv::Mat gray;
    if (response.channels() == 1) {
        gray = response;
    } else {
        cv::cvtColor(response, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat binary;
    cv::threshold(gray, binary, 0.0, 255.0, cv::THRESH_BINARY);
    if (binary.depth() != CV_8U) {
        binary.convertTo(binary, CV_8U);
    }
    return binary;
}

// 生成“到最近结构点距离”的距离图，Chamfer / Hausdorff 共用该表示评分。
bool prepareDistanceMap(const cv::Mat& response, cv::Mat& dist) {
    const cv::Mat binary = toBinaryMask(response);
    if (binary.empty() || cv::countNonZero(binary) == 0) {
        return false;
    }

    // distanceTransform 计算的是非零像素到零像素的距离，因此先反转结构掩膜。
    cv::Mat inverse;
    cv::bitwise_not(binary, inverse);
    cv::distanceTransform(inverse, dist, cv::DIST_L2, 3);
    return true;
}

// 从响应图中采样结构点；先收集全图非零点，再均匀下采样，避免点集中在图像上方。
std::vector<cv::Point2f> collectPoints(const cv::Mat& response, int maxPoints) {
    const cv::Mat gray = toGray32F(response);
    std::vector<cv::Point2f> allPoints;
    if (gray.empty() || maxPoints <= 0) {
        return allPoints;
    }

    allPoints.reserve(gray.rows * gray.cols);
    for (int y = 0; y < gray.rows; ++y) {
        const float* row = gray.ptr<float>(y);
        for (int x = 0; x < gray.cols; ++x) {
            if (row[x] > 0.0f) {
                allPoints.emplace_back(static_cast<float>(x), static_cast<float>(y));
            }
        }
    }

    if (allPoints.size() <= static_cast<size_t>(maxPoints)) {
        return allPoints;
    }

    // 在全响应范围内等间隔抽样，保持确定性，同时覆盖图像上下左右区域。
    std::vector<cv::Point2f> sampled;
    sampled.reserve(maxPoints);
    if (maxPoints == 1) {
        sampled.push_back(allPoints.front());
        return sampled;
    }

    const double step =
        static_cast<double>(allPoints.size() - 1) / static_cast<double>(maxPoints - 1);
    for (int i = 0; i < maxPoints; ++i) {
        const size_t idx = std::min(allPoints.size() - 1, static_cast<size_t>(std::round(i * step)));
        sampled.push_back(allPoints[idx]);
    }
    return sampled;
}

// 计算点集质心，用作平移 ICP 的默认初始化。
cv::Point2d centroid(const std::vector<cv::Point2f>& points) {
    if (points.empty()) {
        return cv::Point2d(0.0, 0.0);
    }

    cv::Point2d sum(0.0, 0.0);
    for (const auto& p : points) {
        sum.x += p.x;
        sum.y += p.y;
    }
    const double inv = 1.0 / static_cast<double>(points.size());
    return cv::Point2d(sum.x * inv, sum.y * inv);
}

} // namespace structure_points
} // namespace ir

