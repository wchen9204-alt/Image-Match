#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace ir {
namespace structure_points {

cv::Mat toGray32F(const cv::Mat& response);

cv::Mat toBinaryMask(const cv::Mat& response);

bool prepareDistanceMap(const cv::Mat& response, cv::Mat& dist);

std::vector<cv::Point2f> collectPoints(const cv::Mat& response, int maxPoints);

cv::Point2d centroid(const std::vector<cv::Point2f>& points);

} // namespace structure_points
} // namespace ir

