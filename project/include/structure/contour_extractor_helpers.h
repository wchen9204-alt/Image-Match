#pragma once

#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

namespace ir::contour_extractor_helpers {

// 将配置中的轮廓检索模式映射为 OpenCV 常量。
int contourRetrievalModeFromString(const std::string& raw);

// 将配置中的轮廓近似模式映射为 OpenCV 常量。
int contourApproxModeFromString(const std::string& raw);

// 将模糊核规范化为合法奇数；小于等于 1 时视为关闭模糊。
int normalizedBlurKernel(int kernel);

// 根据灰度中位数估计一组 Canny 双阈值。
std::pair<double, double> estimateAutoCannyThresholds(const cv::Mat& gray);

// 对单张灰度图执行边缘算子、轮廓提取、过滤和响应图绘制。
bool extractContoursForImage(const cv::Mat& gray,
                             cv::Mat& edgeResponse,
                             cv::Mat& response,
                             std::vector<std::vector<cv::Point>>& contours,
                             bool& responseIsPrimary,
                             const std::string& edgeOperator,
                             bool useFindContours,
                             bool filterContours,
                             int blurKernel,
                             double gaussianSigma,
                             bool autoCanny,
                             double cannyThreshold1,
                             double cannyThreshold2,
                             int apertureSize,
                             double edgeBinaryThreshold,
                             double logSigma,
                             double logZeroCrossingThreshold,
                             int retrievalMode,
                             int approxMode,
                             double minArea,
                             double minPerimeter,
                             int minPoints,
                             int minBboxWidth,
                             int minBboxHeight,
                             double minExtent,
                             double maxAspectRatio,
                             int maxContours,
                             int contourThickness);

} // namespace ir::contour_extractor_helpers
