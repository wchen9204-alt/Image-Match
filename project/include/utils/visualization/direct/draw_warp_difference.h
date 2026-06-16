#pragma once

#include <opencv2/core.hpp>

namespace ir {

/// 将 warped 与 target 的绝对差渲染为伪彩色热力图，用于直接法误差诊断。
cv::Mat renderWarpDifference(const cv::Mat& warped, const cv::Mat& target);

} // namespace ir

