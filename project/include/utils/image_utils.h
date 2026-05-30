#pragma once

#include <opencv2/core.hpp>

namespace ir {

/// 图像处理辅助函数。
namespace image_utils {

/// 将任意类型图像转换为 `[0, 1]` 范围的单通道 `CV_32F` 图像。
cv::Mat toGrayFloat(const cv::Mat& src);

/// 根据变换矩阵和目标尺寸生成有效像素掩码。
cv::Mat warpedValidMask(const cv::Mat& src_size, const cv::Mat& H, const cv::Size& dst_size);

/// 从变换后的图像中直接提取非零像素掩码。
cv::Mat nonZeroMask(const cv::Mat& warped);

/// 根据掩码裁剪两张图像的共同有效区域。
void cropToMask(
    const cv::Mat& a, const cv::Mat& b, const cv::Mat& mask, cv::Mat& a_out, cv::Mat& b_out);

} // namespace image_utils
} // namespace ir
