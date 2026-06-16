#pragma once

#include <opencv2/core.hpp>

namespace ir {

/// 图像处理辅助函数。
namespace image_utils {

/// 将任意类型图像转换为 `[0, 1]` 范围的单通道 `CV_32F` 图像。
cv::Mat toGrayFloat(const cv::Mat& src);

/// 小于 minSize 时返回 0，偶数自动加 1；适合“0 表示关闭”的可选高斯核。
int normalizedOddKernelOrZero(int value, int minSize = 3);

/// 小于 minimum 时回退到 fallback，偶数自动加 1；适合必须有合法核大小的参数。
int normalizedOddKernel(int value, int fallback, int minimum = 1);

/// 归一化 Canny aperture size；OpenCV 只接受 3、5、7。
int normalizedCannyAperture(int value, int fallback = 3);

/// 确保 gray 是 color 对应的单通道灰度图；若 gray 已存在则保持不变。
bool ensureGray(const cv::Mat& color, cv::Mat& gray);

/// 对图像应用可选高斯模糊；核大小小于 3 时保持输入不变。
void applyOptionalGaussianBlur(const cv::Mat& src, cv::Mat& dst, int blurKernel);

/// 将单通道灰度图转换为 `[0, 1]` 范围的 `CV_32F`，并可选预平滑。
bool convertGrayToFloat01(const cv::Mat& gray, cv::Mat& out, int blurKernel = 0);

/// 根据变换矩阵和目标尺寸生成有效像素掩码。
cv::Mat warpedValidMask(const cv::Mat& src_size, const cv::Mat& H, const cv::Size& dst_size);

/// 从变换后的图像中直接提取非零像素掩码。
cv::Mat nonZeroMask(const cv::Mat& warped);

/// 根据掩码裁剪两张图像的共同有效区域。
void cropToMask(
    const cv::Mat& a, const cv::Mat& b, const cv::Mat& mask, cv::Mat& a_out, cv::Mat& b_out);

} // namespace image_utils
} // namespace ir

