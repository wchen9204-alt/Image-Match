#pragma once

#include <opencv2/core.hpp>

namespace ir {

/// 一对待配准图像的公共输入数据。
struct ImagePairData {
    /// 输入图像 1 的原图。
    cv::Mat first;

    /// 输入图像 2 的原图。
    cv::Mat second;

    /// 输入图像 1 对应的灰度图。
    cv::Mat first_gray;

    /// 输入图像 2 对应的灰度图。
    cv::Mat second_gray;

    /// 清空两张图像的原图与灰度图数据。
    void clear() {
        first.release();
        second.release();
        first_gray.release();
        second_gray.release();
    }
};

} // namespace ir
