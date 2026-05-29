#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "core/types.h"

namespace ir {

// ---------------------------------------------------------------------------
// 单张图像的特征数据：关键点、描述子以及匹配时使用的距离类型。
// ---------------------------------------------------------------------------
struct FeatureImageData {
    cv::Mat                       image;        // 原始图像，BGR 或灰度。
    cv::Mat                       gray;         // 用于检测的灰度图。
    std::vector<cv::KeyPoint>     keypoints;
    cv::Mat                       descriptors;  // 每行对应一个关键点，类型由提取器决定。

    void clear() {
        image.release();
        gray.release();
        keypoints.clear();
        descriptors.release();
    }

    bool empty() const {
        return keypoints.empty() || descriptors.empty();
    }
};

// ---------------------------------------------------------------------------
// FeatureData：一对待配准图像的特征数据。
// ---------------------------------------------------------------------------
struct FeatureData {
    FeatureImageData first;
    FeatureImageData second;

    FeatureType type      = FeatureType::UNKNOWN;
    NormType    norm_type = NormType::UNKNOWN;

    void clear() {
        first.clear();
        second.clear();
        type      = FeatureType::UNKNOWN;
        norm_type = NormType::UNKNOWN;
    }

    // 任意一张图缺少关键点或描述子时返回 true。
    bool empty() const {
        return first.empty() || second.empty();
    }
};

} // namespace ir
