#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "core/types.h"

namespace ir {

/// 单张图像对应的特征数据。
struct FeatureImageData {
    /// 原始图像。
    cv::Mat image;
    /// 预处理后的灰度图。
    cv::Mat gray;
    /// 关键点列表。
    std::vector<cv::KeyPoint> keypoints;
    /// 描述子矩阵，每一行对应一个关键点。
    cv::Mat descriptors;

    /// 清空当前图像的特征数据。
    void clear() {
        image.release();
        gray.release();
        keypoints.clear();
        descriptors.release();
    }

    /// 判断当前图像是否没有有效特征。
    bool empty() const {
        return keypoints.empty() || descriptors.empty();
    }
};

/// 一对待配准图像的特征数据。
struct FeatureData {
    FeatureImageData first;
    FeatureImageData second;

    /// 本次特征提取使用的特征类型。
    FeatureType type = FeatureType::UNKNOWN;
    /// 本次特征提取使用的描述子距离类型。
    NormType    norm_type = NormType::UNKNOWN;

    /// 清空两张图像的特征数据和类型信息。
    void clear() {
        first.clear();
        second.clear();
        type      = FeatureType::UNKNOWN;
        norm_type = NormType::UNKNOWN;
    }

    /// 判断是否至少有一张图像没有有效特征。
    bool empty() const {
        return first.empty() || second.empty();
    }
};

} // namespace ir
