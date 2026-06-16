#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <vector>

#include "core/types.h"

namespace ir {

/// 单张图像对应的点特征数据。
struct KeypointImageData {
    /// 关键点列表。
    std::vector<cv::KeyPoint> keypoints;
    /// 描述子矩阵，每一行对应一个关键点。
    cv::Mat descriptors;

    void clear() {
        keypoints.clear();
        descriptors.release();
    }

    bool empty() const { return keypoints.empty() || descriptors.empty(); }
};

/// 一对待配准图像的点特征数据。
struct KeypointData {
    KeypointImageData first;
    KeypointImageData second;

    KeypointType type = KeypointType::UNKNOWN;
    NormType norm_type = NormType::UNKNOWN;

    void clear() {
        first.clear();
        second.clear();
        type = KeypointType::UNKNOWN;
        norm_type = NormType::UNKNOWN;
    }

    bool empty() const { return first.empty() || second.empty(); }
};

} // namespace ir

