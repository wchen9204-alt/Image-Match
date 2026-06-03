#pragma once

#include <opencv2/core.hpp>

#include <string>

#include "core/types.h"

namespace ir {

/// 几何估计阶段的输出数据。
struct GeometryData {
    /// 当前几何模型类型。
    GeometryType type = GeometryType::UNKNOWN;

    /// 当前配准流程可能产出的几何矩阵结果。
    cv::Mat H;
    cv::Mat A;

    /// 几何估计是否成功。
    bool valid = false;
    /// 内点数量。
    int num_inliers = 0;
    /// 内点比例。
    double inlier_ratio = 0.0;
    /// 几何估计阶段的详细状态消息。
    std::string message;

    /// 清空几何估计结果。
    void clear() {
        type = GeometryType::UNKNOWN;
        H.release();
        A.release();
        valid = false;
        num_inliers = 0;
        inlier_ratio = 0.0;
        message.clear();
    }
};

} // namespace ir
