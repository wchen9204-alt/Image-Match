#pragma once

#include <opencv2/core.hpp>

#include "core/types.h"

namespace ir {

// ---------------------------------------------------------------------------
// GeometryData：保存几何估计器输出的模型结果。
//
// 根据 type 使用对应字段：H / A / F / E；本质矩阵还会保存 R、t、K。
// ---------------------------------------------------------------------------
struct GeometryData {
    GeometryType type = GeometryType::UNKNOWN;

    cv::Mat H;   // 单应矩阵。
    cv::Mat A;   // 2x3 仿射矩阵。
    cv::Mat F;   // 基础矩阵。
    cv::Mat E;   // 本质矩阵。
    cv::Mat R;   // 旋转矩阵。
    cv::Mat t;   // 平移向量。
    cv::Mat K;   // 相机内参。

    bool   valid       = false;
    int    num_inliers = 0;
    double inlier_ratio = 0.0;

    void clear() {
        type = GeometryType::UNKNOWN;
        H.release();
        A.release();
        F.release();
        E.release();
        R.release();
        t.release();
        K.release();
        valid        = false;
        num_inliers  = 0;
        inlier_ratio = 0.0;
    }
};

} // namespace ir
