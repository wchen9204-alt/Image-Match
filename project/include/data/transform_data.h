#pragma once

#include <opencv2/core.hpp>

#include "core/transform_type.h"

namespace ir {

// ---------------------------------------------------------------------------
// TransformData：供图像变换模块使用的几何变换结果。
// ---------------------------------------------------------------------------
struct TransformData {
    TransformType type = TransformType::UNKNOWN;

    // 3x3 矩阵，用于透视变换或仿射矩阵扩展后的统一计算。
    cv::Mat M;

    bool valid = false;

    void clear() {
        type  = TransformType::UNKNOWN;
        M.release();
        valid = false;
    }
};

} // namespace ir
