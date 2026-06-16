#pragma once

#include <opencv2/core.hpp>

#include "core/types.h"

namespace ir {

/// 图像变换阶段的结果数据。
struct TransformData {
    /// 当前变换类型。
    TransformType type = TransformType::UNKNOWN;

    /// 统一使用的 3x3 变换矩阵。
    /// 约定：
    /// - PERSPECTIVE: 标准 3x3 单应矩阵
    /// - AFFINE: 扩展后的齐次 3x3 矩阵，最后一行为 [0, 0, 1]
    cv::Mat M;

    /// 变换是否有效。
    bool valid = false;

    /// 清空变换结果。
    void clear() {
        type = TransformType::UNKNOWN;
        M.release();
        valid = false;
    }
};

} // namespace ir

