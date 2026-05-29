#pragma once

#include "core/types.h"

namespace ir {

// ---------------------------------------------------------------------------
// TransformType：图像变换模块支持的几何变换类型。
// ---------------------------------------------------------------------------
enum class TransformType {
    UNKNOWN     = 0,
    PERSPECTIVE,   // 3x3 单应矩阵。
    AFFINE         // 2x3 仿射矩阵。
};

inline TransformType toTransformType(GeometryType g) {
    switch (g) {
        case GeometryType::HOMOGRAPHY: return TransformType::PERSPECTIVE;
        case GeometryType::AFFINE:     return TransformType::AFFINE;
        default:                       return TransformType::UNKNOWN;
    }
}

inline const char* toString(TransformType t) {
    switch (t) {
        case TransformType::PERSPECTIVE: return "PERSPECTIVE";
        case TransformType::AFFINE:      return "AFFINE";
        default:                         return "UNKNOWN";
    }
}

} // namespace ir
