#pragma once

#include "core/types.h"

namespace ir {

/// 变换类型枚举，描述当前图像配准模块支持的几何变换。
enum class TransformType {
    UNKNOWN = 0,
    PERSPECTIVE,  // 3x3 单应矩阵。
    AFFINE        // 2x3 仿射矩阵。
};

/// 将几何模型类型映射为变换类型。
inline TransformType toTransformType(GeometryType g) {
    switch (g) {
        case GeometryType::HOMOGRAPHY: return TransformType::PERSPECTIVE;
        case GeometryType::AFFINE:
        case GeometryType::RIGID:
        case GeometryType::SIMILARITY:
            return TransformType::AFFINE;
        default:
            return TransformType::UNKNOWN;
    }
}

/// 将变换类型转换为字符串。
inline const char* toString(TransformType t) {
    switch (t) {
        case TransformType::PERSPECTIVE: return "PERSPECTIVE";
        case TransformType::AFFINE:      return "AFFINE";
        default:                         return "UNKNOWN";
    }
}

} // namespace ir
