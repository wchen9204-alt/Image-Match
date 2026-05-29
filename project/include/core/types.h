#pragma once

#include <cstdint>
#include <opencv2/core.hpp>
#include <string>

namespace ir {

/// 特征类型枚举，对应 `feature/*.yaml` 中的 `type` 字段。
enum class FeatureType {
    UNKNOWN = 0,
    SIFT,
    SURF,
    ORB,
    BRISK,
    KAZE,
    AKAZE
};

/// 将特征类型转换为字符串，便于配置保存和日志输出。
std::string toString(FeatureType t);

/// 将字符串解析为特征类型，无法识别时返回 `UNKNOWN`。
FeatureType featureTypeFromString(const std::string& s);

/// 描述描述子匹配时使用的距离类型，对应 OpenCV 的 `cv::NormTypes`。
enum class NormType {
    UNKNOWN = 0,
    L1,
    L2,
    HAMMING,
    HAMMING2
};

/// 将距离类型转换为字符串表示。
std::string toString(NormType t);

/// 将字符串解析为距离类型，无法识别时返回 `UNKNOWN`。
NormType normTypeFromString(const std::string& s);

/// 将项目内部的距离类型转换为 OpenCV 的 norm 常量。
int toCvNorm(NormType t);

/// 几何模型类型，用于描述估计得到的是哪一种变换关系。
enum class GeometryType {
    UNKNOWN = 0,
    HOMOGRAPHY,
    AFFINE,
    RIGID,
    SIMILARITY
};

/// 将几何模型类型转换为字符串。
std::string toString(GeometryType t);

/// 将字符串解析为几何模型类型，无法识别时返回 `UNKNOWN`。
GeometryType geometryTypeFromString(const std::string& s);

/// 输入图像索引，表示当前配准过程中的第一张或第二张图像。
enum class ImageIndex : std::uint8_t {
    First  = 0,
    Second = 1
};

/// 将鲁棒估计方法字符串转换为 OpenCV 常量，未知时返回 `-1`。
int robustMethodFromString(const std::string& s);

} // namespace ir
