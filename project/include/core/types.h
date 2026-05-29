#pragma once

#include <string>
#include <cstdint>
#include <opencv2/core.hpp>

namespace ir {

// ---------------------------------------------------------------------------
// 特征类型，对应 feature/*.yaml 中的 type 字段。
// ---------------------------------------------------------------------------
enum class FeatureType {
    UNKNOWN = 0,
    SIFT,
    SURF,
    ORB,
    BRISK,
    KAZE,
    AKAZE
};

std::string toString(FeatureType t);
FeatureType featureTypeFromString(const std::string& s);

// ---------------------------------------------------------------------------
// 描述子匹配使用的距离类型，对应 cv::NormTypes。
// ---------------------------------------------------------------------------
enum class NormType {
    UNKNOWN = 0,
    L1,
    L2,
    HAMMING,
    HAMMING2
};

std::string toString(NormType t);
NormType normTypeFromString(const std::string& s);
int      toCvNorm(NormType t);

// ---------------------------------------------------------------------------
// 几何模型类型。
// ---------------------------------------------------------------------------
enum class GeometryType {
    UNKNOWN = 0,
    HOMOGRAPHY,
    AFFINE,
    FUNDAMENTAL,
    ESSENTIAL
};

std::string  toString(GeometryType t);
GeometryType geometryTypeFromString(const std::string& s);

// ---------------------------------------------------------------------------
// 标识配准图像对中的哪一张图。
// ---------------------------------------------------------------------------
enum class ImageIndex : std::uint8_t {
    First  = 0,
    Second = 1
};

// ---------------------------------------------------------------------------
// 将鲁棒估计方法字符串转换为 OpenCV 常量；未知时返回 -1。
// ---------------------------------------------------------------------------
int robustMethodFromString(const std::string& s);

} // namespace ir
