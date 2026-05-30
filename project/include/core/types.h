#pragma once

#include <cstdint>
#include <string>

#include <opencv2/core.hpp>

namespace ir {

/// 点特征提取器类型，对应 `feature/*.yaml` 中的 `type` 配置。
enum class FeatureType { UNKNOWN = 0, SIFT, SURF, ORB, BRISK, KAZE, AKAZE };

/// 将点特征类型转换为字符串。
std::string toString(FeatureType t);

/// 从字符串解析点特征类型。
FeatureType featureTypeFromString(const std::string& s);

/// 结构特征类型，对应 `structure/*.yaml` 中的 `type` 配置。
enum class StructureType { UNKNOWN = 0, EDGE, LINE, CONTOUR };

/// 将结构特征类型转换为字符串。
std::string toString(StructureType t);

/// 从字符串解析结构特征类型。
StructureType structureTypeFromString(const std::string& s);

/// 描述子距离类型，对应 OpenCV 的 norm 常量。
enum class NormType { UNKNOWN = 0, L1, L2, HAMMING, HAMMING2 };

/// 将距离类型转换为字符串。
std::string toString(NormType t);

/// 从字符串解析距离类型。
NormType normTypeFromString(const std::string& s);

/// 将项目内部距离类型转换为 OpenCV norm 常量。
int toCvNorm(NormType t);

/// 匹配接口类型，可由 YAML 配置选择。
enum class MatchMethod { UNKNOWN = 0, MATCH, KNN, RADIUS };

/// 将匹配接口类型转换为字符串。
std::string toString(MatchMethod t);

/// 从字符串解析匹配接口类型。
MatchMethod matchMethodFromString(const std::string& s);

/// 几何模型类型。
enum class GeometryType { UNKNOWN = 0, HOMOGRAPHY, AFFINE, RIGID, SIMILARITY };

/// 将几何模型类型转换为字符串。
std::string toString(GeometryType t);

/// 从字符串解析几何模型类型。
GeometryType geometryTypeFromString(const std::string& s);

/// 输入图像索引，用于标识当前是第一张还是第二张图像。
enum class ImageIndex : std::uint8_t { First = 0, Second = 1 };

/// 将鲁棒估计方法名称转换为 OpenCV 常量；未知时返回 `-1`。
int robustMethodFromString(const std::string& s);

} // namespace ir
