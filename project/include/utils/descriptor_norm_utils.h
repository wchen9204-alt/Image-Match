#pragma once

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "core/types.h"

namespace ir {
namespace descriptor_norm_utils {

/// 从提供者给出的 YAML 读取范数类型；缺省为 AUTO，缺值时使用 fallback。
NormType readConfiguredNorm(const YAML::Node& cfg, NormType fallback);

/// 根据当前描述子内容推断范数类型。
NormType inferFromDescriptors(const cv::Mat& descriptors);

/// 综合配置范数、提供者范数和描述子内容，得到最终使用的范数类型。
NormType resolve(NormType configuredNorm, NormType providerNorm, const cv::Mat& descriptors);

} // namespace descriptor_norm_utils
} // namespace ir
