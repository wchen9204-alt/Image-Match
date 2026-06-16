#pragma once

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "core/types.h"

namespace ir {
namespace descriptor_norm_utils {

/// ���������ṩ�� YAML ��ȡ�������ͣ�ȱʡ��AUTO ��Ƿ�ֵʹ�� fallback��
NormType readConfiguredNorm(const YAML::Node& cfg, NormType fallback);

/// ���������Ӿ��������ƶϾ������͡�
NormType inferFromDescriptors(const cv::Mat& descriptors);

/// �������վ������ͣ�ƥ������ʽ�������ȣ����ʹ���������ṩ�����ã���󰴾������Ͷ��ס�
NormType resolve(NormType configuredNorm, NormType providerNorm, const cv::Mat& descriptors);

} // namespace descriptor_norm_utils
} // namespace ir
