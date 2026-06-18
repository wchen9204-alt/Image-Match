#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_keypoint_extractor.h"
#include "keypoint/keypoint_extractor_helpers.h"

namespace ir {

/// AKAZE 点特征提取器。
class AkazeExtractor : public IKeypointExtractor {
public:
    /// 根据 YAML 配置初始化 AKAZE 提取器。
    explicit AkazeExtractor(const YAML::Node& cfg);

    std::string name() const override { return "AKAZE"; }
    KeypointType type() const override { return KeypointType::AKAZE; }
    /// AKAZE 可能输出浮点或二值描述子，默认距离类型由配置决定。
    NormType normType() const override { return _norm; }

    /// 执行 AKAZE 关键点检测、边界角点增强和描述子计算。
    bool extract(RegistrationContext& ctx) override;

private:
    int _descriptorType = static_cast<int>(cv::AKAZE::DESCRIPTOR_MLDB);
    int _descriptorSize = 0;
    int _descriptorChannels = 3;
    float _threshold = 0.001f;
    int _nOctaves = 4;
    int _nOctaveLayers = 4;
    int _diffusivity = static_cast<int>(cv::KAZE::DIFF_PM_G2);

    /// 前景边界角点增强配置。
    BoundaryCornerAugmentationConfig _augmentation_config;
    NormType _norm = NormType::HAMMING;
    cv::Ptr<cv::AKAZE> _impl;
};

} // namespace ir
