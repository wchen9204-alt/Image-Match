#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_keypoint_extractor.h"
#include "keypoint/keypoint_extractor_helpers.h"

namespace ir {

/// BRISK 点特征提取器。
class BriskExtractor : public IKeypointExtractor {
public:
    /// 根据 YAML 配置初始化 BRISK 提取器。
    explicit BriskExtractor(const YAML::Node& cfg);

    std::string name() const override { return "BRISK"; }
    KeypointType type() const override { return KeypointType::BRISK; }
    NormType normType() const override { return _norm; }

    /// 执行 BRISK 关键点检测、边界角点增强和描述子计算。
    bool extract(RegistrationContext& ctx) override;

private:
    int _thresh = 30;
    int _octaves = 3;
    float _patternScale = 1.0f;

    /// 前景边界角点增强配置。
    BoundaryCornerAugmentationConfig _augmentation_config;
    NormType _norm = NormType::HAMMING;
    cv::Ptr<cv::BRISK> _impl;
};

} // namespace ir
