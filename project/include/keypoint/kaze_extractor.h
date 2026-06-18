#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_keypoint_extractor.h"
#include "keypoint/keypoint_extractor_helpers.h"

namespace ir {

/// KAZE 点特征提取器。
class KazeExtractor : public IKeypointExtractor {
public:
    /// 根据 YAML 配置初始化 KAZE 提取器。
    explicit KazeExtractor(const YAML::Node& cfg);

    std::string name() const override { return "KAZE"; }
    KeypointType type() const override { return KeypointType::KAZE; }
    NormType normType() const override { return _norm; }

    /// 执行 KAZE 关键点检测、边界角点增强和描述子计算。
    bool extract(RegistrationContext& ctx) override;

private:
    bool _extended = false;
    bool _upright = false;
    float _threshold = 0.001f;
    int _nOctaves = 4;
    int _nOctaveLayers = 4;
    int _diffusivity = static_cast<int>(cv::KAZE::DIFF_PM_G2);

    /// 前景边界角点增强配置。
    BoundaryCornerAugmentationConfig _augmentation_config;
    NormType _norm = NormType::L2;
    cv::Ptr<cv::KAZE> _impl;
};

} // namespace ir
