#pragma once

#include <opencv2/xfeatures2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_keypoint_extractor.h"
#include "keypoint/keypoint_extractor_helpers.h"

namespace ir {

/// SURF 点特征提取器。
class SurfExtractor : public IKeypointExtractor {
public:
    /// 根据 YAML 配置初始化 SURF 提取器。
    explicit SurfExtractor(const YAML::Node& cfg);

    std::string name() const override { return "SURF"; }
    KeypointType type() const override { return KeypointType::SURF; }
    NormType normType() const override { return _norm; }

    /// 执行 SURF 关键点检测、边界角点增强和描述子计算。
    bool extract(RegistrationContext& ctx) override;

private:
    double _hessianThreshold = 400.0;
    int _nOctaves = 4;
    int _nOctaveLayers = 3;
    bool _extended = false;
    bool _upright = false;

    /// 前景边界角点增强配置。
    BoundaryCornerAugmentationConfig _augmentation_config;
    NormType _norm = NormType::L2;
    cv::Ptr<cv::xfeatures2d::SURF> _impl;
};

} // namespace ir
