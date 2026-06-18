#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_keypoint_extractor.h"
#include "keypoint/keypoint_extractor_helpers.h"

namespace ir {

/// SIFT 点特征提取器。
class SiftExtractor : public IKeypointExtractor {
public:
    /// 根据 YAML 配置初始化 SIFT 提取器。
    explicit SiftExtractor(const YAML::Node& cfg);

    std::string name() const override { return "SIFT"; }
    KeypointType type() const override { return KeypointType::SIFT; }
    NormType normType() const override { return _norm; }

    /// 执行 SIFT 关键点检测、边界角点增强和描述子计算。
    bool extract(RegistrationContext& ctx) override;

private:
    int _nfeatures = 0;
    int _nOctaveLayers = 3;
    double _contrastThreshold = 0.04;
    double _edgeThreshold = 10.0;
    double _sigma = 1.6;

    /// 前景边界角点增强配置。
    BoundaryCornerAugmentationConfig _augmentation_config;
    NormType _norm = NormType::L2;
    cv::Ptr<cv::SIFT> _impl;
};

} // namespace ir
