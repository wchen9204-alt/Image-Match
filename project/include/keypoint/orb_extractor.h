#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_keypoint_extractor.h"
#include "keypoint/keypoint_extractor_helpers.h"

namespace ir {

/// ORB 点特征提取器。
class OrbExtractor : public IKeypointExtractor {
public:
    /// 根据 YAML 配置初始化 ORB 提取器。
    explicit OrbExtractor(const YAML::Node& cfg);

    std::string name() const override { return "ORB"; }
    KeypointType type() const override { return KeypointType::ORB; }
    NormType normType() const override { return _norm; }

    /// 执行 ORB 关键点检测、边界角点增强和描述子计算。
    bool extract(RegistrationContext& ctx) override;

private:
    int _nfeatures = 2000;
    float _scaleFactor = 1.2f;
    int _nlevels = 8;
    int _edgeThreshold = 31;
    int _firstLevel = 0;
    int _wtaK = 2;
    int _scoreType = static_cast<int>(cv::ORB::HARRIS_SCORE);
    int _patchSize = 31;
    int _fastThreshold = 20;

    /// 前景边界角点增强配置。
    BoundaryCornerAugmentationConfig _augmentation_config;
    NormType _norm = NormType::HAMMING;
    cv::Ptr<cv::ORB> _impl;
};

} // namespace ir
