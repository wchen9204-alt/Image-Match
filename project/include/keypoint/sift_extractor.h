#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_keypoint_extractor.h"

namespace ir {

/// SIFT 特征提取器。
class SiftExtractor : public IKeypointExtractor {
public:
    /// 根据 YAML 配置初始化 SIFT 参数。
    explicit SiftExtractor(const YAML::Node& cfg);

    std::string name() const override { return "SIFT"; }
    KeypointType type() const override { return KeypointType::SIFT; }
    NormType normType() const override { return NormType::L2; }

    /// 在上下文中提取 SIFT 关键点和描述子。
    bool extract(RegistrationContext& ctx) override;

private:
    int _nfeatures = 0;
    int _nOctaveLayers = 3;
    double _contrastThreshold = 0.04;
    double _edgeThreshold = 10.0;
    double _sigma = 1.6;

    cv::Ptr<cv::SIFT> _impl;
};

} // namespace ir
