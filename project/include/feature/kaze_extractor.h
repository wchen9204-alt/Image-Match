#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_feature_extractor.h"

namespace ir {

/// KAZE 特征提取器。
class KazeExtractor : public IFeatureExtractor {
public:
    /// 根据 YAML 配置初始化 KAZE 参数。
    explicit KazeExtractor(const YAML::Node& cfg);

    std::string name() const override { return "KAZE"; }
    FeatureType type() const override { return FeatureType::KAZE; }
    NormType normType() const override { return NormType::L2; }

    /// 在上下文中提取 KAZE 关键点和描述子。
    bool extract(RegistrationContext& ctx) override;

private:
    bool _extended = false;
    bool _upright = false;
    float _threshold = 0.001f;
    int _nOctaves = 4;
    int _nOctaveLayers = 4;
    int _diffusivity = static_cast<int>(cv::KAZE::DIFF_PM_G2);

    cv::Ptr<cv::KAZE> _impl;
};

} // namespace ir
