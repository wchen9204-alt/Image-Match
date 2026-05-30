#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_feature_extractor.h"

namespace ir {

/// AKAZE 特征提取器。
class AkazeExtractor : public IFeatureExtractor {
public:
    /// 根据 YAML 配置初始化 AKAZE 参数。
    explicit AkazeExtractor(const YAML::Node& cfg);

    std::string name() const override { return "AKAZE"; }
    FeatureType type() const override { return FeatureType::AKAZE; }
    /// AKAZE 默认使用二进制描述子，通常对应 Hamming 距离。
    NormType normType() const override { return _norm; }

    /// 在上下文中提取 AKAZE 关键点和描述子。
    bool extract(RegistrationContext& ctx) override;

private:
    int _descriptorType = static_cast<int>(cv::AKAZE::DESCRIPTOR_MLDB);
    int _descriptorSize = 0;
    int _descriptorChannels = 3;
    float _threshold = 0.001f;
    int _nOctaves = 4;
    int _nOctaveLayers = 4;
    int _diffusivity = static_cast<int>(cv::KAZE::DIFF_PM_G2);

    NormType _norm = NormType::HAMMING;
    cv::Ptr<cv::AKAZE> _impl;
};

} // namespace ir
