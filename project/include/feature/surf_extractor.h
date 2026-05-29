#pragma once

#include <opencv2/xfeatures2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_feature_extractor.h"

namespace ir {

/// SURF 特征提取器。
class SurfExtractor : public IFeatureExtractor {
public:
    /// 根据 YAML 配置初始化 SURF 参数。
    explicit SurfExtractor(const YAML::Node& cfg);

    std::string name() const override { return "SURF"; }
    FeatureType type() const override { return FeatureType::SURF; }
    NormType    normType() const override { return NormType::L2; }

    /// 在上下文中提取 SURF 关键点和描述子。
    bool extract(RegistrationContext& ctx) override;

private:
    double _hessianThreshold = 400.0;
    int    _nOctaves         = 4;
    int    _nOctaveLayers    = 3;
    bool   _extended         = false;
    bool   _upright          = false;

    cv::Ptr<cv::xfeatures2d::SURF> _impl;
};

} // namespace ir

