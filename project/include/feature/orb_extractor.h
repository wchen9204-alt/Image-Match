#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_feature_extractor.h"

namespace ir {

/// ORB 特征提取器。
class OrbExtractor : public IFeatureExtractor {
public:
    /// 根据 YAML 配置初始化 ORB 参数。
    explicit OrbExtractor(const YAML::Node& cfg);

    std::string name() const override { return "ORB"; }
    FeatureType type() const override { return FeatureType::ORB; }
    NormType    normType() const override { return NormType::HAMMING; }

    /// 在上下文中提取 ORB 关键点和描述子。
    bool extract(RegistrationContext& ctx) override;

private:
    int   _nfeatures     = 2000;
    float _scaleFactor   = 1.2f;
    int   _nlevels       = 8;
    int   _edgeThreshold = 31;
    int   _firstLevel    = 0;
    int   WTA_K_         = 2;
    int   _scoreType     = static_cast<int>(cv::ORB::HARRIS_SCORE);
    int   _patchSize     = 31;
    int   _fastThreshold = 20;

    cv::Ptr<cv::ORB> _impl;
};

} // namespace ir

