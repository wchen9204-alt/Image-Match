#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_keypoint_extractor.h"

namespace ir {

/// BRISK 特征提取器。
class BriskExtractor : public IKeypointExtractor {
public:
    /// 根据 YAML 配置初始化 BRISK 参数。
    explicit BriskExtractor(const YAML::Node& cfg);

    std::string name() const override { return "BRISK"; }
    KeypointType type() const override { return KeypointType::BRISK; }
    NormType normType() const override { return NormType::HAMMING; }

    /// 在上下文中提取 BRISK 关键点和描述子。
    bool extract(RegistrationContext& ctx) override;

private:
    int _thresh = 30;
    int _octaves = 3;
    float _patternScale = 1.0f;

    cv::Ptr<cv::BRISK> _impl;
};

} // namespace ir
