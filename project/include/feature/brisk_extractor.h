#pragma once

#include <yaml-cpp/yaml.h>
#include <opencv2/features2d.hpp>

#include "interfaces/i_feature_extractor.h"

namespace ir {

class BriskExtractor : public IFeatureExtractor {
public:
    explicit BriskExtractor(const YAML::Node& cfg);

    std::string name()     const override { return "BRISK"; }
    FeatureType type()     const override { return FeatureType::BRISK; }
    NormType    normType() const override { return NormType::HAMMING; }

    bool extract(RegistrationContext& ctx) override;

private:
    int   thresh_       = 30;
    int   octaves_      = 3;
    float patternScale_ = 1.0f;

    cv::Ptr<cv::BRISK> impl_;
};

} // namespace ir
