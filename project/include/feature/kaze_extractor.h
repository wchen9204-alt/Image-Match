#pragma once

#include <yaml-cpp/yaml.h>
#include <opencv2/features2d.hpp>

#include "interfaces/i_feature_extractor.h"

namespace ir {

class KazeExtractor : public IFeatureExtractor {
public:
    explicit KazeExtractor(const YAML::Node& cfg);

    std::string name()     const override { return "KAZE"; }
    FeatureType type()     const override { return FeatureType::KAZE; }
    NormType    normType() const override { return NormType::L2; }

    bool extract(RegistrationContext& ctx) override;

private:
    bool  extended_     = false;
    bool  upright_      = false;
    float threshold_    = 0.001f;
    int   nOctaves_     = 4;
    int   nOctaveLayers_= 4;
    int   diffusivity_  = static_cast<int>(cv::KAZE::DIFF_PM_G2);

    cv::Ptr<cv::KAZE> impl_;
};

} // namespace ir
