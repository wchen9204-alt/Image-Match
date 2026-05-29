#pragma once

#include <yaml-cpp/yaml.h>
#include <opencv2/features2d.hpp>

#include "interfaces/i_feature_extractor.h"

namespace ir {

class SiftExtractor : public IFeatureExtractor {
public:
    explicit SiftExtractor(const YAML::Node& cfg);

    std::string name()     const override { return "SIFT"; }
    FeatureType type()     const override { return FeatureType::SIFT; }
    NormType    normType() const override { return NormType::L2; }

    bool extract(RegistrationContext& ctx) override;

private:
    int    nfeatures_         = 0;
    int    nOctaveLayers_     = 3;
    double contrastThreshold_ = 0.04;
    double edgeThreshold_     = 10.0;
    double sigma_             = 1.6;

    cv::Ptr<cv::SIFT> impl_;
};

} // namespace ir
