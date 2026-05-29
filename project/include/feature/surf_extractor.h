#pragma once

#include <yaml-cpp/yaml.h>
#include <opencv2/xfeatures2d.hpp>

#include "interfaces/i_feature_extractor.h"

namespace ir {

class SurfExtractor : public IFeatureExtractor {
public:
    explicit SurfExtractor(const YAML::Node& cfg);

    std::string name()     const override { return "SURF"; }
    FeatureType type()     const override { return FeatureType::SURF; }
    NormType    normType() const override { return NormType::L2; }

    bool extract(RegistrationContext& ctx) override;

private:
    double hessianThreshold_ = 400.0;
    int    nOctaves_         = 4;
    int    nOctaveLayers_    = 3;
    bool   extended_         = false;
    bool   upright_          = false;

    cv::Ptr<cv::xfeatures2d::SURF> impl_;
};

} // namespace ir
