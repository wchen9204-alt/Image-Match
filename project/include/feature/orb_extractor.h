#pragma once

#include <yaml-cpp/yaml.h>
#include <opencv2/features2d.hpp>

#include "interfaces/i_feature_extractor.h"

namespace ir {

class OrbExtractor : public IFeatureExtractor {
public:
    explicit OrbExtractor(const YAML::Node& cfg);

    std::string name()     const override { return "ORB"; }
    FeatureType type()     const override { return FeatureType::ORB; }
    NormType    normType() const override { return NormType::HAMMING; }

    bool extract(RegistrationContext& ctx) override;

private:
    int   nfeatures_      = 2000;
    float scaleFactor_    = 1.2f;
    int   nlevels_        = 8;
    int   edgeThreshold_  = 31;
    int   firstLevel_     = 0;
    int   WTA_K_          = 2;
    int   scoreType_      = static_cast<int>(cv::ORB::HARRIS_SCORE);
    int   patchSize_      = 31;
    int   fastThreshold_  = 20;

    cv::Ptr<cv::ORB> impl_;
};

} // namespace ir
