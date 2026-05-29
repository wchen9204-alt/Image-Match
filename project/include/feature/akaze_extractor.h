#pragma once

#include <yaml-cpp/yaml.h>
#include <opencv2/features2d.hpp>

#include "interfaces/i_feature_extractor.h"

namespace ir {

class AkazeExtractor : public IFeatureExtractor {
public:
    explicit AkazeExtractor(const YAML::Node& cfg);

    std::string name()     const override { return "AKAZE"; }
    FeatureType type()     const override { return FeatureType::AKAZE; }
    // AKAZE 默认 MLDB 描述子为二进制，通常使用 HAMMING。
    NormType    normType() const override { return norm_; }

    bool extract(RegistrationContext& ctx) override;

private:
    int   descriptor_type_     = static_cast<int>(cv::AKAZE::DESCRIPTOR_MLDB);
    int   descriptor_size_     = 0;
    int   descriptor_channels_ = 3;
    float threshold_           = 0.001f;
    int   nOctaves_            = 4;
    int   nOctaveLayers_       = 4;
    int   diffusivity_         = static_cast<int>(cv::KAZE::DIFF_PM_G2);

    NormType            norm_ = NormType::HAMMING;
    cv::Ptr<cv::AKAZE>  impl_;
};

} // namespace ir
