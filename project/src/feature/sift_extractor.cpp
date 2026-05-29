#include "feature/sift_extractor.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

SiftExtractor::SiftExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    nfeatures_         = yaml_utils::getInt   (params, "nfeatures",         0);
    nOctaveLayers_     = yaml_utils::getInt   (params, "nOctaveLayers",     3);
    contrastThreshold_ = yaml_utils::getDouble(params, "contrastThreshold", 0.04);
    edgeThreshold_     = yaml_utils::getDouble(params, "edgeThreshold",     10.0);
    sigma_             = yaml_utils::getDouble(params, "sigma",             1.6);

    impl_ = cv::SIFT::create(
        nfeatures_,
        nOctaveLayers_,
        contrastThreshold_,
        edgeThreshold_,
        sigma_);

    IR_LOG_INFO("SIFT created: nfeatures=", nfeatures_,
                ", nOctaveLayers=",   nOctaveLayers_,
                ", contrast=",        contrastThreshold_,
                ", edge=",            edgeThreshold_,
                ", sigma=",           sigma_);
}

bool SiftExtractor::extract(RegistrationContext& ctx) {
    if (!impl_) {
        IR_LOG_ERROR("SIFT extractor not constructed.");
        return false;
    }

    auto& fd = ctx.feature_data;
    fd.type      = FeatureType::SIFT;
    fd.norm_type = NormType::L2;

    if (fd.first.image.empty() || fd.second.image.empty()) {
        IR_LOG_ERROR("SIFT::extract - source images are empty.");
        return false;
    }

    if (fd.first.gray.empty()) {
        cv::cvtColor(fd.first.image, fd.first.gray, cv::COLOR_BGR2GRAY);
    }
    if (fd.second.gray.empty()) {
        cv::cvtColor(fd.second.image, fd.second.gray, cv::COLOR_BGR2GRAY);
    }

    impl_->detectAndCompute(fd.first.gray,  cv::noArray(),
                            fd.first.keypoints,  fd.first.descriptors);
    impl_->detectAndCompute(fd.second.gray, cv::noArray(),
                            fd.second.keypoints, fd.second.descriptors);

    IR_LOG_INFO("SIFT extracted ", fd.first.keypoints.size(),
                " / ",            fd.second.keypoints.size(), " keypoints");
    return !fd.empty();
}

} // namespace ir
