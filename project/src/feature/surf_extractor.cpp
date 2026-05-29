#include "feature/surf_extractor.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

SurfExtractor::SurfExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    hessianThreshold_ = yaml_utils::getDouble(params, "hessianThreshold", 400.0);
    nOctaves_         = yaml_utils::getInt   (params, "nOctaves",         4);
    nOctaveLayers_    = yaml_utils::getInt   (params, "nOctaveLayers",    3);
    extended_         = yaml_utils::getBool  (params, "extended",         false);
    upright_          = yaml_utils::getBool  (params, "upright",          false);

    impl_ = cv::xfeatures2d::SURF::create(
        hessianThreshold_,
        nOctaves_,
        nOctaveLayers_,
        extended_,
        upright_);

    IR_LOG_INFO("SURF created: hessian=", hessianThreshold_,
                ", nOctaves=",            nOctaves_,
                ", nOctaveLayers=",       nOctaveLayers_,
                ", extended=",            extended_,
                ", upright=",             upright_);
}

bool SurfExtractor::extract(RegistrationContext& ctx) {
    if (!impl_) {
        IR_LOG_ERROR("SURF extractor not constructed.");
        return false;
    }

    auto& fd = ctx.feature_data;
    fd.type      = FeatureType::SURF;
    fd.norm_type = NormType::L2;

    if (fd.first.image.empty() || fd.second.image.empty()) {
        IR_LOG_ERROR("SURF::extract - source images are empty.");
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

    IR_LOG_INFO("SURF extracted ", fd.first.keypoints.size(),
                " / ",            fd.second.keypoints.size(), " keypoints");
    return !fd.empty();
}

} // namespace ir
