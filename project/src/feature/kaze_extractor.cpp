#include "feature/kaze_extractor.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

KazeExtractor::KazeExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    extended_      = yaml_utils::getBool (params, "extended",      false);
    upright_       = yaml_utils::getBool (params, "upright",       false);
    threshold_     = yaml_utils::getFloat(params, "threshold",     0.001f);
    nOctaves_      = yaml_utils::getInt  (params, "nOctaves",      4);
    nOctaveLayers_ = yaml_utils::getInt  (params, "nOctaveLayers", 4);
    diffusivity_   = yaml_utils::getInt  (params, "diffusivity",
                                          static_cast<int>(cv::KAZE::DIFF_PM_G2));

    impl_ = cv::KAZE::create(
        extended_,
        upright_,
        threshold_,
        nOctaves_,
        nOctaveLayers_,
        static_cast<cv::KAZE::DiffusivityType>(diffusivity_));

    IR_LOG_INFO("KAZE created: extended=", extended_,
                ", upright=",              upright_,
                ", threshold=",            threshold_,
                ", nOctaves=",             nOctaves_,
                ", nOctaveLayers=",        nOctaveLayers_,
                ", diffusivity=",          diffusivity_);
}

bool KazeExtractor::extract(RegistrationContext& ctx) {
    if (!impl_) {
        IR_LOG_ERROR("KAZE extractor not constructed.");
        return false;
    }

    auto& fd = ctx.feature_data;
    fd.type      = FeatureType::KAZE;
    fd.norm_type = NormType::L2;

    if (fd.first.image.empty() || fd.second.image.empty()) {
        IR_LOG_ERROR("KAZE::extract - source images are empty.");
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

    IR_LOG_INFO("KAZE extracted ", fd.first.keypoints.size(),
                " / ",            fd.second.keypoints.size(), " keypoints");
    return !fd.empty();
}

} // namespace ir
