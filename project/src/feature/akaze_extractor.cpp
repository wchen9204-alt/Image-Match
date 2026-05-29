#include "feature/akaze_extractor.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

AkazeExtractor::AkazeExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    descriptor_type_     = yaml_utils::getInt  (params, "descriptor_type",
                                                static_cast<int>(cv::AKAZE::DESCRIPTOR_MLDB));
    descriptor_size_     = yaml_utils::getInt  (params, "descriptor_size",     0);
    descriptor_channels_ = yaml_utils::getInt  (params, "descriptor_channels", 3);
    threshold_           = yaml_utils::getFloat(params, "threshold",           0.001f);
    nOctaves_            = yaml_utils::getInt  (params, "nOctaves",            4);
    nOctaveLayers_       = yaml_utils::getInt  (params, "nOctaveLayers",       4);
    diffusivity_         = yaml_utils::getInt  (params, "diffusivity",
                                                static_cast<int>(cv::KAZE::DIFF_PM_G2));

    // KAZE 系描述子为浮点型，MLDB 系描述子为二进制。
    const auto dtype = static_cast<cv::AKAZE::DescriptorType>(descriptor_type_);
    if (dtype == cv::AKAZE::DESCRIPTOR_KAZE ||
        dtype == cv::AKAZE::DESCRIPTOR_KAZE_UPRIGHT) {
        norm_ = NormType::L2;
    } else {
        norm_ = NormType::HAMMING;
    }

    impl_ = cv::AKAZE::create(
        dtype,
        descriptor_size_,
        descriptor_channels_,
        threshold_,
        nOctaves_,
        nOctaveLayers_,
        static_cast<cv::KAZE::DiffusivityType>(diffusivity_));

    IR_LOG_INFO("AKAZE created: descriptor_type=", descriptor_type_,
                ", size=",                         descriptor_size_,
                ", channels=",                     descriptor_channels_,
                ", threshold=",                    threshold_,
                ", nOctaves=",                     nOctaves_,
                ", nOctaveLayers=",                nOctaveLayers_,
                ", diffusivity=",                  diffusivity_,
                ", norm=",                         toString(norm_));
}

bool AkazeExtractor::extract(RegistrationContext& ctx) {
    if (!impl_) {
        IR_LOG_ERROR("AKAZE extractor not constructed.");
        return false;
    }

    auto& fd = ctx.feature_data;
    fd.type      = FeatureType::AKAZE;
    fd.norm_type = norm_;

    if (fd.first.image.empty() || fd.second.image.empty()) {
        IR_LOG_ERROR("AKAZE::extract - source images are empty.");
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

    IR_LOG_INFO("AKAZE extracted ", fd.first.keypoints.size(),
                " / ",             fd.second.keypoints.size(), " keypoints");
    return !fd.empty();
}

} // namespace ir
