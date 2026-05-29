#include "feature/brisk_extractor.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

BriskExtractor::BriskExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    thresh_       = yaml_utils::getInt  (params, "thresh",       30);
    octaves_      = yaml_utils::getInt  (params, "octaves",      3);
    patternScale_ = yaml_utils::getFloat(params, "patternScale", 1.0f);

    impl_ = cv::BRISK::create(thresh_, octaves_, patternScale_);

    IR_LOG_INFO("BRISK created: thresh=", thresh_,
                ", octaves=",             octaves_,
                ", patternScale=",        patternScale_);
}

bool BriskExtractor::extract(RegistrationContext& ctx) {
    if (!impl_) {
        IR_LOG_ERROR("BRISK extractor not constructed.");
        return false;
    }

    auto& fd = ctx.feature_data;
    fd.type      = FeatureType::BRISK;
    fd.norm_type = NormType::HAMMING;

    if (fd.first.image.empty() || fd.second.image.empty()) {
        IR_LOG_ERROR("BRISK::extract - source images are empty.");
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

    IR_LOG_INFO("BRISK extracted ", fd.first.keypoints.size(),
                " / ",             fd.second.keypoints.size(), " keypoints");
    return !fd.empty();
}

} // namespace ir
