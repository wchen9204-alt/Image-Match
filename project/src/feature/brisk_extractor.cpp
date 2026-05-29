#include "feature/brisk_extractor.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

BriskExtractor::BriskExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    _thresh       = yaml_utils::getInt  (params, "thresh",       30);
    _octaves      = yaml_utils::getInt  (params, "octaves",      3);
    _patternScale = yaml_utils::getFloat(params, "patternScale", 1.0f);

    _impl = cv::BRISK::create(_thresh, _octaves, _patternScale);

    IR_LOG_INFO("BRISK created: thresh=", _thresh,
                ", octaves=",             _octaves,
                ", patternScale=",        _patternScale);
}

bool BriskExtractor::extract(RegistrationContext& ctx) {
    if (!_impl) {
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

    _impl->detectAndCompute(fd.first.gray,  cv::noArray(),
                            fd.first.keypoints,  fd.first.descriptors);
    _impl->detectAndCompute(fd.second.gray, cv::noArray(),
                            fd.second.keypoints, fd.second.descriptors);

    IR_LOG_INFO("BRISK extracted ", fd.first.keypoints.size(),
                " / ",             fd.second.keypoints.size(), " keypoints");
    return !fd.empty();
}

} // namespace ir

