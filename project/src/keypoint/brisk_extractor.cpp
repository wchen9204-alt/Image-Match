#include "keypoint/brisk_extractor.h"

#include "utils/descriptor_norm_utils.h"
#include "utils/image_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

BriskExtractor::BriskExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    _thresh = yaml_utils::getInt(params, "thresh", 30);
    _octaves = yaml_utils::getInt(params, "octaves", 3);
    _patternScale = yaml_utils::getFloat(params, "patternScale", 1.0f);
    _norm = descriptor_norm_utils::readConfiguredNorm(cfg, NormType::HAMMING);

    _impl = cv::BRISK::create(_thresh, _octaves, _patternScale);

    IR_LOG_INFO("BRISK created: thresh=",
                _thresh,
                ", octaves=",
                _octaves,
                ", patternScale=",
                _patternScale,
                ", norm=",
                toString(_norm));
}

bool BriskExtractor::extract(RegistrationContext& ctx) {
    if (!_impl) {
        IR_LOG_ERROR("BRISK extractor not constructed.");
        return false;
    }

    auto& fd = ctx.keypoint_data;
    auto& images = ctx.images;
    fd.type = KeypointType::BRISK;
    fd.norm_type = _norm;

    if (images.first.empty() || images.second.empty()) {
        IR_LOG_ERROR("BRISK::extract - source images are empty.");
        return false;
    }
    if (!image_utils::ensureGray(images.first, images.first_gray) ||
        !image_utils::ensureGray(images.second, images.second_gray)) {
        IR_LOG_ERROR("BRISK::extract - failed to prepare grayscale images.");
        return false;
    }

    _impl->detectAndCompute(
        images.first_gray, cv::noArray(), fd.first.keypoints, fd.first.descriptors);
    _impl->detectAndCompute(
        images.second_gray, cv::noArray(), fd.second.keypoints, fd.second.descriptors);

    IR_LOG_INFO("BRISK extracted ",
                fd.first.keypoints.size(),
                " / ",
                fd.second.keypoints.size(),
                " keypoints");
    return !fd.empty();
}

} // namespace ir

