#include "keypoint/surf_extractor.h"

#include "utils/descriptor_norm_utils.h"
#include "utils/image_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

SurfExtractor::SurfExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    _hessianThreshold = yaml_utils::getDouble(params, "hessianThreshold", 400.0);
    _nOctaves = yaml_utils::getInt(params, "nOctaves", 4);
    _nOctaveLayers = yaml_utils::getInt(params, "nOctaveLayers", 3);
    _extended = yaml_utils::getBool(params, "extended", false);
    _upright = yaml_utils::getBool(params, "upright", false);
    _norm = descriptor_norm_utils::readConfiguredNorm(cfg, NormType::L2);

    _impl = cv::xfeatures2d::SURF::create(
        _hessianThreshold, _nOctaves, _nOctaveLayers, _extended, _upright);

    IR_LOG_INFO("SURF created: hessian=",
                _hessianThreshold,
                ", nOctaves=",
                _nOctaves,
                ", nOctaveLayers=",
                _nOctaveLayers,
                ", extended=",
                _extended,
                ", upright=",
                _upright,
                ", norm=",
                toString(_norm));
}

bool SurfExtractor::extract(RegistrationContext& ctx) {
    if (!_impl) {
        IR_LOG_ERROR("SURF extractor not constructed.");
        return false;
    }

    auto& fd = ctx.keypoint_data;
    auto& images = ctx.images;
    fd.type = KeypointType::SURF;
    fd.norm_type = _norm;

    if (images.first.empty() || images.second.empty()) {
        IR_LOG_ERROR("SURF::extract - source images are empty.");
        return false;
    }
    if (!image_utils::ensureGray(images.first, images.first_gray) ||
        !image_utils::ensureGray(images.second, images.second_gray)) {
        IR_LOG_ERROR("SURF::extract - failed to prepare grayscale images.");
        return false;
    }

    _impl->detectAndCompute(
        images.first_gray, cv::noArray(), fd.first.keypoints, fd.first.descriptors);
    _impl->detectAndCompute(
        images.second_gray, cv::noArray(), fd.second.keypoints, fd.second.descriptors);

    IR_LOG_INFO("SURF extracted ",
                fd.first.keypoints.size(),
                " / ",
                fd.second.keypoints.size(),
                " keypoints");
    return !fd.empty();
}

} // namespace ir

