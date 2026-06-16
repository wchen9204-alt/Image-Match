#include "keypoint/kaze_extractor.h"

#include "utils/descriptor_norm_utils.h"
#include "utils/image_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

KazeExtractor::KazeExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    _extended = yaml_utils::getBool(params, "extended", false);
    _upright = yaml_utils::getBool(params, "upright", false);
    _threshold = yaml_utils::getFloat(params, "threshold", 0.001f);
    _nOctaves = yaml_utils::getInt(params, "nOctaves", 4);
    _nOctaveLayers = yaml_utils::getInt(params, "nOctaveLayers", 4);
    _diffusivity =
        yaml_utils::getInt(params, "diffusivity", static_cast<int>(cv::KAZE::DIFF_PM_G2));
    _norm = descriptor_norm_utils::readConfiguredNorm(cfg, NormType::L2);

    _impl = cv::KAZE::create(_extended,
                             _upright,
                             _threshold,
                             _nOctaves,
                             _nOctaveLayers,
                             static_cast<cv::KAZE::DiffusivityType>(_diffusivity));

    IR_LOG_INFO("KAZE created: extended=",
                _extended,
                ", upright=",
                _upright,
                ", threshold=",
                _threshold,
                ", nOctaves=",
                _nOctaves,
                ", nOctaveLayers=",
                _nOctaveLayers,
                ", diffusivity=",
                _diffusivity,
                ", norm=",
                toString(_norm));
}

bool KazeExtractor::extract(RegistrationContext& ctx) {
    if (!_impl) {
        IR_LOG_ERROR("KAZE extractor not constructed.");
        return false;
    }

    auto& fd = ctx.keypoint_data;
    auto& images = ctx.images;
    fd.type = KeypointType::KAZE;
    fd.norm_type = _norm;

    if (images.first.empty() || images.second.empty()) {
        IR_LOG_ERROR("KAZE::extract - source images are empty.");
        return false;
    }
    if (!image_utils::ensureGray(images.first, images.first_gray) ||
        !image_utils::ensureGray(images.second, images.second_gray)) {
        IR_LOG_ERROR("KAZE::extract - failed to prepare grayscale images.");
        return false;
    }

    _impl->detectAndCompute(
        images.first_gray, cv::noArray(), fd.first.keypoints, fd.first.descriptors);
    _impl->detectAndCompute(
        images.second_gray, cv::noArray(), fd.second.keypoints, fd.second.descriptors);

    IR_LOG_INFO("KAZE extracted ",
                fd.first.keypoints.size(),
                " / ",
                fd.second.keypoints.size(),
                " keypoints");
    return !fd.empty();
}

} // namespace ir

