#include "feature/kaze_extractor.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

KazeExtractor::KazeExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    _extended      = yaml_utils::getBool (params, "extended",      false);
    _upright       = yaml_utils::getBool (params, "upright",       false);
    _threshold     = yaml_utils::getFloat(params, "threshold",     0.001f);
    _nOctaves      = yaml_utils::getInt  (params, "nOctaves",      4);
    _nOctaveLayers = yaml_utils::getInt  (params, "nOctaveLayers", 4);
    _diffusivity   = yaml_utils::getInt  (params, "diffusivity",
                                          static_cast<int>(cv::KAZE::DIFF_PM_G2));

    _impl = cv::KAZE::create(
        _extended,
        _upright,
        _threshold,
        _nOctaves,
        _nOctaveLayers,
        static_cast<cv::KAZE::DiffusivityType>(_diffusivity));

    IR_LOG_INFO("KAZE created: extended=", _extended,
                ", upright=",              _upright,
                ", threshold=",            _threshold,
                ", nOctaves=",             _nOctaves,
                ", nOctaveLayers=",        _nOctaveLayers,
                ", diffusivity=",          _diffusivity);
}

bool KazeExtractor::extract(RegistrationContext& ctx) {
    if (!_impl) {
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

    _impl->detectAndCompute(fd.first.gray,  cv::noArray(),
                            fd.first.keypoints,  fd.first.descriptors);
    _impl->detectAndCompute(fd.second.gray, cv::noArray(),
                            fd.second.keypoints, fd.second.descriptors);

    IR_LOG_INFO("KAZE extracted ", fd.first.keypoints.size(),
                " / ",            fd.second.keypoints.size(), " keypoints");
    return !fd.empty();
}

} // namespace ir

