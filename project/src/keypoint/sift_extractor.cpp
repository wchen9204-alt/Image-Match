#include "keypoint/sift_extractor.h"

#include "utils/descriptor_norm_utils.h"
#include "utils/image_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

SiftExtractor::SiftExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    _nfeatures = yaml_utils::getInt(params, "nfeatures", 0);
    _nOctaveLayers = yaml_utils::getInt(params, "nOctaveLayers", 3);
    _contrastThreshold = yaml_utils::getDouble(params, "contrastThreshold", 0.04);
    _edgeThreshold = yaml_utils::getDouble(params, "edgeThreshold", 10.0);
    _sigma = yaml_utils::getDouble(params, "sigma", 1.6);
    _norm = descriptor_norm_utils::readConfiguredNorm(cfg, NormType::L2);

    _impl =
        cv::SIFT::create(_nfeatures, _nOctaveLayers, _contrastThreshold, _edgeThreshold, _sigma);

    IR_LOG_INFO("SIFT created: nfeatures=",
                _nfeatures,
                ", nOctaveLayers=",
                _nOctaveLayers,
                ", contrast=",
                _contrastThreshold,
                ", edge=",
                _edgeThreshold,
                ", sigma=",
                _sigma,
                ", norm=",
                toString(_norm));
}

bool SiftExtractor::extract(RegistrationContext& ctx) {
    if (!_impl) {
        IR_LOG_ERROR("SIFT extractor not constructed.");
        return false;
    }

    auto& fd = ctx.keypoint_data;
    auto& images = ctx.images;
    fd.type = KeypointType::SIFT;
    fd.norm_type = _norm;

    if (images.first.empty() || images.second.empty()) {
        IR_LOG_ERROR("SIFT::extract - source images are empty.");
        return false;
    }

    if (!image_utils::ensureGray(images.first, images.first_gray) ||
        !image_utils::ensureGray(images.second, images.second_gray)) {
        IR_LOG_ERROR("SIFT::extract - failed to prepare grayscale images.");
        return false;
    }

    _impl->detectAndCompute(
        images.first_gray, cv::noArray(), fd.first.keypoints, fd.first.descriptors);
    _impl->detectAndCompute(
        images.second_gray, cv::noArray(), fd.second.keypoints, fd.second.descriptors);

    IR_LOG_INFO("SIFT extracted ",
                fd.first.keypoints.size(),
                " / ",
                fd.second.keypoints.size(),
                " keypoints");
    return !fd.empty();
}
} // namespace ir

