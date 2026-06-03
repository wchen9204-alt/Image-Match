#include "keypoint/orb_extractor.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

OrbExtractor::OrbExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    _nfeatures = yaml_utils::getInt(params, "nfeatures", 2000);
    _scaleFactor = yaml_utils::getFloat(params, "scaleFactor", 1.2f);
    _nlevels = yaml_utils::getInt(params, "nlevels", 8);
    _edgeThreshold = yaml_utils::getInt(params, "edgeThreshold", 31);
    _firstLevel = yaml_utils::getInt(params, "firstLevel", 0);
    _wtaK = yaml_utils::getInt(params, "WTA_K", 2);
    _scoreType = yaml_utils::getInt(params, "scoreType", static_cast<int>(cv::ORB::HARRIS_SCORE));
    _patchSize = yaml_utils::getInt(params, "patchSize", 31);
    _fastThreshold = yaml_utils::getInt(params, "fastThreshold", 20);

    if (_wtaK != 2 && _wtaK != 3 && _wtaK != 4) {
        IR_LOG_WARN("ORB WTA_K must be 2, 3, or 4; using 2 instead of ", _wtaK);
        _wtaK = 2;
    }
    _norm = (_wtaK == 3 || _wtaK == 4) ? NormType::HAMMING2 : NormType::HAMMING;

    _impl = cv::ORB::create(_nfeatures,
                            _scaleFactor,
                            _nlevels,
                            _edgeThreshold,
                            _firstLevel,
                            _wtaK,
                            static_cast<cv::ORB::ScoreType>(_scoreType),
                            _patchSize,
                            _fastThreshold);

    IR_LOG_INFO("ORB created: nfeatures=",
                _nfeatures,
                ", scaleFactor=",
                _scaleFactor,
                ", nlevels=",
                _nlevels,
                ", WTA_K=",
                _wtaK,
                ", patchSize=",
                _patchSize,
                ", norm=",
                toString(_norm));
}

bool OrbExtractor::extract(RegistrationContext& ctx) {
    if (!_impl) {
        IR_LOG_ERROR("ORB extractor not constructed.");
        return false;
    }

    auto& fd = ctx.keypoint_data;
    auto& images = ctx.images;
    fd.type = KeypointType::ORB;
    fd.norm_type = _norm;

    if (images.first.empty() || images.second.empty()) {
        IR_LOG_ERROR("ORB::extract - source images are empty.");
        return false;
    }
    if (images.first_gray.empty()) {
        cv::cvtColor(images.first, images.first_gray, cv::COLOR_BGR2GRAY);
    }
    if (images.second_gray.empty()) {
        cv::cvtColor(images.second, images.second_gray, cv::COLOR_BGR2GRAY);
    }

    _impl->detectAndCompute(
        images.first_gray, cv::noArray(), fd.first.keypoints, fd.first.descriptors);
    _impl->detectAndCompute(
        images.second_gray, cv::noArray(), fd.second.keypoints, fd.second.descriptors);

    IR_LOG_INFO("ORB extracted ",
                fd.first.keypoints.size(),
                " / ",
                fd.second.keypoints.size(),
                " keypoints");
    return !fd.empty();
}

} // namespace ir
