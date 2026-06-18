#include "keypoint/akaze_extractor.h"

#include "utils/descriptor_norm_utils.h"
#include "utils/image_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

AkazeExtractor::AkazeExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    _descriptorType =
        yaml_utils::getInt(params, "descriptor_type", static_cast<int>(cv::AKAZE::DESCRIPTOR_MLDB));
    _descriptorSize = yaml_utils::getInt(params, "descriptor_size", 0);
    _descriptorChannels = yaml_utils::getInt(params, "descriptor_channels", 3);
    _threshold = yaml_utils::getFloat(params, "threshold", 0.001f);
    _nOctaves = yaml_utils::getInt(params, "nOctaves", 4);
    _nOctaveLayers = yaml_utils::getInt(params, "nOctaveLayers", 4);
    _diffusivity =
        yaml_utils::getInt(params, "diffusivity", static_cast<int>(cv::KAZE::DIFF_PM_G2));

    // KAZE 系描述子为浮点型，MLDB 系描述子为二进制。
    const auto dtype = static_cast<cv::AKAZE::DescriptorType>(_descriptorType);
    const NormType default_norm =
        (dtype == cv::AKAZE::DESCRIPTOR_KAZE || dtype == cv::AKAZE::DESCRIPTOR_KAZE_UPRIGHT)
            ? NormType::L2
            : NormType::HAMMING;
    _norm = descriptor_norm_utils::readConfiguredNorm(cfg, default_norm);
    _augmentation_config = loadBoundaryCornerAugmentationConfig(cfg);

    _impl = cv::AKAZE::create(dtype,
                              _descriptorSize,
                              _descriptorChannels,
                              _threshold,
                              _nOctaves,
                              _nOctaveLayers,
                              static_cast<cv::KAZE::DiffusivityType>(_diffusivity));

    IR_LOG_INFO("AKAZE created: descriptor_type=",
                _descriptorType,
                ", size=",
                _descriptorSize,
                ", channels=",
                _descriptorChannels,
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

bool AkazeExtractor::extract(RegistrationContext& ctx) {
    if (!_impl) {
        IR_LOG_ERROR("AKAZE extractor not constructed.");
        return false;
    }

    auto& fd = ctx.keypoint_data;
    auto& images = ctx.images;
    fd.type = KeypointType::AKAZE;
    fd.norm_type = _norm;

    if (images.first.empty() || images.second.empty()) {
        IR_LOG_ERROR("AKAZE::extract - source images are empty.");
        return false;
    }
    if (!image_utils::ensureGray(images.first, images.first_gray) ||
        !image_utils::ensureGray(images.second, images.second_gray)) {
        IR_LOG_ERROR("AKAZE::extract - failed to prepare grayscale images.");
        return false;
    }

    // AKAZE 描述子同样依赖 detector 内部状态，边界补点不能直接进入 compute。
    if (_augmentation_config.enabled) {
        IR_LOG_WARN("AKAZE boundary corner augmentation is disabled at runtime: "
                    "OpenCV AKAZE descriptors require detector-owned class_id.");
    }

    _impl->detectAndCompute(
        images.first_gray, cv::noArray(), fd.first.keypoints, fd.first.descriptors);
    _impl->detectAndCompute(
        images.second_gray, cv::noArray(), fd.second.keypoints, fd.second.descriptors);

    IR_LOG_INFO("AKAZE extracted ",
                fd.first.keypoints.size(),
                " / ",
                fd.second.keypoints.size(),
                " keypoints");
    return !fd.empty();
}

} // namespace ir
