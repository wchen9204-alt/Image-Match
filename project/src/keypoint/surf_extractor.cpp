#include "keypoint/surf_extractor.h"

#include "keypoint/keypoint_extractor_helpers.h"
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
    _augmentation_config = loadBoundaryCornerAugmentationConfig(cfg);

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

    return extractKeypointsWithBoundaryAugmentation(ctx,
                                                    KeypointType::SURF,
                                                    _norm,
                                                    "SURF",
                                                    *_impl,
                                                    _augmentation_config);
}

} // namespace ir
