#include "keypoint/brisk_extractor.h"

#include "keypoint/keypoint_extractor_helpers.h"
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
    _augmentation_config = loadBoundaryCornerAugmentationConfig(cfg);

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

    return extractKeypointsWithBoundaryAugmentation(ctx,
                                                    KeypointType::BRISK,
                                                    _norm,
                                                    "BRISK",
                                                    *_impl,
                                                    _augmentation_config);
}

} // namespace ir
