#include "keypoint/sift_extractor.h"

#include "keypoint/keypoint_extractor_helpers.h"
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
    _augmentation_config = loadBoundaryCornerAugmentationConfig(cfg);

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

    return extractKeypointsWithBoundaryAugmentation(ctx,
                                                    KeypointType::SIFT,
                                                    _norm,
                                                    "SIFT",
                                                    *_impl,
                                                    _augmentation_config);
}

} // namespace ir
