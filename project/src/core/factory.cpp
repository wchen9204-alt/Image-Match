#include "core/factory.h"

#include <stdexcept>
#include <string>

#include "feature/akaze_extractor.h"
#include "feature/brisk_extractor.h"
#include "feature/kaze_extractor.h"
#include "feature/orb_extractor.h"
#include "feature/sift_extractor.h"
#include "feature/surf_extractor.h"

#include "matcher/bf_matcher.h"
#include "matcher/flann_matcher.h"

#include "filter/cross_check.h"
#include "filter/gms_filter.h"
#include "filter/ratio_test.h"

#include "geometry/affine_estimator.h"
#include "geometry/essential_estimator.h"
#include "geometry/fundamental_estimator.h"
#include "geometry/homography_estimator.h"

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

std::string typeOf(const YAML::Node& cfg) {
    return yaml_utils::getString(cfg, "type", "");
}

} // namespace

std::shared_ptr<IFeatureExtractor>
Factory::createFeatureExtractor(const YAML::Node& cfg) {
    const std::string t = typeOf(cfg);
    const FeatureType ft = featureTypeFromString(t);
    switch (ft) {
        case FeatureType::SIFT:  return std::make_shared<SiftExtractor>(cfg);
        case FeatureType::SURF:  return std::make_shared<SurfExtractor>(cfg);
        case FeatureType::ORB:   return std::make_shared<OrbExtractor>(cfg);
        case FeatureType::BRISK: return std::make_shared<BriskExtractor>(cfg);
        case FeatureType::KAZE:  return std::make_shared<KazeExtractor>(cfg);
        case FeatureType::AKAZE: return std::make_shared<AkazeExtractor>(cfg);
        default:
            throw std::runtime_error("Factory: unknown feature extractor type: " + t);
    }
}

std::shared_ptr<IMatcher>
Factory::createMatcher(const YAML::Node& cfg) {
    const std::string t = typeOf(cfg);
    if (t == "BF" || t == "BFMatcher" || t == "BRUTE_FORCE") {
        return std::make_shared<BfMatcher>(cfg);
    }
    if (t == "FLANN" || t == "FlannBased" || t == "FlannMatcher") {
        return std::make_shared<FlannMatcher>(cfg);
    }
    throw std::runtime_error("Factory: unknown matcher type: " + t);
}

std::shared_ptr<IFilter>
Factory::createFilter(const YAML::Node& cfg) {
    const std::string t = typeOf(cfg);
    if (t == "RATIO_TEST" || t == "RatioTest") {
        return std::make_shared<RatioTestFilter>(cfg);
    }
    if (t == "CROSS_CHECK" || t == "CrossCheck") {
        return std::make_shared<CrossCheckFilter>(cfg);
    }
    if (t == "GMS") {
        return std::make_shared<GmsFilter>(cfg);
    }
    throw std::runtime_error("Factory: unknown filter type: " + t);
}

std::shared_ptr<IGeometryEstimator>
Factory::createGeometryEstimator(const YAML::Node& cfg) {
    const std::string t = typeOf(cfg);
    const GeometryType gt = geometryTypeFromString(t);
    switch (gt) {
        case GeometryType::HOMOGRAPHY:  return std::make_shared<HomographyEstimator>(cfg);
        case GeometryType::AFFINE:      return std::make_shared<AffineEstimator>(cfg);
        case GeometryType::FUNDAMENTAL: return std::make_shared<FundamentalEstimator>(cfg);
        case GeometryType::ESSENTIAL:   return std::make_shared<EssentialEstimator>(cfg);
        default:
            throw std::runtime_error("Factory: unknown geometry estimator type: " + t);
    }
}

} // namespace ir
