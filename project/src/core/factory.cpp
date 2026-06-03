#include "core/factory.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

#include "keypoint/akaze_extractor.h"
#include "keypoint/brisk_extractor.h"
#include "keypoint/kaze_extractor.h"
#include "keypoint/orb_extractor.h"
#include "keypoint/sift_extractor.h"
#include "keypoint/surf_extractor.h"

#include "matcher/keypoint/bf_matcher.h"
#include "matcher/keypoint/flann_matcher.h"
#include "matcher/structure/chamfer_associator.h"
#include "matcher/structure/hausdorff_associator.h"
#include "matcher/structure/icp_associator.h"
#include "matcher/structure/line_descriptor_associator.h"
#include "matcher/structure/line_segment_associator.h"
#include "matcher/structure/phase_correlate_associator.h"

#include "structure/contour_extractor.h"
#include "structure/edge_extractor.h"
#include "structure/line_extractor.h"

#include "filter/cross_check.h"
#include "filter/distance_distribution_filter.h"
#include "filter/distance_threshold_filter.h"
#include "filter/gms_filter.h"
#include "filter/min_distance_filter.h"
#include "filter/ratio_test.h"

#include "geometry/affine_estimator.h"
#include "geometry/homography_estimator.h"
#include "geometry/rigid_estimator.h"
#include "geometry/similarity_estimator.h"

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

// 工厂统一从 `type` 或 `method` 字段分发具体实现，保持各模块 YAML 结构一致。
std::string typeOf(const YAML::Node& cfg) {
    const std::string type = yaml_utils::getString(cfg, "type", "");
    if (!type.empty()) {
        return type;
    }
    return yaml_utils::getString(cfg, "method", "");
}

// 将 YAML 中的方法名归一化为只含大写字母/数字的 key。
// 这样 PHASE_CORRELATE / PhaseCorrelate / phase_correlate 可走同一分支。
std::string methodKey(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char c : value) {
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(std::toupper(c)));
        }
    }
    return out;
}

} // namespace

std::shared_ptr<IKeypointExtractor> Factory::createKeypointExtractor(const YAML::Node& cfg) {
    const std::string t = typeOf(cfg);
    const KeypointType ft = keypointTypeFromString(t);

    switch (ft) {
    case KeypointType::SIFT:
        return std::make_shared<SiftExtractor>(cfg);
    case KeypointType::SURF:
        return std::make_shared<SurfExtractor>(cfg);
    case KeypointType::ORB:
        return std::make_shared<OrbExtractor>(cfg);
    case KeypointType::BRISK:
        return std::make_shared<BriskExtractor>(cfg);
    case KeypointType::KAZE:
        return std::make_shared<KazeExtractor>(cfg);
    case KeypointType::AKAZE:
        return std::make_shared<AkazeExtractor>(cfg);
    default:
        throw std::runtime_error("Factory: unknown keypoint extractor type: " + t);
    }
}

std::shared_ptr<IStructureExtractor> Factory::createStructureExtractor(const YAML::Node& cfg) {
    const std::string t = typeOf(cfg);
    const StructureType st = structureTypeFromString(t);

    switch (st) {
    case StructureType::EDGE:
        return std::make_shared<EdgeExtractor>(cfg);
    case StructureType::LINE:
        return std::make_shared<LineExtractor>(cfg);
    case StructureType::CONTOUR:
        return std::make_shared<ContourExtractor>(cfg);
    default:
        throw std::runtime_error("Factory: unknown structure extractor type: " + t);
    }
}

std::shared_ptr<IStructureAssociator> Factory::createStructureAssociator(const YAML::Node& cfg) {
    const YAML::Node assoc_cfg = cfg["association"] ? cfg["association"] : cfg;
    const std::string t = typeOf(assoc_cfg);
    const std::string key = methodKey(t);
    const YAML::Node all_params = assoc_cfg["params"];

    if (key == "PHASECORRELATE") {
        const YAML::Node params =
            all_params && all_params["phase_correlate"] ? all_params["phase_correlate"]
                                                         : assoc_cfg;
        return std::make_shared<PhaseCorrelateAssociator>(params);
    }
    if (key == "CHAMFER") {
        const YAML::Node params =
            all_params && all_params["chamfer"] ? all_params["chamfer"] : assoc_cfg;
        return std::make_shared<ChamferAssociator>(params);
    }
    if (key == "HAUSDORFF") {
        const YAML::Node params =
            all_params && all_params["hausdorff"] ? all_params["hausdorff"] : assoc_cfg;
        return std::make_shared<HausdorffAssociator>(params);
    }
    if (key == "ICP") {
        const YAML::Node params = all_params && all_params["icp"] ? all_params["icp"] : assoc_cfg;
        return std::make_shared<IcpAssociator>(params);
    }
    if (key == "LINESEGMENT" || key == "LINESEGMENTS" || key == "LINESEGMENTMATCH") {
        const YAML::Node params =
            all_params && all_params["line_segment"] ? all_params["line_segment"] : assoc_cfg;
        return std::make_shared<LineSegmentAssociator>(params);
    }
    if (key == "LINEDESCRIPTOR" || key == "LINEDESCRIPTORS" || key == "LBD") {
        const YAML::Node params =
            all_params && all_params["line_descriptor"] ? all_params["line_descriptor"]
                                                         : assoc_cfg;
        return std::make_shared<LineDescriptorAssociator>(params);
    }
    throw std::runtime_error("Factory: unknown structure associator type: " + t);
}

std::shared_ptr<IMatcher> Factory::createMatcher(const YAML::Node& cfg) {
    const std::string t = typeOf(cfg);

    if (t == "BF" || t == "BFMatcher" || t == "BRUTE_FORCE") {
        return std::make_shared<BfMatcher>(cfg);
    }
    if (t == "FLANN" || t == "FlannBased" || t == "FlannMatcher") {
        return std::make_shared<FlannMatcher>(cfg);
    }
    throw std::runtime_error("Factory: unknown matcher type: " + t);
}

std::shared_ptr<IFilter> Factory::createFilter(const YAML::Node& cfg) {
    const std::string t = typeOf(cfg);

    if (t == "RATIO_TEST" || t == "RatioTest") {
        return std::make_shared<RatioTestFilter>(cfg);
    }
    if (t == "CROSS_CHECK" || t == "CrossCheck") {
        return std::make_shared<CrossCheckFilter>(cfg);
    }
    if (t == "DISTANCE_THRESHOLD" || t == "DistanceThreshold") {
        return std::make_shared<DistanceThresholdFilter>(cfg);
    }
    if (t == "MIN_DISTANCE" || t == "MinDistance" || t == "MIN_DIST") {
        return std::make_shared<MinDistanceFilter>(cfg);
    }
    if (t == "DISTANCE_DISTRIBUTION" || t == "DistanceDistribution") {
        return std::make_shared<DistanceDistributionFilter>(cfg);
    }
    if (t == "GMS") {
        return std::make_shared<GmsFilter>(cfg);
    }
    throw std::runtime_error("Factory: unknown filter type: " + t);
}

std::shared_ptr<IGeometryEstimator> Factory::createGeometryEstimator(const YAML::Node& cfg) {
    const std::string t = typeOf(cfg);
    const GeometryType gt = geometryTypeFromString(t);

    switch (gt) {
    case GeometryType::HOMOGRAPHY:
        return std::make_shared<HomographyEstimator>(cfg);
    case GeometryType::AFFINE:
        return std::make_shared<AffineEstimator>(cfg);
    case GeometryType::RIGID:
        return std::make_shared<RigidEstimator>(cfg);
    case GeometryType::SIMILARITY:
        return std::make_shared<SimilarityEstimator>(cfg);
    default:
        throw std::runtime_error("Factory: unknown geometry estimator type: " + t);
    }
}

} // namespace ir
