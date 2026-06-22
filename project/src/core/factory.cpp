#include "core/factory.h"

#include <algorithm>
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
#include "matcher/structure/contour_descriptor_associator.h"
#include "matcher/structure/hausdorff_associator.h"
#include "matcher/structure/icp_associator.h"
#include "matcher/structure/line_descriptor_associator.h"
#include "matcher/structure/line_segment_associator.h"

#include "structure/contour_extractor.h"
#include "structure/edge_extractor.h"
#include "structure/line_extractor.h"

#include "filter/cross_check.h"
#include "filter/distance_distribution_filter.h"
#include "filter/distance_threshold_filter.h"
#include "filter/gms_filter.h"
#include "filter/min_distance_filter.h"
#include "filter/ratio_test.h"
#include "filter/pairwise_rigid_consistency_filter.h"

#include "direct/dense/dis_flow_aligner.h"
#include "direct/dense/farneback_flow_aligner.h"
#include "direct/frequency/fourier_mellin_aligner.h"
#include "direct/global/ecc_aligner.h"
#include "direct/global/esm_rigid_aligner.h"
#include "direct/sparse/klt_sparse_aligner.h"

#include "geometry/affine_estimator.h"
#include "geometry/homography_estimator.h"
#include "geometry/rigid_estimator.h"
#include "geometry/similarity_estimator.h"

#include "utils/logger.h"
#include "utils/string_utils.h"
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
    const std::string key = string_utils::normalizedKey(t);
    const YAML::Node all_params = assoc_cfg["params"];

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
    if (key == "CONTOURDESCRIPTOR" || key == "CONTOUR_DESCRIPTOR") {
        const YAML::Node params =
            all_params && all_params["contour_descriptor"] ? all_params["contour_descriptor"]
                                                            : assoc_cfg;
        return std::make_shared<ContourDescriptorAssociator>(params);
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
    const std::string key = string_utils::normalizedKey(t);

    if (key == "RATIOTEST") {
        return std::make_shared<RatioTestFilter>(cfg);
    }
    if (key == "CROSSCHECK") {
        return std::make_shared<CrossCheckFilter>(cfg);
    }
    if (key == "DISTANCETHRESHOLD") {
        return std::make_shared<DistanceThresholdFilter>(cfg);
    }
    if (key == "MINDISTANCE" || key == "MINDIST") {
        return std::make_shared<MinDistanceFilter>(cfg);
    }
    if (key == "DISTANCEDISTRIBUTION") {
        return std::make_shared<DistanceDistributionFilter>(cfg);
    }
    if (key == "GMS") {
        return std::make_shared<GmsFilter>(cfg);
    }
    if (key == "PAIRWISERIGIDCONSISTENCY" || key == "RIGIDCONSISTENCY") {
        return std::make_shared<PairwiseRigidConsistencyFilter>(cfg);
    }
    throw std::runtime_error("Factory: unknown filter type: " + t);
}

std::shared_ptr<IDirectAligner> Factory::createDirectAligner(const YAML::Node& cfg) {
    const std::string method =
        yaml_utils::getString(cfg, "method", yaml_utils::getString(cfg, "type"));
    const std::string key = string_utils::normalizedKey(method);

    if (key == "ECC") {
        return std::make_shared<EccAligner>(cfg);
    }
    if (key == "ESMRIGID" || key == "RIGIDESM" || key == "ESM") {
        return std::make_shared<EsmRigidAligner>(cfg);
    }
    if (key == "FOURIERMELLIN" || key == "FOURIERMELLINTRANSFORM" ||
        key == "FOURIERMELLINALIGNER" || key == "DIRECTFOURIERMELLIN" || key == "FMT") {
        return std::make_shared<DirectFourierMellinAligner>(cfg);
    }
    if (key == "KLT" || key == "PYRAMIDALKLT" || key == "SPARSEKLT") {
        return std::make_shared<KltSparseAligner>(cfg);
    }
    if (key == "FARNEBACK" || key == "FARNEBACKFLOW") {
        return std::make_shared<FarnebackFlowAligner>(cfg);
    }
    if (key == "DIS" || key == "DISFLOW" || key == "DENSEINVERSESEARCH") {
        return std::make_shared<DisFlowAligner>(cfg);
    }

    throw std::runtime_error("Factory: unknown direct aligner method: " + method);
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
