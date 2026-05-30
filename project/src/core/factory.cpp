#include "core/factory.h"

#include <stdexcept>
#include <string>

#include "feature/akaze_extractor.h"
#include "feature/brisk_extractor.h"
#include "feature/kaze_extractor.h"
#include "feature/orb_extractor.h"
#include "feature/sift_extractor.h"
#include "feature/surf_extractor.h"

#include "structure/contour_extractor.h"
#include "structure/edge_extractor.h"
#include "structure/line_extractor.h"

#include "matcher/bf_matcher.h"
#include "matcher/flann_matcher.h"

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

// 工厂统一从 `type` 字段分发具体实现，保持各模块 YAML 结构一致。
std::string typeOf(const YAML::Node& cfg) {
    return yaml_utils::getString(cfg, "type", "");
}

} // namespace

std::shared_ptr<IFeatureExtractor> Factory::createFeatureExtractor(const YAML::Node& cfg) {
    const std::string t = typeOf(cfg);
    const FeatureType ft = featureTypeFromString(t);
    // 特征提取器通过统一枚举做一次标准化，降低 YAML 别名差异。
    switch (ft) {
    case FeatureType::SIFT:
        return std::make_shared<SiftExtractor>(cfg);
    case FeatureType::SURF:
        return std::make_shared<SurfExtractor>(cfg);
    case FeatureType::ORB:
        return std::make_shared<OrbExtractor>(cfg);
    case FeatureType::BRISK:
        return std::make_shared<BriskExtractor>(cfg);
    case FeatureType::KAZE:
        return std::make_shared<KazeExtractor>(cfg);
    case FeatureType::AKAZE:
        return std::make_shared<AkazeExtractor>(cfg);
    default:
        throw std::runtime_error("Factory: unknown feature extractor type: " + t);
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

std::shared_ptr<IMatcher> Factory::createMatcher(const YAML::Node& cfg) {
    const std::string t = typeOf(cfg);
    // 匹配器保留字符串别名兼容，便于与 OpenCV / 旧配置命名对齐。
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
    // 过滤器链按声明顺序运行，因此这里仅负责单个过滤器实例化。
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
    // 几何估计器通过枚举映射统一管理，避免调用方关心具体类名。
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
