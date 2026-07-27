#include "matcher/keypoint/flann_matcher.h"

#include <opencv2/features2d.hpp>
#include <opencv2/flann/miniflann.hpp>

#include "utils/descriptor_norm_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

// FLANN 在浮点距离下需要 CV_32F 描述子。
void prepareFlannDescriptors(const KeypointData& fd, NormType effective, cv::Mat& d1, cv::Mat& d2) {
    d1 = fd.first.descriptors;
    d2 = fd.second.descriptors;
    if (effective == NormType::L1 || effective == NormType::L2) {
        if (d1.type() != CV_32F) {
            d1.convertTo(d1, CV_32F);
        }
        if (d2.type() != CV_32F) {
            d2.convertTo(d2, CV_32F);
        }
    }
}

} // namespace

FlannMatcher::FlannMatcher(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    _normType = normTypeFromString(yaml_utils::getString(params, "norm_type", "AUTO"));
    _method = matchMethodFromString(yaml_utils::getString(params, "method", "knn"));
    if (_method == MatchMethod::UNKNOWN) {
        _method = MatchMethod::KNN;
    }

    _knnK = yaml_utils::getInt(params, "knn_k", 2);
    if (_knnK < 1) {
        _knnK = 1;
    }

    _radius = yaml_utils::getFloat(params, "radius", 100.0f);
    if (_radius <= 0.0f) {
        _radius = 100.0f;
    }

    if (params && params["kdtree"] && params["kdtree"].IsMap()) {
        _kdTrees = yaml_utils::getInt(params["kdtree"], "trees", 5);
    }
    if (params && params["lsh"] && params["lsh"].IsMap()) {
        _lshTableNumber = yaml_utils::getInt(params["lsh"], "table_number", 12);
        _lshKeySize = yaml_utils::getInt(params["lsh"], "key_size", 20);
        _lshMultiProbeLevel = yaml_utils::getInt(params["lsh"], "multi_probe_level", 2);
    }
    if (params && params["search"] && params["search"].IsMap()) {
        _searchChecks = yaml_utils::getInt(params["search"], "checks", 50);
        _searchEps = yaml_utils::getFloat(params["search"], "eps", 0.0f);
        _searchSorted = yaml_utils::getBool(params["search"], "sorted", true);
    }

    IR_LOG_INFO("FlannMatcher created: norm=",
                toString(_normType),
                ", method=",
                toString(_method),
                ", knn_k=",
                _knnK,
                ", radius=",
                _radius,
                ", kd_trees=",
                _kdTrees,
                ", lsh_table=",
                _lshTableNumber,
                ", lsh_key=",
                _lshKeySize,
                ", lsh_probes=",
                _lshMultiProbeLevel,
                ", checks=",
                _searchChecks);
}

bool FlannMatcher::match(RegistrationContext& ctx) {
    auto& fd = ctx.keypoint_data;
    auto& md = ctx.keypoint_match_data;

    // 每次匹配前清空旧结果，避免复用上下文时混入上一组数据。
    md.clear();

    if (fd.first.descriptors.empty() || fd.second.descriptors.empty()) {
        IR_LOG_ERROR("FlannMatcher::match - empty descriptors.");
        return false;
    }

    const NormType effective =
        descriptor_norm_utils::resolve(_normType, fd.norm_type, fd.first.descriptors);

    cv::Ptr<cv::flann::IndexParams> index;
    if (effective == NormType::HAMMING || effective == NormType::HAMMING2) {
        // 二进制描述子使用 LSH，避免 KD-tree 不适配汉明空间。
        index = cv::makePtr<cv::flann::LshIndexParams>(
            _lshTableNumber, _lshKeySize, _lshMultiProbeLevel);
    } else {
        // 浮点描述子使用 KD-tree 索引。
        index = cv::makePtr<cv::flann::KDTreeIndexParams>(_kdTrees);
    }
    cv::Ptr<cv::flann::SearchParams> search =
        cv::makePtr<cv::flann::SearchParams>(_searchChecks, _searchEps, _searchSorted);

    cv::FlannBasedMatcher matcher(index, search);

    cv::Mat d1;
    cv::Mat d2;
    prepareFlannDescriptors(fd, effective, d1, d2);

    IR_LOG_INFO("FlannMatcher matching with norm=",
                toString(effective),
                ", method=",
                toString(_method),
                ", descriptors: ",
                d1.rows,
                " x ",
                d1.cols,
                " vs ",
                d2.rows,
                " x ",
                d2.cols);

    md.match_method = _method;
    switch (_method) {
    case MatchMethod::MATCH:
        matcher.match(d1, d2, md.raw_matches);
        return !md.raw_matches.empty();
    case MatchMethod::KNN:
        matcher.knnMatch(d1, d2, md.neighbour_matches_by_query, _knnK);
        md.buildRawMatchesFromNeighbours();
        return !md.raw_matches.empty();
    case MatchMethod::RADIUS:
        matcher.radiusMatch(d1, d2, md.neighbour_matches_by_query, _radius);
        md.buildRawMatchesFromNeighbours();
        return !md.raw_matches.empty();
    case MatchMethod::UNKNOWN:
    default:
        IR_LOG_ERROR("FlannMatcher::match - unsupported method: ", toString(_method));
        return false;
    }
}

} // namespace ir


