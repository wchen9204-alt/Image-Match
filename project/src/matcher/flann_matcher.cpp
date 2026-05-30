#include "matcher/flann_matcher.h"

#include <opencv2/features2d.hpp>
#include <opencv2/flann/miniflann.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

NormType resolveNormType(const FeatureData& fd, NormType configuredNorm) {
    NormType effective = configuredNorm;
    if (effective == NormType::UNKNOWN) {
        effective = fd.norm_type;
    }
    if (effective == NormType::UNKNOWN) {
        effective = (fd.first.descriptors.type() == CV_8U) ? NormType::HAMMING : NormType::L2;
    }
    return effective;
}

void prepareFlannDescriptors(const FeatureData& fd, NormType effective, cv::Mat& d1, cv::Mat& d2) {
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
    auto& fd = ctx.feature_data;
    auto& md = ctx.match_data;
    md.clear();

    if (fd.first.descriptors.empty() || fd.second.descriptors.empty()) {
        IR_LOG_ERROR("FlannMatcher::match - empty descriptors.");
        return false;
    }

    const NormType effective = resolveNormType(fd, _normType);

    cv::Ptr<cv::flann::IndexParams> index;
    if (effective == NormType::HAMMING || effective == NormType::HAMMING2) {
        index = cv::makePtr<cv::flann::LshIndexParams>(
            _lshTableNumber, _lshKeySize, _lshMultiProbeLevel);
    } else {
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

    switch (_method) {
    case MatchMethod::MATCH: {
        std::vector<cv::DMatch> matches;
        matcher.match(d1, d2, matches);

        md.raw_knn.reserve(matches.size());
        for (const auto& match : matches) {
            md.raw_knn.push_back({match});
        }
        md.filtered = matches;

        IR_LOG_INFO("FlannMatcher produced ", md.filtered.size(), " matches (method=MATCH)");
        return !md.filtered.empty();
    }
    case MatchMethod::KNN:
        matcher.knnMatch(d1, d2, md.raw_knn, _knnK);
        IR_LOG_INFO("FlannMatcher produced ",
                    md.raw_knn.size(),
                    " query rows (method=KNN, k=",
                    _knnK,
                    ", norm=",
                    toString(effective),
                    ")");
        return !md.raw_knn.empty();
    case MatchMethod::RADIUS:
        matcher.radiusMatch(d1, d2, md.raw_knn, _radius);
        IR_LOG_INFO("FlannMatcher produced ",
                    md.raw_knn.size(),
                    " query rows (method=RADIUS, radius=",
                    _radius,
                    ", norm=",
                    toString(effective),
                    ")");
        return !md.raw_knn.empty();
    case MatchMethod::UNKNOWN:
    default:
        IR_LOG_ERROR("FlannMatcher::match - unsupported method: ", toString(_method));
        return false;
    }
}

} // namespace ir
