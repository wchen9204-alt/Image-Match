#include "matcher/bf_matcher.h"

#include <opencv2/features2d.hpp>

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

} // namespace

BfMatcher::BfMatcher(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    const std::string normString = yaml_utils::getString(params, "norm_type", "AUTO");
    const std::string methodString = yaml_utils::getString(params, "method", "match");

    _normType = normTypeFromString(normString);
    _method = matchMethodFromString(methodString);
    if (_method == MatchMethod::UNKNOWN) {
        _method = MatchMethod::MATCH;
    }

    _knnK = yaml_utils::getInt(params, "knn_k", 2);
    if (_knnK < 1) {
        _knnK = 1;
    }

    _radius = yaml_utils::getFloat(params, "radius", 100.0f);
    if (_radius <= 0.0f) {
        _radius = 100.0f;
    }

    _crossCheck = yaml_utils::getBool(params, "crossCheck", false);
    if (_crossCheck && _method != MatchMethod::MATCH) {
        IR_LOG_WARN("BFMatcher crossCheck only applies to method=MATCH; disabling it for method=",
                    toString(_method));
        _crossCheck = false;
    }

    IR_LOG_INFO("BFMatcher created: norm=",
                normString,
                " (resolved to ",
                toString(_normType),
                ")",
                ", method=",
                toString(_method),
                ", knn_k=",
                _knnK,
                ", radius=",
                _radius,
                ", crossCheck=",
                _crossCheck);
}

bool BfMatcher::match(RegistrationContext& ctx) {
    auto& fd = ctx.feature_data;
    auto& md = ctx.match_data;
    md.clear();

    if (fd.first.descriptors.empty() || fd.second.descriptors.empty()) {
        IR_LOG_ERROR("BFMatcher::match - empty descriptors.");
        return false;
    }

    const NormType effective = resolveNormType(fd, _normType);
    const int cvNorm = toCvNorm(effective);

    IR_LOG_INFO("BFMatcher matching with norm=",
                toString(effective),
                " (cv_norm=",
                cvNorm,
                "), method=",
                toString(_method),
                ", descriptors: ",
                fd.first.descriptors.rows,
                " x ",
                fd.first.descriptors.cols,
                " vs ",
                fd.second.descriptors.rows,
                " x ",
                fd.second.descriptors.cols);

    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cvNorm, _crossCheck);

    switch (_method) {
    case MatchMethod::MATCH: {
        std::vector<cv::DMatch> matches;
        matcher->match(fd.first.descriptors, fd.second.descriptors, matches);

        md.raw_knn.reserve(matches.size());
        for (const auto& match : matches) {
            md.raw_knn.push_back({match});
        }
        md.filtered = matches;

        IR_LOG_INFO("BFMatcher produced ",
                    md.filtered.size(),
                    " matches (method=MATCH, crossCheck=",
                    _crossCheck,
                    ")");
        return !md.filtered.empty();
    }
    case MatchMethod::KNN:
        matcher->knnMatch(fd.first.descriptors, fd.second.descriptors, md.raw_knn, _knnK);
        IR_LOG_INFO(
            "BFMatcher produced ", md.raw_knn.size(), " query rows (method=KNN, k=", _knnK, ")");
        return !md.raw_knn.empty();
    case MatchMethod::RADIUS:
        matcher->radiusMatch(fd.first.descriptors, fd.second.descriptors, md.raw_knn, _radius);
        IR_LOG_INFO("BFMatcher produced ",
                    md.raw_knn.size(),
                    " query rows (method=RADIUS, radius=",
                    _radius,
                    ")");
        return !md.raw_knn.empty();
    case MatchMethod::UNKNOWN:
    default:
        IR_LOG_ERROR("BFMatcher::match - unsupported method: ", toString(_method));
        return false;
    }
}

} // namespace ir
