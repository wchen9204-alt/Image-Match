#include "matcher/keypoint/bf_matcher.h"

#include <opencv2/features2d.hpp>

#include "utils/descriptor_norm_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

BfMatcher::BfMatcher(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    const std::string norm_string = yaml_utils::getString(params, "norm_type", "AUTO");
    const std::string method_string = yaml_utils::getString(params, "method", "match");

    _normType = normTypeFromString(norm_string);
    _method = matchMethodFromString(method_string);
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
        // Cross-check 只适用于一对一最近邻匹配。
        IR_LOG_WARN("BFMatcher crossCheck only applies to method=MATCH; disabling it for method=",
                    toString(_method));
        _crossCheck = false;
    }

    IR_LOG_INFO("BFMatcher created: norm=",
                norm_string,
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
    auto& fd = ctx.keypoint_data;
    auto& md = ctx.keypoint_match_data;

    // 每次匹配前清空旧结果，避免复用上下文时混入上一组数据。
    md.clear();

    if (fd.first.descriptors.empty() || fd.second.descriptors.empty()) {
        IR_LOG_ERROR("BFMatcher::match - empty descriptors.");
        return false;
    }

    const NormType effective =
        descriptor_norm_utils::resolve(_normType, fd.norm_type, fd.first.descriptors);
    const int cv_norm = toCvNorm(effective);

    IR_LOG_INFO("BFMatcher matching with norm=",
                toString(effective),
                " (cv_norm=",
                cv_norm,
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

    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv_norm, _crossCheck);

    switch (_method) {
    case MatchMethod::MATCH: {
        // 将 MATCH 结果包装成单元素 KNN 行，便于后续过滤器复用。
        std::vector<cv::DMatch> matches;
        matcher->match(fd.first.descriptors, fd.second.descriptors, matches);

        md.raw_matches_by_query.reserve(matches.size());
        for (const auto& match : matches) {
            md.raw_matches_by_query.push_back({match});
        }
        md.filtered_matches = matches;

        IR_LOG_INFO("BFMatcher produced ",
                    md.filtered_matches.size(),
                    " matches (method=MATCH, crossCheck=",
                    _crossCheck,
                    ")");
        return !md.filtered_matches.empty();
    }
    case MatchMethod::KNN:
        // KNN 保留邻居排序信息，供 ratio test 等过滤器使用。
        matcher->knnMatch(fd.first.descriptors, fd.second.descriptors, md.raw_matches_by_query, _knnK);
        IR_LOG_INFO(
            "BFMatcher produced ", md.raw_matches_by_query.size(), " query rows (method=KNN, k=", _knnK, ")");
        return !md.raw_matches_by_query.empty();
    case MatchMethod::RADIUS:
        // 半径匹配保留局部邻域候选，后续阶段再决定如何筛选。
        matcher->radiusMatch(fd.first.descriptors, fd.second.descriptors, md.raw_matches_by_query, _radius);
        IR_LOG_INFO("BFMatcher produced ",
                    md.raw_matches_by_query.size(),
                    " query rows (method=RADIUS, radius=",
                    _radius,
                    ")");
        return !md.raw_matches_by_query.empty();
    case MatchMethod::UNKNOWN:
    default:
        IR_LOG_ERROR("BFMatcher::match - unsupported method: ", toString(_method));
        return false;
    }
}

} // namespace ir


