#include "matcher/bf_matcher.h"

#include <opencv2/features2d.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

// 距离类型优先使用显式配置，其次继承特征阶段约定，最后再按描述子类型兜底推断。
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
        // Cross-check 要求一一最近邻关系，与 k-NN / radius 结果语义不兼容。
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

    // 每次匹配前清空旧结果，避免复用上下文时把不同图像对的数据混在一起。
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
        // `MATCH` 分支统一包装成单元素 knn 行，便于后续过滤器复用同一数据结构。
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
        // k-NN 结果保留候选邻居排序信息，供 ratio test 等过滤器继续使用。
        matcher->knnMatch(fd.first.descriptors, fd.second.descriptors, md.raw_knn, _knnK);
        IR_LOG_INFO(
            "BFMatcher produced ", md.raw_knn.size(), " query rows (method=KNN, k=", _knnK, ")");
        return !md.raw_knn.empty();
    case MatchMethod::RADIUS:
        // 半径搜索更适合保留局部邻域密度信息，后续阶段自行决定如何裁剪。
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
