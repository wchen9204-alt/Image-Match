#include "matcher/bf_matcher.h"

#include <opencv2/features2d.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

BfMatcher::BfMatcher(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    const std::string norm_str =
        yaml_utils::getString(params, "norm_type", "AUTO");
    norm_type_  = normTypeFromString(norm_str);
    _crossCheck = yaml_utils::getBool(params, "crossCheck", false);
    knn_k_      = yaml_utils::getInt (params, "knn_k",      2);
    if (knn_k_ < 1) knn_k_ = 1;

    IR_LOG_INFO("BFMatcher created: norm=", norm_str,
                " (resolved to ", toString(norm_type_), ")",
                ", crossCheck=", _crossCheck,
                ", knn_k=",      knn_k_);
}

bool BfMatcher::match(RegistrationContext& ctx) {
    auto& fd = ctx.feature_data;
    auto& md = ctx.match_data;
    md.clear();

    if (fd.first.descriptors.empty() || fd.second.descriptors.empty()) {
        IR_LOG_ERROR("BFMatcher::match - empty descriptors.");
        return false;
    }

    // 解析实际距离类型；AUTO 时使用 FeatureData 给出的类型。
    NormType effective = norm_type_;
    if (effective == NormType::UNKNOWN) {
        effective = fd.norm_type;
    }
    if (effective == NormType::UNKNOWN) {
        // 如果未配置，则根据描述子类型兜底判断。
        effective = (fd.first.descriptors.type() == CV_8U)
                        ? NormType::HAMMING
                        : NormType::L2;
    }

    const int cv_norm = toCvNorm(effective);
    IR_LOG_INFO("BFMatcher matching with norm=", toString(effective),
                " (cv_norm=", cv_norm, "), descriptors: ",
                fd.first.descriptors.rows, " x ",
                fd.first.descriptors.cols, " vs ",
                fd.second.descriptors.rows, " x ",
                fd.second.descriptors.cols);

    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv_norm, _crossCheck);

    if (_crossCheck) {
        // OpenCV 内置 crossCheck 不能与 knnMatch 同时使用。
        std::vector<cv::DMatch> single;
        matcher->match(fd.first.descriptors, fd.second.descriptors, single);
        md.raw_knn.reserve(single.size());
        for (const auto& m : single) {
            md.raw_knn.push_back({m});
        }
        md.filtered = std::move(single);
    } else {
        matcher->knnMatch(fd.first.descriptors,
                          fd.second.descriptors,
                          md.raw_knn,
                          knn_k_);
    }

    IR_LOG_INFO("BFMatcher produced ", md.raw_knn.size(), " query rows (k=", knn_k_, ")");
    return !md.raw_knn.empty();
}

} // namespace ir

