#include "matcher/flann_matcher.h"

#include <opencv2/flann/miniflann.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

FlannMatcher::FlannMatcher(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    norm_type_ = normTypeFromString(yaml_utils::getString(params, "norm_type", "AUTO"));
    knn_k_     = yaml_utils::getInt(params, "knn_k", 2);
    if (knn_k_ < 1) knn_k_ = 1;

    if (params && params["kdtree"] && params["kdtree"].IsMap()) {
        kd_trees_ = yaml_utils::getInt(params["kdtree"], "trees", 5);
    }
    if (params && params["lsh"] && params["lsh"].IsMap()) {
        lsh_table_number_      = yaml_utils::getInt(params["lsh"], "table_number",      12);
        lsh_key_size_          = yaml_utils::getInt(params["lsh"], "key_size",          20);
        lsh_multi_probe_level_ = yaml_utils::getInt(params["lsh"], "multi_probe_level", 2);
    }
    if (params && params["search"] && params["search"].IsMap()) {
        search_checks_ = yaml_utils::getInt  (params["search"], "checks", 50);
        search_eps_    = yaml_utils::getFloat(params["search"], "eps",    0.0f);
        search_sorted_ = yaml_utils::getBool (params["search"], "sorted", true);
    }

    IR_LOG_INFO("FlannMatcher created: norm=", toString(norm_type_),
                ", knn_k=",         knn_k_,
                ", kd_trees=",      kd_trees_,
                ", lsh_table=",     lsh_table_number_,
                ", lsh_key=",       lsh_key_size_,
                ", lsh_probes=",    lsh_multi_probe_level_,
                ", checks=",        search_checks_);
}

bool FlannMatcher::match(RegistrationContext& ctx) {
    auto& fd = ctx.feature_data;
    auto& md = ctx.match_data;
    md.clear();

    if (fd.first.descriptors.empty() || fd.second.descriptors.empty()) {
        IR_LOG_ERROR("FlannMatcher::match - empty descriptors.");
        return false;
    }

    NormType effective = norm_type_;
    if (effective == NormType::UNKNOWN) effective = fd.norm_type;
    if (effective == NormType::UNKNOWN) {
        effective = (fd.first.descriptors.type() == CV_8U)
                        ? NormType::HAMMING
                        : NormType::L2;
    }

    cv::Ptr<cv::flann::IndexParams> index;
    if (effective == NormType::HAMMING || effective == NormType::HAMMING2) {
        index = cv::makePtr<cv::flann::LshIndexParams>(
            lsh_table_number_, lsh_key_size_, lsh_multi_probe_level_);
    } else {
        index = cv::makePtr<cv::flann::KDTreeIndexParams>(kd_trees_);
    }
    cv::Ptr<cv::flann::SearchParams> search =
        cv::makePtr<cv::flann::SearchParams>(search_checks_, search_eps_, search_sorted_);

    cv::FlannBasedMatcher matcher(index, search);

    // KDTree 需要 CV_32F；二进制描述子走 LSH 时保持 CV_8U 即可。
    cv::Mat d1 = fd.first.descriptors;
    cv::Mat d2 = fd.second.descriptors;
    if (effective == NormType::L1 || effective == NormType::L2) {
        if (d1.type() != CV_32F) d1.convertTo(d1, CV_32F);
        if (d2.type() != CV_32F) d2.convertTo(d2, CV_32F);
    }

    matcher.knnMatch(d1, d2, md.raw_knn, knn_k_);

    IR_LOG_INFO("FlannMatcher produced ", md.raw_knn.size(),
                " query rows (k=", knn_k_, ", norm=", toString(effective), ")");
    return !md.raw_knn.empty();
}

} // namespace ir
