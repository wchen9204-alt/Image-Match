#include "filter/ratio_test.h"

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

RatioTestFilter::RatioTestFilter(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    ratio_ = yaml_utils::getFloat(params, "ratio", 0.75f);
    IR_LOG_INFO("RatioTestFilter ratio=", ratio_);
}

bool RatioTestFilter::apply(RegistrationContext& ctx) {
    auto& md = ctx.match_data;

    if (md.raw_knn.empty()) {
        IR_LOG_WARN("RatioTestFilter: no raw_knn matches to filter.");
        md.filtered.clear();
        return false;
    }

    std::vector<cv::DMatch> kept;
    kept.reserve(md.raw_knn.size());

    for (const auto& neighbours : md.raw_knn) {
        if (neighbours.size() < 2) {
            // k<2 时无法做比值检验，保留当前单个匹配。
            if (!neighbours.empty()) kept.push_back(neighbours.front());
            continue;
        }
        const cv::DMatch& m1 = neighbours[0];
        const cv::DMatch& m2 = neighbours[1];
        if (m2.distance > 0.0f && (m1.distance / m2.distance) < ratio_) {
            kept.push_back(m1);
        }
    }

    md.filtered = std::move(kept);
    IR_LOG_INFO("RatioTestFilter kept ", md.filtered.size(),
                " / ", md.raw_knn.size(), " matches");
    return true;
}

} // namespace ir
