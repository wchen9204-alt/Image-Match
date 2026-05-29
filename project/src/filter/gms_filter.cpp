#include "filter/gms_filter.h"

#include <opencv2/xfeatures2d.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

GmsFilter::GmsFilter(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    _withRotation    = yaml_utils::getBool  (params, "withRotation",    false);
    _withScale       = yaml_utils::getBool  (params, "withScale",       false);
    _thresholdFactor = yaml_utils::getDouble(params, "thresholdFactor", 6.0);

    IR_LOG_INFO("GmsFilter: withRotation=", _withRotation,
                ", withScale=",             _withScale,
                ", thresholdFactor=",       _thresholdFactor);
}

bool GmsFilter::apply(RegistrationContext& ctx) {
    const auto& fd = ctx.feature_data;
    auto&       md = ctx.match_data;

    if (fd.first.image.empty() || fd.second.image.empty()) {
        IR_LOG_ERROR("GMS: source images empty.");
        return false;
    }

    std::vector<cv::DMatch> input;
    if (!md.filtered.empty()) {
        input = md.filtered;
    } else {
        input.reserve(md.raw_knn.size());
        for (const auto& nb : md.raw_knn) {
            if (!nb.empty()) input.push_back(nb.front());
        }
    }
    if (input.empty()) {
        IR_LOG_WARN("GMS: no input matches.");
        return false;
    }

    std::vector<cv::DMatch> kept;
    cv::xfeatures2d::matchGMS(
        fd.first.image.size(),
        fd.second.image.size(),
        fd.first.keypoints,
        fd.second.keypoints,
        input,
        kept,
        _withRotation,
        _withScale,
        _thresholdFactor);

    IR_LOG_INFO("GMS kept ", kept.size(), " / ", input.size(), " matches");
    md.filtered = std::move(kept);
    return true;
}

} // namespace ir

