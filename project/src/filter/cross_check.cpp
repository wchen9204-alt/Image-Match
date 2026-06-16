#include "filter/cross_check.h"

#include <opencv2/features2d.hpp>
#include <unordered_map>

#include "utils/descriptor_norm_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

CrossCheckFilter::CrossCheckFilter(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    _enabled = yaml_utils::getBool(params, "enabled", true);
    IR_LOG_INFO("CrossCheckFilter [keypoint-only] enabled=", _enabled);
}

bool CrossCheckFilter::apply(RegistrationContext& ctx) {
    if (!ctx.structure_match_data.raw_matches_knn.empty() ||
        !ctx.structure_match_data.filtered_matches.empty()) {
        IR_LOG_WARN(
            "CrossCheckFilter [keypoint-only]: structure pipeline is not supported, pass-through.");
        return true;
    }

    auto& fd = ctx.keypoint_data;
    auto& md = ctx.keypoint_match_data;

    if (!_enabled) {
        IR_LOG_INFO("CrossCheckFilter [keypoint-only] disabled - pass-through.");
        return true;
    }

    if (md.filtered_matches.empty()) {
        IR_LOG_WARN("CrossCheckFilter [keypoint-only]: no matches available to verify.");
        return false;
    }

    const std::vector<cv::DMatch> forward = md.filtered_matches;
    if (forward.empty()) {
        IR_LOG_WARN("CrossCheckFilter [keypoint-only]: forward set is empty.");
        md.filtered_matches.clear();
        return false;
    }

    const NormType norm =
        descriptor_norm_utils::resolve(NormType::UNKNOWN, fd.norm_type, fd.first.descriptors);
    cv::Ptr<cv::BFMatcher> rev = cv::BFMatcher::create(toCvNorm(norm), false);

    std::vector<cv::DMatch> reverse;
    rev->match(fd.second.descriptors, fd.first.descriptors, reverse);

    std::unordered_map<int, int> reverse_best;
    reverse_best.reserve(reverse.size());
    for (const auto& r : reverse) {
        reverse_best[r.queryIdx] = r.trainIdx;
    }

    std::vector<cv::DMatch> kept;
    kept.reserve(forward.size());
    for (const auto& m : forward) {
        const auto it = reverse_best.find(m.trainIdx);
        if (it != reverse_best.end() && it->second == m.queryIdx) {
            kept.push_back(m);
        }
    }

    IR_LOG_INFO("CrossCheckFilter [keypoint-only] kept ",
                kept.size(),
                " / ",
                forward.size(),
                " matches");
    md.filtered_matches = std::move(kept);
    return true;
}

} // namespace ir


