#include "filter/ratio_test.h"

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

RatioTestFilter::RatioTestFilter(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    _ratio = yaml_utils::getFloat(params, "ratio", 0.75f);
    IR_LOG_INFO("RatioTestFilter ratio=", _ratio);
}

bool RatioTestFilter::apply(RegistrationContext& ctx) {
    // --- 结构法路径：从 structure_match_data 读取 KNN ---
    auto& smd = ctx.structure_match_data;
    if (!smd.raw_matches_knn.empty()) {
        std::vector<cv::DMatch> kept;
        kept.reserve(smd.raw_matches_knn.size());
        for (const auto& neighbours : smd.raw_matches_knn) {
            if (neighbours.size() < 2) {
                if (!neighbours.empty())
                    kept.push_back(neighbours.front());
                continue;
            }
            const cv::DMatch& m1 = neighbours[0];
            const cv::DMatch& m2 = neighbours[1];
            if (m2.distance > 0.0f && (m1.distance / m2.distance) < _ratio) {
                kept.push_back(m1);
            }
        }
        smd.filtered_matches = std::move(kept);
        IR_LOG_INFO("RatioTestFilter [structure] kept ",
                    smd.filtered_matches.size(),
                    " / ",
                    smd.raw_matches_knn.size(),
                    " matches");
        return true;
    }

    // --- 点特征路径（原有逻辑） ---
    auto& md = ctx.keypoint_match_data;

    if (md.raw_knn.empty()) {
        IR_LOG_WARN("RatioTestFilter: no raw_knn matches to filter.");
        md.filtered.clear();
        return false;
    }

    std::vector<cv::DMatch> kept;
    kept.reserve(md.raw_knn.size());

    for (const auto& neighbours : md.raw_knn) {
        if (neighbours.size() < 2) {
            if (!neighbours.empty())
                kept.push_back(neighbours.front());
            continue;
        }
        const cv::DMatch& m1 = neighbours[0];
        const cv::DMatch& m2 = neighbours[1];
        if (m2.distance > 0.0f && (m1.distance / m2.distance) < _ratio) {
            kept.push_back(m1);
        }
    }

    md.filtered = std::move(kept);
    IR_LOG_INFO("RatioTestFilter [keypoint] kept ",
                md.filtered.size(),
                " / ",
                md.raw_knn.size(),
                " matches");
    return true;
}

} // namespace ir

