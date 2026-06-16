#include "filter/ratio_test.h"

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

// 读取比值检验阈值，用于剔除歧义较大的最近邻匹配。
RatioTestFilter::RatioTestFilter(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    _ratio = yaml_utils::getFloat(params, "ratio", 0.75f);
    IR_LOG_INFO("RatioTestFilter ratio=", _ratio);
}

bool RatioTestFilter::apply(RegistrationContext& ctx) {
    // 结构法路径：从结构匹配数据中读取近邻候选匹配。
    auto& smd = ctx.structure_match_data;
    if (!smd.raw_matches_knn.empty()) {
        std::vector<cv::DMatch> kept;
        kept.reserve(smd.raw_matches_knn.size());
        // 步骤一：对每个查询行的第一、第二候选做距离比值检验。
        for (const auto& neighbours : smd.raw_matches_knn) {
            if (neighbours.size() < 2) {
                if (!neighbours.empty()) {
                    kept.push_back(neighbours.front());
                }
                continue;
            }
            const cv::DMatch& m1 = neighbours[0];
            const cv::DMatch& m2 = neighbours[1];
            if (m2.distance > 0.0f && (m1.distance / m2.distance) < _ratio) {
                kept.push_back(m1);
            }
        }
        // 步骤二：将保留下来的结构匹配写回，供后续过滤器继续处理。
        smd.filtered_matches = std::move(kept);
        IR_LOG_INFO("RatioTestFilter [structure] kept ",
                    smd.filtered_matches.size(),
                    " / ",
                    smd.raw_matches_knn.size(),
                    " matches");
        return true;
    }

    // 点特征法路径：将原始近邻候选过滤为当前筛选结果。
    auto& md = ctx.keypoint_match_data;
    if (md.raw_matches_by_query.empty()) {
        IR_LOG_WARN("RatioTestFilter: no raw_matches_by_query matches to filter.");
        md.filtered_matches.clear();
        return false;
    }

    std::vector<cv::DMatch> kept;
    kept.reserve(md.raw_matches_by_query.size());
    // 步骤一：逐行对点特征近邻候选执行比值检验。
    for (const auto& neighbours : md.raw_matches_by_query) {
        if (neighbours.size() < 2) {
            if (!neighbours.empty()) {
                kept.push_back(neighbours.front());
            }
            continue;
        }
        const cv::DMatch& m1 = neighbours[0];
        const cv::DMatch& m2 = neighbours[1];
        if (m2.distance > 0.0f && (m1.distance / m2.distance) < _ratio) {
            kept.push_back(m1);
        }
    }

    // 步骤二：用通过检验的结果覆盖当前筛选结果。
    md.filtered_matches = std::move(kept);
    IR_LOG_INFO("RatioTestFilter [keypoint] kept ",
                md.filtered_matches.size(),
                " / ",
                md.raw_matches_by_query.size(),
                " matches");
    return true;
}

}


