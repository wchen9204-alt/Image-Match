#include "filter/cross_check.h"

#include <opencv2/features2d.hpp>
#include <unordered_map>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

CrossCheckFilter::CrossCheckFilter(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    _enabled = yaml_utils::getBool(params, "enabled", true);
    IR_LOG_INFO("CrossCheckFilter enabled=", _enabled);
}

bool CrossCheckFilter::apply(RegistrationContext& ctx) {
    // --- 结构法路径：CrossCheck 需要描述子矩阵做反向匹配，线匹配当前不支持 ---
    if (!ctx.structure_match_data.raw_matches_knn.empty() ||
        !ctx.structure_match_data.filtered_matches.empty()) {
        IR_LOG_WARN("CrossCheckFilter [structure]: not supported for line matches, pass-through.");
        return true;
    }

    auto& fd = ctx.keypoint_data;
    auto& md = ctx.keypoint_match_data;

    if (!_enabled) {
        IR_LOG_INFO("CrossCheckFilter disabled - pass-through.");
        return true;
    }

    if (md.filtered.empty()) {
        IR_LOG_WARN("CrossCheckFilter: no matches available to verify.");
        return false;
    }

    std::vector<cv::DMatch> forward = md.filtered;

    if (forward.empty()) {
        IR_LOG_WARN("CrossCheckFilter: forward set is empty.");
        md.filtered.clear();
        return false;
    }

    // ���� 1-NN �������������������ӡ�
    NormType norm = fd.norm_type;
    if (norm == NormType::UNKNOWN) {
        norm = (fd.first.descriptors.type() == CV_8U) ? NormType::HAMMING : NormType::L2;
    }

    cv::Ptr<cv::BFMatcher> rev = cv::BFMatcher::create(toCvNorm(norm), false);

    std::vector<cv::DMatch> reverse;
    rev->match(fd.second.descriptors, fd.first.descriptors, reverse);

    // ����������������¼ÿ�� train �����ӵ���� query �����ӡ�
    std::unordered_map<int, int> reverse_best;
    reverse_best.reserve(reverse.size());
    for (const auto& r : reverse) {
        // ���������� query ���Եڶ���ͼ��train ���Ե�һ��ͼ��
        reverse_best[r.queryIdx] = r.trainIdx;
    }

    std::vector<cv::DMatch> kept;
    kept.reserve(forward.size());
    for (const auto& m : forward) {
        // ����ƥ���� query ���Ե�һ��ͼ��train ���Եڶ���ͼ��
        const auto it = reverse_best.find(m.trainIdx);
        if (it != reverse_best.end() && it->second == m.queryIdx) {
            kept.push_back(m);
        }
    }

    IR_LOG_INFO("CrossCheckFilter kept ", kept.size(), " / ", forward.size(), " matches");
    md.filtered = std::move(kept);
    return true;
}

} // namespace ir

