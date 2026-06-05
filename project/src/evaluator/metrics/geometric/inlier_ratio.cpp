#include "evaluator/metrics/geometric/inlier_ratio.h"

namespace ir {

MetricResult InlierRatioMetric::compute(const RegistrationContext& ctx, const Sample& /*sample*/) {
    MetricResult r{name(), 0.0, false, ""};

    // 点特征法：从 keypoint_match_data 计算
    const int filtered = static_cast<int>(ctx.keypoint_match_data.filtered.size());
    const int inliers = static_cast<int>(ctx.keypoint_match_data.inliers.size());
    if (filtered > 0) {
        r.value = static_cast<double>(inliers) / static_cast<double>(filtered);
        r.valid = true;
        return r;
    }

    // 结构法：从 result 中已统计的 inlier_ratio 获取
    if (ctx.result.inlier_ratio > 0.0) {
        r.value = ctx.result.inlier_ratio;
        r.valid = true;
        return r;
    }

    r.note = "no filtered matches";
    return r;
}

} // namespace ir

