#include "evaluator/metrics/geometric/inlier_ratio.h"

namespace ir {

MetricResult InlierRatioMetric::compute(const RegistrationContext& ctx, const Sample& /*sample*/) {
    MetricResult r{name(), 0.0, false, ""};

    const int filtered = static_cast<int>(ctx.match_data.filtered.size());
    const int inliers = static_cast<int>(ctx.match_data.inliers.size());

    if (filtered <= 0) {
        r.note = "no filtered matches";
        return r;
    }
    r.value = static_cast<double>(inliers) / static_cast<double>(filtered);
    r.valid = true;
    return r;
}

} // namespace ir
