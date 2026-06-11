#include "evaluator/metrics/geometric/inlier_ratio.h"

#include "data/correspondence_view.h"

namespace ir {

MetricResult InlierRatioMetric::compute(const RegistrationContext& ctx, const Sample& /*sample*/) {
    MetricResult r{name(), 0.0, false, ""};

    // 指标优先按当前上下文显式来源读取对应点视图，避免在多方法族之间猜测数据源。
    const CorrespondenceSource source = correspondenceSourceFromContext(ctx);
    const CorrespondenceView view =
        source == CorrespondenceSource::NONE ? buildBestCorrespondenceView(ctx)
                                             : buildCorrespondenceView(ctx, source);
    if (!view.empty()) {
        r.value = static_cast<double>(view.inlierCount()) / static_cast<double>(view.filteredCount());
        r.valid = true;
        r.note = view.source_name;
        return r;
    }

    // ECC/相位相关等非点对直接法没有对应点视图，使用几何阶段已写入的置信度。
    if (ctx.result.inlier_ratio > 0.0) {
        r.value = ctx.result.inlier_ratio;
        r.valid = true;
        return r;
    }

    r.note = "no correspondences";
    return r;
}

} // namespace ir

