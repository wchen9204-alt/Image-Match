#include "evaluator/metrics/keypoint/repeatability.h"

#include <opencv2/core.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

RepeatabilityMetric::RepeatabilityMetric(const YAML::Node& params) {
    _pixelThreshold = yaml_utils::getDouble(params, "pixel_threshold", 3.0);
}

MetricResult RepeatabilityMetric::compute(const RegistrationContext& ctx, const Sample& sample) {
    MetricResult r{name(), 0.0, false, ""};

    if (!sample.has_ground_truth()) {
        r.note = "no ground-truth H";
        return r;
    }
    const auto& kp1 = ctx.keypoint_data.first.keypoints;
    const auto& kp2 = ctx.keypoint_data.second.keypoints;
    if (kp1.empty() || kp2.empty()) {
        r.note = "no keypoints";
        return r;
    }

    std::vector<cv::Point2f> p1;
    p1.reserve(kp1.size());
    for (const auto& k : kp1) {
        p1.push_back(k.pt);
    }

    std::vector<cv::Point2f> p1_in_2;
    cv::perspectiveTransform(p1, p1_in_2, sample.H_gt);

    const cv::Size t_sz = ctx.images.second.size();
    const double thr2 = _pixelThreshold * _pixelThreshold;

    int valid = 0;
    int repeated = 0;
    for (size_t i = 0; i < p1_in_2.size(); ++i) {
        const auto& q = p1_in_2[i];
        if (q.x < 0 || q.y < 0 || q.x >= t_sz.width || q.y >= t_sz.height) {
            continue;
        }

        ++valid;
        for (const auto& k : kp2) {
            const double dx = q.x - k.pt.x;
            const double dy = q.y - k.pt.y;
            if (dx * dx + dy * dy <= thr2) {
                ++repeated;
                break;
            }
        }
    }

    if (valid == 0) {
        r.note = "no keypoints fall inside target frame";
        return r;
    }

    r.value = static_cast<double>(repeated) / static_cast<double>(valid);
    r.valid = true;
    return r;
}

} // namespace ir

