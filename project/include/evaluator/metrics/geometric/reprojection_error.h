#pragma once

#include <cmath>
#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "evaluator/evaluator.h"
#include "utils/yaml_utils.h"

namespace ir {

/// 重投影误差指标。
class ReprojectionErrorMetric : public IMetric {
public:
    explicit ReprojectionErrorMetric(const YAML::Node& params) {
        _symmetric = yaml_utils::getBool(params, "symmetric", true);
    }

    std::string name() const override { return "REPROJECTION_ERROR"; }

    MetricResult compute(const RegistrationContext& ctx, const Sample& /*sample*/) override {
        MetricResult r{name(), 0.0, false, ""};
        const auto& gd = ctx.geometry_data;
        if (!gd.valid || gd.A.empty()) { r.note = "no valid geometry"; return r; }

        // 单侧重投影：用内点计算 source → target 再反推的对称误差
        double sum = 0.0;
        int count = 0;
        const auto& pts1 = ctx.keypoint_data.first.keypoints;
        const auto& pts2 = ctx.keypoint_data.second.keypoints;
        const auto& mask = ctx.keypoint_match_data.inlier_mask;
        const auto& filtered = ctx.keypoint_match_data.filtered;

        for (size_t i = 0; i < filtered.size() && i < mask.size(); ++i) {
            if (!mask[i]) continue;
            const auto& kp1 = pts1[filtered[i].queryIdx];
            const auto& kp2 = pts2[filtered[i].trainIdx];
            double x = gd.A.at<double>(0, 0) * kp1.pt.x + gd.A.at<double>(0, 1) * kp1.pt.y + gd.A.at<double>(0, 2);
            double y = gd.A.at<double>(1, 0) * kp1.pt.x + gd.A.at<double>(1, 1) * kp1.pt.y + gd.A.at<double>(1, 2);
            double dx = x - kp2.pt.x, dy = y - kp2.pt.y;
            sum += std::sqrt(dx * dx + dy * dy);
            ++count;
        }
        if (count == 0) { r.note = "no inliers"; return r; }
        r.value = sum / count;
        r.valid = true;
        return r;
    }

private:
    bool _symmetric = true;
};

} // namespace ir
