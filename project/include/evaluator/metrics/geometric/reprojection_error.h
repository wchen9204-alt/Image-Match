#pragma once

#include <cmath>
#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "data/correspondence_view.h"
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
        if (!gd.valid || (gd.A.empty() && gd.H.empty())) { r.note = "no valid geometry"; return r; }

        // 通过统一对应点视图读取点对，优先使用上下文显式来源。
        const CorrespondenceSource source = correspondenceSourceFromContext(ctx);
        const CorrespondenceView view =
            source == CorrespondenceSource::NONE ? buildBestCorrespondenceView(ctx)
                                                 : buildCorrespondenceView(ctx, source);
        if (view.empty()) { r.note = "no correspondences"; return r; }

        // 单侧重投影：用内点计算 source -> target 的平均像素误差。
        double sum = 0.0;
        int count = 0;
        for (size_t i = 0; i < view.filtered.size(); ++i) {
            if (!view.inlier_mask.empty() &&
                (i >= view.inlier_mask.size() || view.inlier_mask[i] == 0)) {
                continue;
            }

            const auto& m = view.filtered[i];
            if (m.queryIdx < 0 || m.trainIdx < 0 ||
                m.queryIdx >= static_cast<int>(view.first_keypoints.size()) ||
                m.trainIdx >= static_cast<int>(view.second_keypoints.size())) {
                continue;
            }

            const cv::Point2f p1 = view.first_keypoints[m.queryIdx].pt;
            const cv::Point2f p2 = view.second_keypoints[m.trainIdx].pt;
            double x = 0.0;
            double y = 0.0;
            if (!gd.A.empty()) {
                x = gd.A.at<double>(0, 0) * p1.x + gd.A.at<double>(0, 1) * p1.y + gd.A.at<double>(0, 2);
                y = gd.A.at<double>(1, 0) * p1.x + gd.A.at<double>(1, 1) * p1.y + gd.A.at<double>(1, 2);
            } else {
                const double w = gd.H.at<double>(2, 0) * p1.x + gd.H.at<double>(2, 1) * p1.y + gd.H.at<double>(2, 2);
                if (std::abs(w) < 1e-12) {
                    continue;
                }
                x = (gd.H.at<double>(0, 0) * p1.x + gd.H.at<double>(0, 1) * p1.y + gd.H.at<double>(0, 2)) / w;
                y = (gd.H.at<double>(1, 0) * p1.x + gd.H.at<double>(1, 1) * p1.y + gd.H.at<double>(1, 2)) / w;
            }
            double dx = x - p2.x, dy = y - p2.y;
            sum += std::sqrt(dx * dx + dy * dy);
            ++count;
        }
        if (count == 0) { r.note = "no inliers"; return r; }
        r.value = sum / count;
        r.valid = true;
        r.note = view.source_name;
        return r;
    }

private:
    bool _symmetric = true;
};

} // namespace ir

