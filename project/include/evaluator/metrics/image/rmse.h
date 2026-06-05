#pragma once

#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include "evaluator/evaluator.h"

namespace ir {

/// RMSE 指标。
class RmseMetric : public IMetric {
public:
    explicit RmseMetric(const YAML::Node& /*params*/) {}

    std::string name() const override { return "RMSE"; }

    MetricResult compute(const RegistrationContext& ctx, const Sample& /*sample*/) override {
        MetricResult r;
        r.name = name();
        if (ctx.warped_image.empty() || ctx.images.second.empty()) {
            r.note = "warped or target image empty";
            return r;
        }
        cv::Mat s, t;
        if (ctx.warped_image.channels() == 1) s = ctx.warped_image;
        else cv::cvtColor(ctx.warped_image, s, cv::COLOR_BGR2GRAY);
        if (ctx.images.second.channels() == 1) t = ctx.images.second;
        else cv::cvtColor(ctx.images.second, t, cv::COLOR_BGR2GRAY);
        if (s.size() != t.size()) { r.note = "size mismatch"; return r; }

        s.convertTo(s, CV_32F);
        t.convertTo(t, CV_32F);
        cv::Mat diff;
        cv::absdiff(s, t, diff);
        diff = diff.mul(diff);
        r.value = std::sqrt(cv::mean(diff)[0]);
        r.valid = true;
        return r;
    }
};

} // namespace ir
