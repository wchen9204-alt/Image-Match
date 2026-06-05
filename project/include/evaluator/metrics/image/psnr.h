#pragma once

#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include "evaluator/evaluator.h"
#include "utils/yaml_utils.h"

namespace ir {

/// PSNR 指标。
class PsnrMetric : public IMetric {
public:
    explicit PsnrMetric(const YAML::Node& params) {
        _maxValue = yaml_utils::getDouble(params, "max_value", 255.0);
    }

    std::string name() const override { return "PSNR"; }

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

        cv::Mat diff;
        s.convertTo(s, CV_32F);
        t.convertTo(t, CV_32F);
        cv::absdiff(s, t, diff);
        diff = diff.mul(diff);
        const double mse = cv::mean(diff)[0];
        if (mse < 1e-10) { r.value = 100.0; r.valid = true; return r; }
        r.value = 10.0 * std::log10((_maxValue * _maxValue) / mse);
        r.valid = true;
        return r;
    }

private:
    double _maxValue = 255.0;
};

} // namespace ir
