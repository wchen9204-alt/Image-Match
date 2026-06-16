#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include "evaluator/evaluator.h"
#include "utils/yaml_utils.h"

namespace ir {

/// SSIM 指标（简化实现，使用 OpenCV 高斯窗口逐块计算）。
class SsimMetric : public IMetric {
public:
    explicit SsimMetric(const YAML::Node& params) {
        _window = yaml_utils::getInt(params, "window", 11);
        _sigma = yaml_utils::getDouble(params, "sigma", 1.5);
    }

    std::string name() const override { return "SSIM"; }

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

        // 简化 SSIM：逐 8x8 块计算均值/方差/协方差，取平均
        s.convertTo(s, CV_32F, 1.0 / 255.0);
        t.convertTo(t, CV_32F, 1.0 / 255.0);
        const double C1 = 0.01 * 0.01, C2 = 0.03 * 0.03;
        const int bs = std::max(4, _window / 2);

        double ssimSum = 0.0;
        int blocks = 0;
        for (int y = 0; y + bs <= s.rows; y += bs / 2) {
            for (int x = 0; x + bs <= s.cols; x += bs / 2) {
                cv::Rect roi(x, y, bs, bs);
                cv::Mat sb = s(roi), tb = t(roi);
                cv::Scalar ms, mt, devs, devt;
                cv::meanStdDev(sb, ms, devs);
                cv::meanStdDev(tb, mt, devt);
                double cov = 0.0;
                for (int dy = 0; dy < bs; ++dy) {
                    const float* ps = sb.ptr<float>(dy);
                    const float* pt = tb.ptr<float>(dy);
                    for (int dx = 0; dx < bs; ++dx)
                        cov += (ps[dx] - ms[0]) * (pt[dx] - mt[0]);
                }
                cov /= (bs * bs - 1);
                const double vs = devs[0] * devs[0], vt = devt[0] * devt[0];
                ssimSum += ((2.0 * ms[0] * mt[0] + C1) * (2.0 * cov + C2)) /
                           ((ms[0] * ms[0] + mt[0] * mt[0] + C1) * (vs + vt + C2));
                ++blocks;
            }
        }
        if (blocks == 0) {
            r.note = "no valid blocks";
            return r;
        }
        r.value = ssimSum / blocks;
        r.valid = true;
        return r;
    }

private:
    int _window = 11;
    double _sigma = 1.5;
};

} // namespace ir

