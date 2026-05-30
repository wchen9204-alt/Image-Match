#pragma once

#include <yaml-cpp/yaml.h>

#include "evaluator/evaluator.h"

namespace ir {

/// SSIM 指标。
class SsimMetric : public IMetric {
public:
    /// 根据 YAML 参数初始化指标。
    explicit SsimMetric(const YAML::Node& params);

    /// 返回指标名称。
    std::string name() const override { return "SSIM"; }

    /// 计算结构相似度。
    MetricResult compute(const RegistrationContext& ctx, const Sample& sample) override;

private:
    int _window = 11;
    double _sigma = 1.5;
};

} // namespace ir
