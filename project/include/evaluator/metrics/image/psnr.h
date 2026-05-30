#pragma once

#include <yaml-cpp/yaml.h>

#include "evaluator/evaluator.h"

namespace ir {

/// PSNR 指标。
class PsnrMetric : public IMetric {
public:
    /// 根据 YAML 参数初始化指标。
    explicit PsnrMetric(const YAML::Node& params);

    /// 返回指标名称。
    std::string name() const override { return "PSNR"; }

    /// 计算峰值信噪比。
    MetricResult compute(const RegistrationContext& ctx, const Sample& sample) override;

private:
    double _maxValue = 255.0;
};

} // namespace ir
