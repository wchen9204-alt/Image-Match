#pragma once

#include <yaml-cpp/yaml.h>

#include "evaluator/evaluator.h"

namespace ir {

/// 重投影误差指标。
class ReprojectionErrorMetric : public IMetric {
public:
    /// 根据 YAML 参数初始化指标。
    explicit ReprojectionErrorMetric(const YAML::Node& params);

    /// 返回指标名称。
    std::string name() const override { return "REPROJECTION_ERROR"; }

    /// 计算平均重投影误差。
    MetricResult compute(const RegistrationContext& ctx, const Sample& sample) override;

private:
    bool _symmetric = true;
};

} // namespace ir
