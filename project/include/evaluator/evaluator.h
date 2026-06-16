#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "core/context.h"
#include "data/evaluation_data.h"
#include "dataset/sample.h"

namespace ir {

/// 单个评测指标的接口。
class IMetric {
public:
    virtual ~IMetric() = default;

    /// 返回指标名称。
    virtual std::string name() const = 0;

    /// 根据上下文和样本计算指标值。
    virtual MetricResult compute(const RegistrationContext& ctx, const Sample& sample) = 0;
};

/// 评测器，负责加载指标配置并执行一组指标计算。
class Evaluator {
public:
    Evaluator() = default;

    /// 从 metrics.yaml 加载指标列表。
    bool loadFromYaml(const std::filesystem::path& yaml_path);

    /// 从已经解析的 YAML 根节点加载指标列表。
    bool loadFromNode(const YAML::Node& root);

    /// 清空当前指标列表。
    void clear() { _metrics.clear(); }

    /// 手动添加一个指标实例。
    void add(std::shared_ptr<IMetric> m) {
        if (m)
            _metrics.push_back(std::move(m));
    }

    /// 运行全部指标，并写入 `ctx.evaluation`。
    void evaluate(RegistrationContext& ctx, const Sample& sample) const;

    /// 根据指标名和参数创建具体指标实例。
    static std::shared_ptr<IMetric> createMetric(const std::string& name, const YAML::Node& params);

    /// 返回当前已加载的指标列表。
    const std::vector<std::shared_ptr<IMetric>>& metrics() const { return _metrics; }

private:
    std::vector<std::shared_ptr<IMetric>> _metrics;
};

} // namespace ir

