#pragma once

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "core/context.h"
#include "data/evaluation_data.h"
#include "dataset/sample.h"

namespace ir {

// ---------------------------------------------------------------------------
// IMetric：单个标量评价指标的接口。
//
// 具体指标位于 evaluator/metrics/{feature,geometric,image}。
// ---------------------------------------------------------------------------
class IMetric {
public:
    virtual ~IMetric() = default;

    virtual std::string name() const = 0;

    virtual MetricResult compute(const RegistrationContext& ctx,
                                 const Sample& sample) = 0;
};

// ---------------------------------------------------------------------------
// Evaluator：根据 metrics.yaml 创建并运行一组评价指标。
// ---------------------------------------------------------------------------
class Evaluator {
public:
    Evaluator() = default;

    // 从 metrics.yaml 构建指标列表。
    bool loadFromYaml(const std::filesystem::path& yaml_path);

    // 从已解析的 YAML 节点构建指标列表。
    bool loadFromNode(const YAML::Node& root);

    // 清空当前指标。
    void clear() { metrics_.clear(); }

    // 手动添加指标实例。
    void add(std::shared_ptr<IMetric> m) {
        if (m) metrics_.push_back(std::move(m));
    }

    // 运行全部指标，并写入 ctx.evaluation。
    void evaluate(RegistrationContext& ctx, const Sample& sample) const;

    // 按名称和参数创建指标；未知名称返回 nullptr。
    static std::shared_ptr<IMetric> createMetric(const std::string& name,
                                                 const YAML::Node& params);

    const std::vector<std::shared_ptr<IMetric>>& metrics() const { return metrics_; }

private:
    std::vector<std::shared_ptr<IMetric>> metrics_;
};

} // namespace ir
