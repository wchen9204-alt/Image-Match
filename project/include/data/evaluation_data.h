#pragma once

#include <map>
#include <string>
#include <vector>

namespace ir {

/// 单个评测指标的结果。
struct MetricResult {
    /// 指标名。
    std::string name;
    /// 指标数值。
    double value = 0.0;
    /// 指标结果是否有效。
    bool valid = false;
    /// 可选说明信息。
    std::string note;
};

/// 一次配准运行的所有评测指标。
struct EvaluationData {
    /// 按 metrics.yaml 中的顺序保存的指标列表。
    std::vector<MetricResult> metrics;

    /// 按名称查找指标。
    const MetricResult* find(const std::string& name) const {
        for (const auto& m : metrics) {
            if (m.name == name)
                return &m;
        }
        return nullptr;
    }

    /// 将有效指标转换为 key/value 形式，便于统计和导出。
    std::map<std::string, double> asMap() const {
        std::map<std::string, double> out;
        for (const auto& m : metrics) {
            if (m.valid)
                out[m.name] = m.value;
        }
        return out;
    }

    /// 清空指标列表。
    void clear() { metrics.clear(); }
};

} // namespace ir

