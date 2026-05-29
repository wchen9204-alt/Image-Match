#pragma once

#include <map>
#include <string>
#include <vector>

namespace ir {

// ---------------------------------------------------------------------------
// MetricResult：单个评价指标的结果。
// ---------------------------------------------------------------------------
struct MetricResult {
    std::string name;
    double      value   = 0.0;
    bool        valid   = false;
    std::string note;   // 可选的说明信息。
};

// ---------------------------------------------------------------------------
// EvaluationData：一次配准运行产生的全部评价指标。
// ---------------------------------------------------------------------------
struct EvaluationData {
    // 按 metrics.yaml 声明顺序保存的指标列表。
    std::vector<MetricResult> metrics;

    // 按名称快速查询指标。
    const MetricResult* find(const std::string& name) const {
        for (const auto& m : metrics) {
            if (m.name == name) return &m;
        }
        return nullptr;
    }

    // 转成扁平 key/value，只包含有效指标。
    std::map<std::string, double> asMap() const {
        std::map<std::string, double> out;
        for (const auto& m : metrics) {
            if (m.valid) out[m.name] = m.value;
        }
        return out;
    }

    void clear() { metrics.clear(); }
};

} // namespace ir
