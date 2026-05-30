#pragma once

#include <map>
#include <string>
#include <vector>

#include "data/evaluation_data.h"

namespace ir {

/// 单个指标的汇总统计结果。
struct MetricStats {
    int count = 0;
    double mean = 0.0;
    double median = 0.0;
    double stddev = 0.0;
    double minv = 0.0;
    double maxv = 0.0;
};

/// 按 pipeline 聚合评测指标并计算统计信息。
class Statistics {
public:
    /// 向统计器中加入一次评测结果。
    void push(const EvaluationData& ev);

    /// 返回所有指标的汇总统计。
    std::map<std::string, MetricStats> summary() const;

    /// 返回每个指标的原始采样值。
    const std::map<std::string, std::vector<double>>& raw() const { return _raw; }

    /// 清空所有已收集的数据。
    void clear() { _raw.clear(); }

private:
    std::map<std::string, std::vector<double>> _raw;
};

} // namespace ir
