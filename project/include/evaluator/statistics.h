#pragma once

#include <map>
#include <string>
#include <vector>

#include "data/evaluation_data.h"

namespace ir {

// ---------------------------------------------------------------------------
// 数据集级别的指标汇总结果。
// ---------------------------------------------------------------------------
struct MetricStats {
    int    count   = 0;
    double mean    = 0.0;
    double median  = 0.0;
    double stddev  = 0.0;
    double minv    = 0.0;
    double maxv    = 0.0;
};

// ---------------------------------------------------------------------------
// Statistics：按 pipeline 聚合每个样本的评价指标。
// ---------------------------------------------------------------------------
class Statistics {
public:
    void push(const EvaluationData& ev);

    // 指标名称到汇总统计的映射。
    std::map<std::string, MetricStats> summary() const;

    // 原始数据，供绘图或后续分析使用。
    const std::map<std::string, std::vector<double>>& raw() const { return raw_; }

    void clear() { raw_.clear(); }

private:
    std::map<std::string, std::vector<double>> raw_;
};

} // namespace ir
