#pragma once

#include <string>

#include <opencv2/core.hpp>

namespace ir::height_difference_evaluator {

struct Options {
    bool compensate_global_offset = false;
    int percentile = 90;
    double max_abs_error = 0.10;
    bool allow_local_noise_fallback = false;
    double local_noise_p75_max_abs_error = 0.10;
    double local_noise_min_containment = 0.90;
};

/// 重叠区域的高度差统计；高度以归一化灰度表示，范围约为 [0, 1]。
struct Statistics {
    int raw_abs_diff_count = 0;
    double raw_abs_diff_mean = -1.0;
    double raw_abs_diff_p50 = -1.0;
    double raw_abs_diff_p75 = -1.0;
    double raw_abs_diff_p90 = -1.0;
    double raw_abs_diff_p95 = -1.0;
    double raw_abs_diff_max = -1.0;
    double height_offset = -1.0;
    double compensated_mean = -1.0;
    double compensated_p50 = -1.0;
    double compensated_p75 = -1.0;
    double compensated_p90 = -1.0;
    double compensated_p95 = -1.0;
    double compensated_max = -1.0;
};

struct Result {
    Statistics statistics;
    bool pass = false;
    bool raw_pass = false;
    bool local_noise_pass = false;
    bool compensation_attempted = false;
    bool compensated_pass = false;
    std::string failure_message;
};

/// 在有效重叠区域计算原始高度差，并按需要执行局部噪声或全局偏移补偿验证。
Result evaluate(const cv::Mat& warped,
                const cv::Mat& target,
                const cv::Mat& overlap_mask,
                double local_noise_containment,
                const Options& options);

double selectRawPercentile(const Statistics& statistics, int percentile);
double selectCompensatedPercentile(const Statistics& statistics, int percentile);
double selectPercentile(int percentile, double p50, double p75, double p90, double p95);

/// 比较两个候选结果的包含率与高度差综合质量；用于选择较优初始化结果。
bool preferByContainmentAndHeight(int percentile,
                                  double min_containment,
                                  double max_abs_error,
                                  double initializer_containment,
                                  double initializer_height_difference,
                                  double direct_containment,
                                  double direct_height_difference);

} // namespace ir::height_difference_evaluator
