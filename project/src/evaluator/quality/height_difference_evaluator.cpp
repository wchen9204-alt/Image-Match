#include "evaluator/quality/height_difference_evaluator.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace ir::height_difference_evaluator {
namespace {

struct SortedStatistics {
    int count = 0;
    double mean = -1.0;
    double p50 = -1.0;
    double p75 = -1.0;
    double p90 = -1.0;
    double p95 = -1.0;
    double max = -1.0;
};

double percentileValue(const std::vector<float>& sorted_errors, double percentile) {
    const size_t index = std::min(
        sorted_errors.size() - 1,
        static_cast<size_t>(std::ceil(static_cast<double>(sorted_errors.size()) * percentile)) -
            1);
    return static_cast<double>(sorted_errors[index]);
}

SortedStatistics summarize(std::vector<float>& errors) {
    SortedStatistics statistics;
    if (errors.empty()) {
        return statistics;
    }
    std::sort(errors.begin(), errors.end());
    statistics.count = static_cast<int>(errors.size());
    statistics.mean = std::accumulate(errors.begin(), errors.end(), 0.0) /
                      static_cast<double>(errors.size());
    statistics.p50 = percentileValue(errors, 0.50);
    statistics.p75 = percentileValue(errors, 0.75);
    statistics.p90 = percentileValue(errors, 0.90);
    statistics.p95 = percentileValue(errors, 0.95);
    statistics.max = static_cast<double>(errors.back());
    return statistics;
}

Statistics computeStatistics(const cv::Mat& warped,
                             const cv::Mat& target,
                             const cv::Mat& overlap_mask,
                             bool compensate_global_offset) {
    Statistics statistics;
    if (warped.empty() || target.empty() || warped.size() != target.size() ||
        overlap_mask.empty() || overlap_mask.size() != warped.size() ||
        cv::countNonZero(overlap_mask) == 0) {
        return statistics;
    }

    cv::Mat warped_gray;
    cv::Mat target_gray;
    if (warped.channels() == 1) {
        warped_gray = warped;
    } else {
        cv::cvtColor(warped, warped_gray, cv::COLOR_BGR2GRAY);
    }
    if (target.channels() == 1) {
        target_gray = target;
    } else {
        cv::cvtColor(target, target_gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat warped_float;
    cv::Mat target_float;
    warped_gray.convertTo(warped_float, CV_32F, 1.0 / 255.0);
    target_gray.convertTo(target_float, CV_32F, 1.0 / 255.0);

    std::vector<float> raw_errors;
    std::vector<float> height_offsets;
    raw_errors.reserve(static_cast<size_t>(cv::countNonZero(overlap_mask)));
    height_offsets.reserve(raw_errors.capacity());
    for (int y = 0; y < warped_float.rows; ++y) {
        const auto* warped_row = warped_float.ptr<float>(y);
        const auto* target_row = target_float.ptr<float>(y);
        const auto* mask_row = overlap_mask.ptr<unsigned char>(y);
        for (int x = 0; x < warped_float.cols; ++x) {
            if (mask_row[x] == 0) {
                continue;
            }
            const float delta = target_row[x] - warped_row[x];
            raw_errors.push_back(std::abs(delta));
            if (compensate_global_offset) {
                height_offsets.push_back(delta);
            }
        }
    }
    if (raw_errors.empty()) {
        return statistics;
    }

    const SortedStatistics raw = summarize(raw_errors);
    statistics.raw_abs_diff_count = raw.count;
    statistics.raw_abs_diff_mean = raw.mean;
    statistics.raw_abs_diff_p50 = raw.p50;
    statistics.raw_abs_diff_p75 = raw.p75;
    statistics.raw_abs_diff_p90 = raw.p90;
    statistics.raw_abs_diff_p95 = raw.p95;
    statistics.raw_abs_diff_max = raw.max;
    statistics.height_offset = 0.0;
    if (!compensate_global_offset) {
        return statistics;
    }

    std::sort(height_offsets.begin(), height_offsets.end());
    const size_t trim_each_side = static_cast<size_t>(
        std::floor(static_cast<double>(height_offsets.size()) * 0.05));
    const size_t begin = std::min(trim_each_side, height_offsets.size());
    const size_t end = height_offsets.size() > begin
                           ? std::max(begin + 1, height_offsets.size() - trim_each_side)
                           : begin;
    const auto first = height_offsets.begin() + static_cast<std::ptrdiff_t>(begin);
    const auto last = height_offsets.begin() + static_cast<std::ptrdiff_t>(end);
    const float trimmed_mean = first < last
                                   ? static_cast<float>(std::accumulate(first, last, 0.0) /
                                                        static_cast<double>(last - first))
                                   : 0.0F;
    statistics.height_offset = trimmed_mean;

    std::vector<float> compensated_errors;
    compensated_errors.reserve(raw_errors.size());
    for (int y = 0; y < warped_float.rows; ++y) {
        const auto* warped_row = warped_float.ptr<float>(y);
        const auto* target_row = target_float.ptr<float>(y);
        const auto* mask_row = overlap_mask.ptr<unsigned char>(y);
        for (int x = 0; x < warped_float.cols; ++x) {
            if (mask_row[x] == 0) {
                continue;
            }
            const float corrected_warped =
                std::clamp(warped_row[x] + trimmed_mean, 0.0F, 1.0F);
            compensated_errors.push_back(std::abs(corrected_warped - target_row[x]));
        }
    }
    const SortedStatistics compensated = summarize(compensated_errors);
    statistics.compensated_mean = compensated.mean;
    statistics.compensated_p50 = compensated.p50;
    statistics.compensated_p75 = compensated.p75;
    statistics.compensated_p90 = compensated.p90;
    statistics.compensated_p95 = compensated.p95;
    statistics.compensated_max = compensated.max;
    return statistics;
}

} // namespace

double selectRawPercentile(const Statistics& statistics, int percentile) {
    switch (percentile) {
    case 50: return statistics.raw_abs_diff_p50;
    case 75: return statistics.raw_abs_diff_p75;
    case 90: return statistics.raw_abs_diff_p90;
    case 95: return statistics.raw_abs_diff_p95;
    default: return -1.0;
    }
}

double selectCompensatedPercentile(const Statistics& statistics, int percentile) {
    switch (percentile) {
    case 50: return statistics.compensated_p50;
    case 75: return statistics.compensated_p75;
    case 90: return statistics.compensated_p90;
    case 95: return statistics.compensated_p95;
    default: return -1.0;
    }
}

double selectPercentile(int percentile, double p50, double p75, double p90, double p95) {
    switch (percentile) {
    case 50: return p50;
    case 75: return p75;
    case 90: return p90;
    case 95: return p95;
    default: return -1.0;
    }
}

bool preferByContainmentAndHeight(int percentile,
                                  double min_containment,
                                  double max_abs_error,
                                  double initializer_containment,
                                  double initializer_height_difference,
                                  double direct_containment,
                                  double direct_height_difference) {
    if (percentile != 50 && percentile != 75 && percentile != 90 && percentile != 95 ||
        min_containment < 0.0 || min_containment >= 1.0 || max_abs_error <= 0.0 ||
        initializer_height_difference < 0.0 || direct_height_difference < 0.0) {
        return false;
    }
    const auto score = [min_containment, max_abs_error](double containment,
                                                         double height_difference) {
        const double containment_score =
            std::clamp((containment - min_containment) / (1.0 - min_containment), 0.0, 1.0);
        const double height_score =
            std::clamp(1.0 - height_difference / max_abs_error, 0.0, 1.0);
        return 0.35 * containment_score + 0.65 * height_score;
    };
    return score(initializer_containment, initializer_height_difference) >
           score(direct_containment, direct_height_difference) + 1e-6;
}

Result evaluate(const cv::Mat& warped,
                const cv::Mat& target,
                const cv::Mat& overlap_mask,
                double local_noise_containment,
                const Options& options) {
    Result result;
    result.statistics = computeStatistics(warped, target, overlap_mask, false);
    const double raw_percentile = selectRawPercentile(result.statistics, options.percentile);
    if (raw_percentile < 0.0) {
        result.failure_message =
            "warp height difference validation failed: unsupported or unavailable percentile " +
            std::to_string(options.percentile);
        return result;
    }

    result.raw_pass = raw_percentile <= options.max_abs_error;
    if (result.raw_pass) {
        result.pass = true;
        return result;
    }

    result.local_noise_pass =
        options.allow_local_noise_fallback &&
        result.statistics.raw_abs_diff_p75 >= 0.0 &&
        result.statistics.raw_abs_diff_p75 <= options.local_noise_p75_max_abs_error &&
        local_noise_containment >= options.local_noise_min_containment;
    if (result.local_noise_pass) {
        result.pass = true;
        return result;
    }

    if (options.compensate_global_offset) {
        result.compensation_attempted = true;
        result.statistics = computeStatistics(warped, target, overlap_mask, true);
        const double compensated_percentile =
            selectCompensatedPercentile(result.statistics, options.percentile);
        if (compensated_percentile >= 0.0 &&
            compensated_percentile <= options.max_abs_error) {
            result.compensated_pass = true;
            result.pass = true;
            return result;
        }
        if (compensated_percentile < 0.0) {
            result.failure_message = "warp compensated height difference validation failed";
            return result;
        }
        result.failure_message =
            "warp height difference P" + std::to_string(options.percentile) +
            " above threshold: " + std::to_string(raw_percentile) + " > " +
            std::to_string(options.max_abs_error) + "; compensated P" +
            std::to_string(options.percentile) + "=" +
            std::to_string(compensated_percentile) + " > " +
            std::to_string(options.max_abs_error);
        return result;
    }

    result.failure_message =
        "warp height difference P" + std::to_string(options.percentile) +
        " above threshold: " + std::to_string(raw_percentile) + " > " +
        std::to_string(options.max_abs_error);
    return result;
}

} // namespace ir::height_difference_evaluator
