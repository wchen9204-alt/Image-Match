#pragma once

#include <filesystem>
#include <opencv2/core.hpp>
#include <string>

namespace ir {

// ---------------------------------------------------------------------------
// Sample：一组源图、目标图及可选真值单应矩阵。
// ---------------------------------------------------------------------------
struct Sample {
    std::string                name;          // 唯一样本标识，例如 "test1"。
    std::filesystem::path      source_path;
    std::filesystem::path      target_path;

    // 可选真值变换：source -> target；数据集未提供时为空。
    cv::Mat                    H_gt;

    bool has_ground_truth() const { return !H_gt.empty(); }
};

} // namespace ir
