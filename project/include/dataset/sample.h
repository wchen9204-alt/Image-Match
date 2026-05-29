#pragma once

#include <filesystem>
#include <opencv2/core.hpp>
#include <string>

namespace ir {

/// 一个数据样本，通常包含源图、目标图以及可选真值单应矩阵。
struct Sample {
    /// 样本名，用于结果文件命名和日志输出。
    std::string           name;
    /// 源图路径。
    std::filesystem::path source_path;
    /// 目标图路径。
    std::filesystem::path target_path;

    /// 可选真值变换，通常是 source -> target 的单应矩阵。
    cv::Mat H_gt;

    /// 判断当前样本是否携带真值矩阵。
    bool has_ground_truth() const { return !H_gt.empty(); }
};

} // namespace ir
