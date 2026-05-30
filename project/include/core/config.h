#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace ir {

/// 单个 pipeline 配置解析后的结果。
///
/// 该结构保存 pipeline YAML 里提到的各个子配置文件路径，以及输入输出
/// 路径和可视化选项，供上层 `Registration` 使用。
struct PipelineConfig {
    /// 实验或 pipeline 的名称。
    std::string name;

    /// 特征提取器、匹配器和几何估计器的配置文件路径。
    std::filesystem::path feature_path;
    std::filesystem::path matcher_path;
    std::filesystem::path geometry_path;

    /// 过滤器配置文件路径，按顺序依次执行。
    std::vector<std::filesystem::path> filter_paths;

    /// 输入图像和结果输出目录。
    std::filesystem::path image1_path;
    std::filesystem::path image2_path;
    std::filesystem::path output_dir;

    /// 可视化和输出控制选项。
    bool draw_keypoints = false;
    bool draw_matches = true;
    bool draw_inliers_only = true;
    int max_matches_drawn = 100;
    bool warp = true;
    bool show_source_window = false;
    bool show_target_window = false;
    bool show_warped_window = false;
    int wait_key = 0;
};

/// 配置文件加载与路径解析工具。
class Config {
public:
    /// 从磁盘读取一个 YAML 文件。
    ///
    /// 读取失败或 YAML 语法错误时会抛出 `std::runtime_error`。
    static YAML::Node load(const std::filesystem::path& path);

    /// 读取一个 pipeline YAML，并解析其中引用到的子配置路径。
    static PipelineConfig loadPipeline(const std::filesystem::path& path);

    /// 将相对路径解析成相对于 `base_dir` 的绝对或规范化路径。
    static std::filesystem::path resolvePath(const std::filesystem::path& base_dir,
                                             const std::string& relative_or_absolute);
};

} // namespace ir
