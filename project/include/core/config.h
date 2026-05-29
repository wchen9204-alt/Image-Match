#pragma once

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <string>
#include <vector>

namespace ir {

// ---------------------------------------------------------------------------
// PipelineConfig：pipeline YAML 的解析结果。
// ---------------------------------------------------------------------------
struct PipelineConfig {
    // 实验名称；如果 YAML 中没有 name，则默认使用 YAML 文件名。
    std::string name;

    // 子模块配置文件路径：分别指向 feature / matcher / geometry 的 YAML。
    std::filesystem::path              feature_path;
    std::filesystem::path              matcher_path;
    std::filesystem::path              geometry_path;

    // 匹配过滤器配置文件路径列表；会按 YAML 中 filters 的顺序依次执行。
    std::vector<std::filesystem::path> filter_paths;

    // 输入与输出路径：两张待配准图片，以及结果输出目录。
    std::filesystem::path image1_path;
    std::filesystem::path image2_path;
    std::filesystem::path output_dir;

    // 可视化与结果生成选项。
    bool draw_matches       = true;
    bool draw_inliers_only  = true;
    int  max_matches_drawn  = 100;
    bool warp               = true;
};

// ---------------------------------------------------------------------------
// Config：配置文件加载和路径解析工具。
// ---------------------------------------------------------------------------
class Config {
public:
    // 从磁盘读取 YAML 文件。
    // 读取失败或 YAML 语法错误时抛出 std::runtime_error。
    static YAML::Node load(const std::filesystem::path& path);

    // 读取一份 pipeline YAML，并解析其中引用的所有子配置路径。
    static PipelineConfig loadPipeline(const std::filesystem::path& path);

    // 将相对或绝对路径解析成可使用路径。
    static std::filesystem::path resolvePath(
        const std::filesystem::path& base_dir,
        const std::string&           relative_or_absolute);
};

}
