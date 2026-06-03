#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace ir {

/// 配准方法族，用于区分点特征法与结构法的输出和摘要组织。
enum class MethodFamily {
    KEYPOINT,
    STRUCTURE
};

/// 单个 pipeline 配置解析后的结果。
struct PipelineConfig {
    /// 实验或 pipeline 的名称。
    std::string name;

    /// 点特征、结构特征、匹配器与几何估计器的配置文件路径。
    std::filesystem::path keypoint_path;
    std::filesystem::path structure_path;
    std::filesystem::path matcher_path;
    std::filesystem::path geometry_path;

    /// 过滤器配置文件路径，按顺序依次执行。
    std::vector<std::filesystem::path> filter_paths;

    /// 输入图像与输出目录。
    std::filesystem::path image1_path;
    std::filesystem::path image2_path;
    std::filesystem::path output_dir;

    /// 可视化与输出控制选项。
    bool draw_keypoints = false;
    bool draw_matches = true;
    bool draw_inliers_only = false;
    int max_matches_drawn = 100;
    bool warp = true;
    bool show_source_window = false;
    bool show_target_window = false;
    bool show_warped_window = false;
    int wait_key = 0;

    /// 结果有效性校验：根据 warped source 与 target 的前景 IoU 判断是否真正重合。
    bool validate_warp_overlap = false;
    double min_warp_overlap_iou = 0.20;
    int warp_overlap_foreground_threshold = 10;

    /// 根据是否配置结构提取器判断当前 pipeline 所属的方法族。
    MethodFamily methodFamily() const {
        return structure_path.empty() ? MethodFamily::KEYPOINT : MethodFamily::STRUCTURE;
    }
};

/// 配置文件加载与路径解析工具。
class Config {
public:
    /// 从磁盘读取一个 YAML 文件。
    static YAML::Node load(const std::filesystem::path& path);

    /// 读取一个 pipeline YAML，并解析其中引用到的子配置路径。
    static PipelineConfig loadPipeline(const std::filesystem::path& path);

    /// 将相对路径解析成相对于 `base_dir` 的规范化路径。
    static std::filesystem::path resolvePath(const std::filesystem::path& base_dir,
                                             const std::string& relative_or_absolute);
};

} // namespace ir
