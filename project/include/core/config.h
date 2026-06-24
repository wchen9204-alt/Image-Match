#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace ir {

/// 配准方法族，用于区分点特征法、结构法、直接法等的输出和摘要组织。
enum class MethodFamily {
    KEYPOINT,
    STRUCTURE,
    DIRECT,
    LEARNING
};

/// 匹配可视化的数据层级，用于区分过滤前、过滤后和几何内点。
enum class MatchView {
    RAW,
    FILTERED,
    INLIERS
};

/// 直接法最终成功判定时，如何参考点特征初始化结果。
enum class DirectValidationReferenceMode {
    /// 只使用直接法最终输出的 warp 质量作为成功判定依据。
    DIRECT_ONLY,
    /// 同时保留 direct 最终结果和已接受的点特征初值：
    /// 1. 两者都失败则失败；
    /// 2. 只有一方成功则直接采用成功的一方；
    /// 3. 两者都成功则比较双方质量，选择更优结果作为最终判定依据。
    BEST_OF_DIRECT_AND_INITIALIZER
};

/// 将 MethodFamily 枚举转为输出目录名。
inline const char* methodFamilyDir(MethodFamily f) {
    switch (f) {
    case MethodFamily::KEYPOINT:  return "keypoint";
    case MethodFamily::STRUCTURE: return "structure";
    case MethodFamily::DIRECT:    return "direct";
    case MethodFamily::LEARNING:  return "learning";
    default:                      return "unknown";
    }
}

/// 将 MethodFamily 枚举转为人类可读标签。
inline const char* methodFamilyLabel(MethodFamily f) {
    switch (f) {
    case MethodFamily::KEYPOINT:  return "KeypointPipeline";
    case MethodFamily::STRUCTURE: return "StructurePipeline";
    case MethodFamily::DIRECT:    return "DirectPipeline";
    case MethodFamily::LEARNING:  return "LearningPipeline";
    default:                      return "UnknownPipeline";
    }
}

/// 单个 pipeline 配置解析后的结果。
struct PipelineConfig {
    /// 直接法点特征初始化的候选方法配置。
    struct FeatureInitializerCandidateConfig {
        /// 候选方法显示名，通常为 SIFT / SURF / ORB / BRISK / KAZE / AKAZE。
        std::string name;

        /// 候选方法对应的 keypoint YAML 配置路径。
        std::filesystem::path keypoint_path;
    };

    /// warp overlap 验证参数，负责前景覆盖和局部包含关系。
    struct WarpOverlapValidationConfig {
        /// 是否启用 warped source 与 target 的前景局部包含率验证。
        bool containment_enabled = false;
        /// 局部包含率最低阈值；通常用于“一张图是另一张图局部”的场景。
        double min_containment = 0.20;
        /// 是否启用 warped source / target 的双向前景 coverage 验证。
        bool bidirectional_coverage_enabled = false;
        /// 双向 coverage 最低阈值；实际比较时取 source / target coverage 的较大值。
        double min_bidirectional_coverage = -1.0;
        /// containment 与 bidirectional coverage 同时启用时，是否允许“任一达标即可通过”。
        bool accept_if_either_passes = false;
        /// 生成前景 mask 时使用的灰度阈值。
        int foreground_threshold = 10;
    };

    /// warp 光度验证参数，负责重叠区域 NMAD 判定。
    struct PhotometricValidationConfig {
        /// 是否启用 warped source 与 target 重叠区域的光度误差验证。
        bool enabled = false;
        /// 常规光度误差验证的最大 NMAD 阈值。
        double max_nmad = 0.15;
        /// 仅靠 coverage 放行、但 containment 未达标时使用的更严格 NMAD 阈值；小于 0 表示禁用。
        double max_nmad_for_coverage_only = -1.0;
    };

    /// warp 边缘对齐验证参数，负责拦截“覆盖了但内容没对上”的误判。
    struct EdgeAlignmentValidationConfig {
        /// 是否启用 warped source 与 target 重叠区域的边缘对齐 IoU 验证。
        bool enabled = false;
        /// 边缘对齐 IoU 最低阈值。
        double min_iou = 0.08;
        /// Canny 低阈值。
        int canny_low_threshold = 50;
        /// Canny 高阈值。
        int canny_high_threshold = 150;
        /// 边缘 mask 膨胀核尺寸，用于容忍少量像素级偏移；小于等于 1 表示不膨胀。
        int dilate_size = 3;
        /// 参与验证的最低边缘像素数；低纹理图像低于该值时判为无效。
        int min_edge_pixels = 20;
    };

    /// warp 质量验证参数集合，供最终验证和初始化候选验证复用。
    struct WarpQualityValidationConfig {
        WarpOverlapValidationConfig overlap;
        PhotometricValidationConfig photometric;
        EdgeAlignmentValidationConfig edge_alignment;
    };

    /// 点特征初始化候选接受条件。
    struct FeatureInitializerAcceptanceConfig {
        /// 点特征初始化几何门控的最低内点数。
        int min_inliers = 6;
        /// 点特征初始化几何门控的最低内点率；小于 0 表示禁用。
        double min_inlier_ratio = -1.0;
        /// 点特征初始化几何门控的最低内点空间覆盖率；小于 0 表示禁用。
        double min_inlier_spatial_coverage = 0.05;
    };

    /// 直接法前置点特征初始化配置。
    struct FeatureInitializerConfig {
        /// 是否启用直接法前置点特征初始值估计。
        bool enabled = false;
        /// 直接法最终成功判定时，是否允许回看已接受的点特征初值。
        DirectValidationReferenceMode final_validation_reference =
            DirectValidationReferenceMode::DIRECT_ONLY;
        /// 直接法前置点特征初始化使用的单个点特征候选。
        FeatureInitializerCandidateConfig candidate;
        /// 点特征初始化使用的 matcher YAML。
        std::filesystem::path matcher_path;
        /// 点特征初始化使用的 filter YAML 列表。
        std::vector<std::filesystem::path> filter_paths;
        /// 点特征初始化使用的 geometry YAML。
        std::filesystem::path geometry_path;
        /// 点特征初始化几何接受条件。
        FeatureInitializerAcceptanceConfig acceptance;
        /// 点特征初始矩阵的临时 warp 质量验证参数。
        WarpQualityValidationConfig validation;
    };

    std::string name;

    /// 显式声明的方法族，由 YAML 中 method_family 字段指定。
    MethodFamily method_family = MethodFamily::KEYPOINT;

    /// 各方法族的配置文件路径（按需填写）；direct_path 指向具体直接法算法配置。
    std::filesystem::path keypoint_path;
    std::filesystem::path structure_path;
    std::filesystem::path direct_path;
    std::filesystem::path learning_path;
    std::filesystem::path matcher_path;
    std::filesystem::path geometry_path;

    std::vector<std::filesystem::path> filter_paths;
    std::filesystem::path evaluator_path;

    std::filesystem::path image1_path;
    std::filesystem::path image2_path;
    std::filesystem::path output_dir;

    bool draw_keypoints = false;
    bool draw_matches = true;
    bool draw_inliers_only = false;
    std::vector<MatchView> match_views = {
        MatchView::RAW,
        MatchView::FILTERED,
        MatchView::INLIERS
    };
    int max_matches_drawn = 100;
    bool warp = true;
    bool show_source_window = false;
    bool show_target_window = false;
    bool show_warped_window = false;
    int wait_key = 0;

    /// 最终配准结果的 warp 质量验证配置。
    WarpQualityValidationConfig warp_quality;

    /// 是否启用方法特征质量验证，例如点特征数量、结构数量和结构候选匹配数量。
    bool validate_method_quality = false;
    /// 方法特征质量不满足阈值时，是否直接判定样例失败。
    bool fail_on_method_quality = true;
    /// 点特征法/学习法要求两幅图至少检测到的特征点数量；小于等于 0 表示禁用。
    int min_method_keypoints = 0;
    /// 结构法要求两幅图至少提取到的结构元素数量；小于等于 0 表示禁用。
    int min_method_structures = 0;
    /// 结构法要求至少产生的候选结构匹配数量；小于等于 0 表示禁用。
    int min_method_structure_matches = 0;

    /// 是否启用匹配质量验证。
    bool validate_match_quality = false;
    /// 匹配质量不满足阈值时，是否直接判定样例失败；否则仅输出告警。
    bool fail_on_match_quality = false;
    /// 最少内点数阈值；小于等于 0 表示禁用该条件。
    int min_match_inliers = 0;
    /// 最低内点率阈值；小于 0 表示禁用该条件。
    double min_match_inlier_ratio = -1.0;
    /// 最大重投影误差阈值；小于 0 表示禁用该条件。
    double max_match_reproj_error = -1.0;
    /// 最终内点在 source / target 前景包围盒中的最低空间覆盖率；小于 0 表示禁用。
    double min_inlier_spatial_coverage = -1.0;

    /// 是否启用直接法专属质量验证，例如 ECC/相位相关的 confidence 阈值。
    bool validate_direct_quality = false;
    /// 直接法专属质量不满足阈值时，是否直接判定样例失败。
    bool fail_on_direct_quality = true;
    /// 直接法最小 confidence 阈值；小于 0 表示禁用该条件。
    double min_direct_confidence = -1.0;

    /// 是否启用结构响应图 warp 后的前景重叠验证。
    bool validate_structure_overlap = false;
    /// 结构重叠验证所需的最小 IoU 阈值。
    double min_structure_overlap_iou = 0.20;
    /// 结构响应图生成前景 mask 时使用的阈值。
    int structure_overlap_foreground_threshold = 0;
    /// 结构重叠验证前可选的膨胀核尺寸，用于增强细线结构的重叠稳定性。
    int structure_overlap_dilate_size = 3;

    /// 直接法前置点特征初始化配置。
    FeatureInitializerConfig feature_initializer;

    MethodFamily methodFamily() const { return method_family; }
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

