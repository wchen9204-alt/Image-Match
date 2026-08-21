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

/// 结构法匹配图使用的对应关系来源。
enum class StructureMatchSource {
    RAW,
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

/// 直接法消费点特征初值的策略。
enum class FeatureInitializerSeedMode {
    /// 仅使用通过完整初始化质量门控的点特征结果；未通过时直接法从原图开始。
    ACCEPTED_ONLY,
    /// 只要几何估计产出合法变换就作为初值；质量门控失败时仍不能作为最终结果。
    ESTIMATED_WHEN_AVAILABLE
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
        /// 生成前景 mask 时使用的灰度阈值。
        int foreground_threshold = 20;
        /// 局部包含率的匹配容差，单位为像素；0 表示严格像素重叠。
        int containment_tolerance_pixels = 0;
    };

    /// warp 高度差验证参数，使用有效重叠前景内绝对差的指定分位数。
    struct HeightDifferenceValidationConfig {
        /// 是否启用 warped source 与 target 重叠区域的高度差 P90 验证。
        bool enabled = false;
        /// 是否先估计并补偿全局高度偏移；默认关闭。
        bool compensate_global_height_offset = true;
        /// 用于成功判定的分位数；支持 50、75、90、95，默认 90。
        int percentile = 90;
        /// 所选分位数允许的最大绝对高度差；当前单位为归一化灰度，默认 0.10。
        double max_abs_error = 0.10;
        /// 是否启用局部噪声尾部例外判定。
        bool local_noise_fallback_enabled = false;
        /// 局部噪声例外要求原始 P75 不超过该阈值。
        double local_noise_p75_max_abs_error = 0.10;
        /// 局部噪声例外要求较小前景的包含率不低于该值。
        double local_noise_min_containment = 0.90;
    };

    /// 基于最终变换矩阵的长条线组结构验证；明确冲突会参与最终成功判定。
    struct EdgeStructureDiagnosticConfig {
        bool enabled = false;
        int visibility_threshold = 0;
        double min_foreground_elongation_ratio = 4.00;
        /// 全部前景沿 PCA 主轴的最低连续占用率。
        double min_axis_occupancy = 0.75;
        /// 主轴截面中心相对 PCA 中心线的 P90 偏差 / 长边上限。
        double max_centerline_deviation_ratio = 0.03;
        int max_canvas_side_pixels = 16384;
        int max_canvas_pixels = 100000000;

        /// 同一物理边缘重复线组的法向去重容差；较远的两条边界仍分别保留。
        double duplicate_line_normal_tolerance_pixels = 2.0;
        /// 重复线组去重要求其切向包络区间的最小重叠比例。
        double duplicate_line_min_span_overlap_ratio = 0.80;
        /// 同一粗长边双侧边界的最小/最大法向间距，单位为像素。
        double outer_longitudinal_edge_min_normal_separation_pixels = 1.0;
        double outer_longitudinal_edge_max_normal_separation_pixels = 12.0;
        /// 判定为同一粗长边的双侧边界时，切向包络最小重叠比例。
        double outer_longitudinal_edge_min_span_overlap_ratio = 0.75;
        /// 初始片段固定由 EDLines 在共同可见区域的灰度图上检测。
        double min_fragment_length_pixels = 6.0;

        double group_max_angle_difference_degrees = 12.0;
        double group_max_normal_distance_pixels = 3.0;
        /// 首次拟合后，切向不重叠的同轴线组允许重新分配的法向距离。
        double post_fit_group_normal_distance_pixels = 3.0;
        double min_line_group_actual_length_pixels = 6.0;
        double min_line_group_continuity_ratio = 0.10;
        double max_line_group_gap_ratio = 0.45;
        double max_fragment_direction_spread_degrees = 10.0;
        double max_line_fit_residual_pixels = 5.0;

        double min_main_line_actual_length_ratio = 0.30;
        double direction_cluster_tolerance_degrees = 6.0;
        double min_main_direction_support_ratio = 0.60;
        double max_main_direction_spread_degrees = 4.0;
        double min_main_direction_margin = 0.15;
        double max_main_direction_difference_degrees = 5.0;
        double max_axis_classification_error_degrees = 8.0;
        // 按各自完整前景在参考主轴/法线方向的跨度过滤短线组。
        double min_horizontal_actual_length_ratio = 0.65;
        double min_vertical_actual_length_ratio = 0.45;
        /// 竖直方向双方未解释支撑均超过此比例时，判为系统性结构冲突。
        double max_vertical_unmatched_length_ratio = 0.70;

        double profile_smoothing_sigma = 1.0;
        double min_peak_prominence = 0.03;
        double candidate_position_tolerance_pixels = 15.0;
        double final_position_tolerance_pixels = 3.0;
        double candidate_min_span_overlap_ratio = 0.05;
        double min_shorter_line_overlap_ratio = 0.90;
        double max_line_pair_angle_difference_degrees = 0.3;
        double match_position_cost_weight = 0.45;
        double match_overlap_cost_weight = 0.30;
        double match_angle_cost_weight = 0.20;
        double match_prominence_cost_weight = 0.05;
        double min_strong_line_actual_length_pixels = 12.0;
        double min_strong_peak_prominence = 0.05;
        double ambiguity_score_margin = 0.05;
    };

    /// warp 质量验证参数集合，供最终验证和初始化候选验证复用。
    struct WarpQualityValidationConfig {
        WarpOverlapValidationConfig overlap;
        HeightDifferenceValidationConfig height_difference;
        EdgeStructureDiagnosticConfig edge_structure_diagnostic;
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
        /// 初值消费策略；默认仅使用通过完整质量门控的结果。
        FeatureInitializerSeedMode seed_mode = FeatureInitializerSeedMode::ACCEPTED_ONLY;
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

    /// 图片输出总开关；false 时跳过所有 PNG 编码和写盘。
    bool save_visuals = true;
    bool draw_keypoints = false;
    bool draw_matches = true;
    std::vector<MatchView> match_views = {
        MatchView::RAW,
        MatchView::FILTERED,
        MatchView::INLIERS
    };
    int max_matches_drawn = 100;
    bool save_originals = true;
    bool save_warped = true;
    bool save_blend = true;
    bool save_false_color_overlay = true;
    /// false 图生成前景 mask 时使用的灰度阈值；仅影响可视化，不参与配准结果判定。
    int false_color_foreground_threshold = 0;
    /// 是否保存 source/target 原图前景 mask，以及 warped source/target 前景重叠图。
    bool save_foreground_masks = true;
    bool save_structure_responses = true;
    StructureMatchSource structure_match_source = StructureMatchSource::RAW;
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

    /// 重置 batch/compare 使用的可视化字段，避免继承单算法 pipeline 的输出偏好。
    static void resetVisualization(PipelineConfig& cfg);

    /// 将 visualization YAML 中出现的字段覆盖到 pipeline 配置。
    static void applyVisualizationOverrides(PipelineConfig& cfg,
                                            const YAML::Node& visualization);

    /// 将相对路径解析成相对于 `base_dir` 的规范化路径。
    static std::filesystem::path resolvePath(const std::filesystem::path& base_dir,
                                             const std::string& relative_or_absolute);
};

} // namespace ir
