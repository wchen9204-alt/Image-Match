#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>

#include "core/context.h"

namespace ir {

/// 前景边界角点增强使用的角点检测方法。
enum class BoundaryCornerMethod {
    /// Shi-Tomasi，适合作为默认边界补点方法。
    SHI_TOMASI,
    /// Harris，更强调规则拐角响应。
    HARRIS,
};

/// 前景边界角点增强配置。
struct BoundaryCornerAugmentationConfig {
    /// 是否启用前景边界角点增强。
    bool enabled = false;
    /// 边界角点检测方法；可选 `shi_tomasi` / `harris`。
    BoundaryCornerMethod corner_method = BoundaryCornerMethod::SHI_TOMASI;
    /// 最多补充的边界角点数量。
    int max_corners = 40;
    /// 角点质量阈值。
    double quality_level = 0.01;
    /// 新增角点与已有关键点的最小间距。
    double min_distance = 8.0;
    /// 角点检测使用的邻域大小。
    int block_size = 3;
    /// Harris 角点参数。
    double harris_k = 0.04;
    /// 前景边界带宽，单位为像素。
    int boundary_band = 6;
    /// 前景二值化阈值。
    int foreground_threshold = 10;
    /// 新增角点写回 `cv::KeyPoint` 时的 size。
    float keypoint_size = 12.0f;
    /// 关键点总数上限；小于等于 0 表示不限制。
    int max_total_keypoints = -1;
};

/// 从 YAML 读取前景边界角点增强配置。
BoundaryCornerAugmentationConfig loadBoundaryCornerAugmentationConfig(const YAML::Node& cfg);

/// 初始化点特征上下文，并准备灰度图输入。
bool prepareKeypointExtractionContext(RegistrationContext& ctx,
                                      KeypointType type,
                                      NormType norm,
                                      const std::string& extractor_name);

/// 在前景边界附近补充角点，并合并到现有关键点集合。
int augmentKeypointsWithBoundaryCorners(const cv::Mat& gray,
                                        std::vector<cv::KeyPoint>& keypoints,
                                        const BoundaryCornerAugmentationConfig& config);

/// 对支持外部补点的提取器执行统一的 detect / augment / compute 流程。
bool extractKeypointsWithBoundaryAugmentation(RegistrationContext& ctx,
                                              KeypointType type,
                                              NormType norm,
                                              const std::string& extractor_name,
                                              cv::Feature2D& extractor,
                                              const BoundaryCornerAugmentationConfig& config);

} // namespace ir
