#pragma once

#include <opencv2/core.hpp>

#include <string>

#include "core/types.h"

namespace ir {

/// 直接法前置点特征初始化阶段的结果。
/// 这里只记录“点特征粗估是否可作为直接法初值，以及它自身的质量统计”，
/// 不直接代表最终配准结果。
struct FeatureInitializerData {
    /// 本轮是否尝试过点特征初始化。
    bool attempted = false;

    /// 是否通过初始化阶段自己的接受条件并产出可用初值。
    bool accepted = false;

    /// 被接受的点特征方法名称。
    std::string method;

    /// 初始化阶段状态说明；失败时记录拒绝原因。
    std::string message;

    /// 被接受初值对应的几何模型类型。
    GeometryType type = GeometryType::UNKNOWN;

    /// source -> target 的 2x3 变换矩阵。
    cv::Mat A;

    /// source -> target 的 3x3 单应矩阵。
    cv::Mat H;

    /// 初始化阶段统计。
    int num_keypoints_first = 0;
    int num_keypoints_second = 0;
    int num_raw_matches = 0;
    int num_filtered_matches = 0;
    int num_inliers = 0;

    /// 初始化阶段质量指标。
    double inlier_ratio = -1.0;
    double inlier_spatial_coverage = -1.0;
    double warp_overlap_containment = -1.0;
    double warp_source_coverage = -1.0;
    double warp_target_coverage = -1.0;
    double warp_bidirectional_coverage = -1.0;
    double warp_edge_alignment_iou = -1.0;
    double warp_photometric_error = -1.0;

    /// 清空初始化阶段缓存。
    void clear() {
        attempted = false;
        accepted = false;
        method.clear();
        message.clear();
        type = GeometryType::UNKNOWN;
        A.release();
        H.release();
        num_keypoints_first = 0;
        num_keypoints_second = 0;
        num_raw_matches = 0;
        num_filtered_matches = 0;
        num_inliers = 0;
        inlier_ratio = -1.0;
        inlier_spatial_coverage = -1.0;
        warp_overlap_containment = -1.0;
        warp_source_coverage = -1.0;
        warp_target_coverage = -1.0;
        warp_bidirectional_coverage = -1.0;
        warp_edge_alignment_iou = -1.0;
        warp_photometric_error = -1.0;
    }
};

} // namespace ir
