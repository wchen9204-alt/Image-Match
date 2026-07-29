#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "core/types.h"

namespace ir {

/// 几何估计阶段的输出数据。
struct GeometryData {
    /// 当前几何模型类型。
    GeometryType type = GeometryType::UNKNOWN;

    /// 当前配准流程可能产出的几何矩阵结果。
    cv::Mat H;
    cv::Mat A;

    /// 几何估计是否成功。
    bool valid = false;
    /// 内点数量。
    int num_inliers = 0;
    /// 内点比例。
    double inlier_ratio = 0.0;

    /// 进入额外候选前的 baseline 是否已达到最少内点要求。
    bool baseline_valid = false;
    /// 进入额外候选前的 baseline 内点数；未得到 baseline 时为 0。
    int baseline_num_inliers = 0;
    /// 进入额外候选前的 baseline 平均重投影误差；无可用内点时为 -1。
    double baseline_mean_reproj_error = -1.0;
    /// 本次是否进入额外 rigid 候选生成与评分流程。
    bool candidate_fallback_attempted = false;
    /// 候选流程状态；总开关与前置条件满足时为 enabled，否则为 not_available。
    std::string candidate_fallback_trigger_reason = "not_available";

    /// 几何估计对应点的内点掩码，与本次估计读取的 CorrespondenceView.filtered 对齐。
    std::vector<unsigned char> inlier_mask;

    /// 本次几何估计实际读取的对应点来源类型名称，通常为 KEYPOINT / STRUCTURE / DIRECT / LEARNING。
    std::string correspondence_source;

    /// 本次几何估计实际读取的候选对应点数量。
    int num_correspondences = 0;

    /// 几何估计阶段的详细状态消息。
    std::string message;

    /// 清空几何估计结果。
    void clear() {
        type = GeometryType::UNKNOWN;
        H.release();
        A.release();
        valid = false;
        num_inliers = 0;
        inlier_ratio = 0.0;
        baseline_valid = false;
        baseline_num_inliers = 0;
        baseline_mean_reproj_error = -1.0;
        candidate_fallback_attempted = false;
        candidate_fallback_trigger_reason = "not_triggered";
        inlier_mask.clear();
        correspondence_source.clear();
        num_correspondences = 0;
        message.clear();
    }
};

} // namespace ir

