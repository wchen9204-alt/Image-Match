#pragma once

#include <string>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "core/context.h"
#include "direct/common/direct_geometry_common.h"

namespace ir {
namespace dense_flow_common {

using RobustFitOptions = direct_geometry_common::RobustFitOptions;

/// 稠密光流后处理参数，Farneback / DIS / TV-L1 等方法可共享。
struct PostprocessOptions {
    /// 源图梯度幅值采样阈值，小于等于 0 时不按纹理过滤。
    double gradient_threshold = 0.0;

    /// 是否从稠密光流采样点拟合一个可用于全局 warp 的变换矩阵。
    bool fit_global_transform = true;

    /// 全局拟合模型，可选 RIGID / SIMILARITY / AFFINE / HOMOGRAPHY。
    std::string fit_model = "RIGID";

    /// 稠密光流采样步长，步长越小点对越多、计算越重。
    int sample_step = 8;

    /// 保留采样点的最小光流幅值，小于 0 时不启用下限过滤。
    double min_flow_magnitude = -1.0;

    /// 保留采样点的最大光流幅值，小于 0 时不启用上限过滤。
    double max_flow_magnitude = -1.0;

    /// 采样点距图像边界的最小安全边距，单位为像素。
    int border_margin = 0;

    /// 是否启用前后向光流一致性检查。
    bool forward_backward_check = false;

    /// 前后向光流回投误差阈值，单位为像素。
    double fb_threshold = 1.5;

    /// 全局几何鲁棒拟合参数。
    RobustFitOptions robust;

    /// 全局变换接受所需的最少内点数。
    int min_inliers = 20;
};

/// 从 YAML 参数节点读取稠密光流通用后处理配置。
PostprocessOptions readPostprocessOptions(const YAML::Node& params);

/// 为稠密光流准备灰度输入；当前只做可选预平滑，保持原图尺度不变。
cv::Mat prepareGray(const cv::Mat& gray, int blurKernel);

/// 为稠密光流直接法统一准备输入灰度图，并检查尺寸合法性。
bool prepareFlowInputs(RegistrationContext& ctx,
                       const std::string& methodLabel,
                       int blurKernel,
                       cv::Mat& firstGray,
                       cv::Mat& secondGray);

/// 将稠密光流转换为采样点对并拟合全局几何，同时写回 direct_data 与 geometry_data。
bool finalizeFlowAlignment(RegistrationContext& ctx,
                           const cv::Mat& firstGray,
                           const cv::Mat& backwardFlow,
                           const PostprocessOptions& options,
                           const std::string& methodLabel);

} // namespace dense_flow_common
} // namespace ir
