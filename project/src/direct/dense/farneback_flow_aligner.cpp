#include "direct/dense/farneback_flow_aligner.h"

#include <algorithm>

#include <opencv2/video/tracking.hpp>

#include "utils/yaml_utils.h"

namespace ir {

FarnebackFlowAligner::FarnebackFlowAligner(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    // 兼容算法配置文件中直接写参数或写在 params 节点下两种形式。
    _pyrScale = yaml_utils::getDouble(params, "pyramid_scale", 0.5);
    _levels = yaml_utils::getInt(params, "levels", 3);
    _winsize = yaml_utils::getInt(params, "window_size", 15);
    _iterations = yaml_utils::getInt(params, "iterations", 3);
    _polyN = yaml_utils::getInt(params, "poly_n", 5);
    _polySigma = yaml_utils::getDouble(params, "poly_sigma", 1.2);
    _flags = yaml_utils::getInt(params, "flags", 0);
    _blurKernel = yaml_utils::getInt(params, "blur_kernel", 0);
    _postprocess = dense_flow_common::readPostprocessOptions(params);
}

bool FarnebackFlowAligner::align(RegistrationContext& ctx) {
    auto& dd = ctx.direct_data;
    auto& gd = ctx.geometry_data;
    dd.clear();
    gd.clear();
    dd.method = name();

    // 1. 读取并预处理灰度图，作为稠密光流求解输入。
    if (ctx.images.first_gray.empty() || ctx.images.second_gray.empty()) {
        dd.message = "Farneback requires non-empty grayscale images";
        gd.message = dd.message;
        return false;
    }

    const cv::Mat firstGray =
        dense_flow_common::prepareGray(ctx.images.first_gray, _blurKernel);
    const cv::Mat secondGray =
        dense_flow_common::prepareGray(ctx.images.second_gray, _blurKernel);
    if (firstGray.empty() || secondGray.empty()) {
        dd.message = "Farneback failed to prepare grayscale images";
        gd.message = dd.message;
        return false;
    }

    // 2. 在整幅图上计算源图到目标图的稠密 Farneback 光流。
    // OpenCV Farneback 直接输出每个像素从源图到目标图的二维位移。
    cv::calcOpticalFlowFarneback(firstGray,
                                 secondGray,
                                 dd.flow,
                                 _pyrScale,
                                 std::max(1, _levels),
                                 std::max(3, _winsize),
                                 std::max(1, _iterations),
                                 std::max(5, _polyN),
                                 _polySigma,
                                 _flags);

    // 3. 若开启前后向一致性检查，则额外估计 target->source 光流作为后处理输入。
    cv::Mat backwardFlow;
    if (_postprocess.forward_backward_check) {
        // 前后向一致性检查用于剔除局部光流不自洽的采样点。
        cv::calcOpticalFlowFarneback(secondGray,
                                     firstGray,
                                     backwardFlow,
                                     _pyrScale,
                                     std::max(1, _levels),
                                     std::max(3, _winsize),
                                     std::max(1, _iterations),
                                     std::max(5, _polyN),
                                     _polySigma,
                                     _flags);
    }

    // 4. 复用共享后处理：采样、过滤、全局几何拟合，并把结果写回统一输出结构。
    return dense_flow_common::finalizeFlowAlignment(
        ctx, firstGray, backwardFlow, _postprocess, name());
}

} // namespace ir
