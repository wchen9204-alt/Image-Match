#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

#include "interfaces/i_direct_aligner.h"

namespace ir {

/// 灰度图相位相关直接法配准器，估计全局平移。
/// 该方法只输出平移矩阵，不估计旋转或缩放，适合位移主导且两图尺寸一致的场景。
class DirectPhaseCorrelationAligner : public IDirectAligner {
public:
    explicit DirectPhaseCorrelationAligner(const YAML::Node& cfg);

    std::string name() const override { return "DIRECT_PHASE_CORRELATION"; }
    bool align(RegistrationContext& ctx) override;

private:
    /// 相位相关响应阈值，低于该值时认为平移估计不可靠。
    double _responseThreshold = 0.01;

    /// 相位相关前的高斯模糊核大小，小于 3 时不模糊，偶数会在实现中修正为奇数。
    int _blurKernel = 5;

    /// 是否使用 Hann 窗口抑制边界效应。
    bool _useHannWindow = true;

    /// 是否启用加权相位相关；当前版本主要提供基于梯度幅值的空间权重。
    bool _weighted = false;

    /// 加权模式；可选 NONE / GRADIENT / SOURCE_GRADIENT / TARGET_GRADIENT。
    std::string _weightMode = "GRADIENT";

    /// 权重图平滑核大小，小于 3 时不平滑，偶数会在实现中修正为奇数。
    int _weightBlurKernel = 0;

    /// 权重图幂指数，值越大越强调高纹理区域。
    double _weightPower = 1.0;

    /// 权重图最小保底值，避免低纹理区域被完全压成 0。
    double _weightFloor = 0.05;

    /// 是否启用亚像素峰值置信度检查。
    bool _confidenceCheck = false;

    /// 主峰与次峰比阈值，低于该值时认为峰值歧义较大。
    double _peakRatioThreshold = 0.0;

    /// 峰值局部尖锐度阈值，使用主峰与 8 邻域平均绝对值之比衡量。
    double _subpixelConfidenceThreshold = 0.0;

    /// 搜索次峰时在主峰周围排除的半径，单位为像素。
    int _peakExclusionRadius = 5;
};

} // namespace ir
