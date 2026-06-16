#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

#include "interfaces/i_direct_aligner.h"

namespace ir {

/// Fourier-Mellin 频域直接法配准器，估计源图到目标图的相似变换。
/// 该方法先在频谱幅值的 log-polar 空间估计旋转和尺度，再用相位相关估计平移。
class DirectFourierMellinAligner : public IDirectAligner {
public:
    explicit DirectFourierMellinAligner(const YAML::Node& cfg);

    std::string name() const override { return "DIRECT_FOURIER_MELLIN"; }
    bool align(RegistrationContext& ctx) override;

private:
    /// 输入灰度图预平滑核大小；小于 3 时不平滑，偶数会在实现中修正为奇数。
    int _blurKernel = 3;

    /// 是否使用 Hann 窗抑制频域和相位相关中的边界效应。
    bool _windowed = true;

    /// 是否使用多尺度候选生成；开启后会在多个金字塔层估计旋转/尺度候选。
    bool _usePyramid = false;

    /// 金字塔层数；包含原图层，最小为 1。
    int _pyramidLevels = 3;

    /// 每层相对上一层的缩放比例；默认 0.5。
    double _pyramidScale = 0.5;

    /// log-polar 图宽；小于等于 0 时使用当前估计层宽度。
    int _logPolarCols = 0;

    /// log-polar 图高；小于等于 0 时使用当前估计层高度。
    int _logPolarRows = 0;

    /// 频谱幅值图高斯平滑核大小；小于 3 时不平滑。
    int _magnitudeBlurKernel = 3;

    /// 抑制频谱中心 DC 分量的半径；小于等于 0 时不抑制。
    int _dcSuppressRadius = 3;

    /// 允许的最小尺度；用于过滤 log-polar 估计得到的异常候选。
    double _minScale = 0.25;

    /// 允许的最大尺度；用于过滤 log-polar 估计得到的异常候选。
    double _maxScale = 4.0;

    /// 旋转/尺度相位相关响应阈值；低于该值时拒绝对应候选层。
    double _rotationScaleResponseThreshold = 0.0;

    /// 平移相位相关响应阈值；低于该值时认为最终相似变换不可靠。
    double _translationResponseThreshold = 0.01;

    /// 是否额外尝试 180 度歧义候选；频谱幅值存在中心对称性，默认开启更稳健。
    bool _tryHalfTurnAmbiguity = true;
};

} // namespace ir

