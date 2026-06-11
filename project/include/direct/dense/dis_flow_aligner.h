#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

#include "direct/dense/dense_flow_common.h"
#include "interfaces/i_direct_aligner.h"

namespace ir {

/// DIS 稠密光流直接法配准器。
/// 先用 OpenCV Dense Inverse Search 估计稠密光流，再复用稠密光流通用后处理拟合全局几何。
class DisFlowAligner : public IDirectAligner {
public:
    explicit DisFlowAligner(const YAML::Node& cfg);

    std::string name() const override { return "DIS_FLOW"; }
    bool align(RegistrationContext& ctx) override;

private:
    /// OpenCV DIS preset，可选 ULTRAFAST / FAST / MEDIUM。
    int _preset = 1;

    /// DIS 最细金字塔层；小于 0 时保留 preset 默认值。
    int _finestScale = -1;

    /// DIS patch 大小；小于等于 0 时保留 preset 默认值。
    int _patchSize = -1;

    /// 相邻 patch 的步长；小于等于 0 时保留 preset 默认值。
    int _patchStride = -1;

    /// patch inverse search 阶段梯度下降迭代次数；小于 0 时保留 preset 默认值。
    int _gradientDescentIterations = -1;

    /// 变分细化迭代次数；小于 0 时保留 preset 默认值。
    int _variationalRefinementIterations = -1;

    /// 变分细化 smoothness 权重；小于 0 时保留 preset 默认值。
    float _variationalRefinementAlpha = -1.0f;

    /// 变分细化 color constancy 权重；小于 0 时保留 preset 默认值。
    float _variationalRefinementDelta = -1.0f;

    /// 变分细化 gradient constancy 权重；小于 0 时保留 preset 默认值。
    float _variationalRefinementGamma = -1.0f;

    /// 变分细化鲁棒惩罚 epsilon；小于 0 时保留 preset 默认值。
    float _variationalRefinementEpsilon = -1.0f;

    /// 是否对 patch 做均值归一化，增强亮度变化鲁棒性。
    bool _useMeanNormalization = true;

    /// 是否启用空间传播，通常能提升大位移区域的光流连贯性。
    bool _useSpatialPropagation = true;

    /// 光流输入灰度图的预平滑核大小，小于 3 时不平滑。
    int _blurKernel = 0;

    /// 稠密光流采样、过滤和全局几何拟合的通用后处理参数。
    dense_flow_common::PostprocessOptions _postprocess;
};

} // namespace ir
