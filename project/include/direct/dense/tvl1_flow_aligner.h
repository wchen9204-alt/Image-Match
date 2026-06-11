#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

#include "direct/dense/dense_flow_common.h"
#include "interfaces/i_direct_aligner.h"

namespace ir {

/// TV-L1 稠密光流直接法配准器。
/// 依赖 OpenCV contrib optflow 模块；计算 Dual TV-L1 光流后复用稠密光流通用后处理拟合全局几何。
class Tvl1FlowAligner : public IDirectAligner {
public:
    explicit Tvl1FlowAligner(const YAML::Node& cfg);

    std::string name() const override { return "TVL1_FLOW"; }
    bool align(RegistrationContext& ctx) override;

private:
    /// 数值格式时间步长，影响迭代稳定性。
    double _tau = 0.25;

    /// 数据项权重，值越大越强调亮度一致性。
    double _lambda = 0.15;

    /// tightness 权重，控制主变量与辅助变量的一致性。
    double _theta = 0.3;

    /// 光照变化附加项权重，0 表示不启用。
    double _gamma = 0.0;

    /// 图像金字塔层数。
    int _scalesNumber = 5;

    /// 每层 warp 次数。
    int _warpingsNumber = 5;

    /// 迭代停止阈值，值越小越精细但更慢。
    double _epsilon = 0.01;

    /// 内层迭代次数。
    int _innerIterations = 30;

    /// 外层迭代次数。
    int _outerIterations = 10;

    /// 金字塔层间缩放比例，必须小于 1。
    double _scaleStep = 0.8;

    /// 中值滤波核大小，常用 1 / 3 / 5；1 表示不滤波。
    int _medianFiltering = 5;

    /// 是否使用零光流作为 OpenCV TV-L1 的初始光流。
    bool _useInitialFlow = false;

    /// 光流输入灰度图的预平滑核大小，小于 3 时不平滑。
    int _blurKernel = 0;

    /// 稠密光流采样、过滤和全局几何拟合的通用后处理参数。
    dense_flow_common::PostprocessOptions _postprocess;
};

} // namespace ir
