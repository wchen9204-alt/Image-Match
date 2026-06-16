#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

#include "direct/dense/dense_flow_common.h"
#include "interfaces/i_direct_aligner.h"

namespace ir {

/// Farneback 稠密光流直接法配准器。
/// 先估计整幅图的稠密光流，再按固定步长采样点对以拟合全局几何变换。
class FarnebackFlowAligner : public IDirectAligner {
public:
    explicit FarnebackFlowAligner(const YAML::Node& cfg);

    std::string name() const override { return "FARNEBACK_FLOW"; }
    bool align(RegistrationContext& ctx) override;

private:
    /// Farneback 金字塔缩放比例。
    double _pyrScale = 0.5;

    /// Farneback 金字塔层数。
    int _levels = 3;

    /// Farneback 窗口大小。
    int _winsize = 15;

    /// 每层金字塔的迭代次数。
    int _iterations = 3;

    /// 多项式展开邻域大小。
    int _polyN = 5;

    /// 多项式展开高斯标准差。
    double _polySigma = 1.2;

    /// 传给 OpenCV Farneback 的 flags。
    int _flags = 0;

    /// 光流输入灰度图的预平滑核大小，小于 3 时不平滑。
    int _blurKernel = 0;

    /// 稠密光流采样、过滤和全局几何拟合的通用后处理参数。
    dense_flow_common::PostprocessOptions _postprocess;
};

} // namespace ir

