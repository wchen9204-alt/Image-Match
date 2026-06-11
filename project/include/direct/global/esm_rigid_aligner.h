#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

#include "interfaces/i_direct_aligner.h"

namespace ir {

/// ESM 刚体全局直接法配准器。
/// 该方法复用同步下采样的刚体金字塔直接法框架，在每层使用源图与目标图梯度的对称平均构造 Jacobian。
class EsmRigidAligner : public IDirectAligner {
public:
    explicit EsmRigidAligner(const YAML::Node& cfg);

    std::string name() const override { return "ESM_RIGID"; }
    bool align(RegistrationContext& ctx) override;

private:
    /// 每层金字塔的最大迭代次数。
    int _maxIterations = 50;

    /// 参数增量收敛阈值。
    double _epsilon = 1e-4;

    /// 同步下采样图像金字塔层数，包含原图层；用于 coarse-to-fine 优化，不表示跨尺度输入。
    int _pyramidLevels = 4;

    /// 灰度图预平滑核大小，小于 3 时不平滑，偶数会自动修正为奇数。
    int _blurKernel = 5;

    /// 源图梯度幅值采样阈值，用于跳过低纹理区域。
    double _gradientThreshold = 1e-3;

    /// 采样步长，值越大参与全局 SSD 的像素越少。
    int _sampleStep = 2;

};

} // namespace ir
