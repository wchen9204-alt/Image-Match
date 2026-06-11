#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

#include "interfaces/i_direct_aligner.h"

namespace ir {

/// ECC 全局直接法配准器。
/// 通过最大化两幅灰度图的增强相关系数直接估计全局变换，适合整体光度关系较稳定的场景。
class EccAligner : public IDirectAligner {
public:
    explicit EccAligner(const YAML::Node& cfg);

    std::string name() const override { return "ECC"; }
    bool align(RegistrationContext& ctx) override;

private:
    /// OpenCV ECC motion model，可选 TRANSLATION / RIGID(EUCLIDEAN) / AFFINE / HOMOGRAPHY。
    std::string _motionModel = "RIGID";

    /// ECC 优化最大迭代次数。
    int _maxIterations = 100;

    /// ECC 收敛阈值。
    double _epsilon = 1e-6;

    /// ECC 内部高斯滤波核大小，用于平滑光度噪声。
    int _gaussianFilterSize = 5;
};

} // namespace ir
