#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

#include "interfaces/i_direct_aligner.h"

namespace ir {

/// 稀疏 KLT 光流直接法配准器。
/// 先在源图检测角点，再用金字塔 LK 光流跟踪到目标图，最后从点对中拟合全局几何变换。
class KltSparseAligner : public IDirectAligner {
public:
    explicit KltSparseAligner(const YAML::Node& cfg);

    std::string name() const override { return "KLT_SPARSE"; }
    bool align(RegistrationContext& ctx) override;

private:
    /// 源图最多检测的 Shi-Tomasi 角点数量。
    int _maxCorners = 500;

    /// goodFeaturesToTrack 的角点质量阈值。
    double _qualityLevel = 0.01;

    /// 候选角点之间的最小距离，避免跟踪点过度密集。
    double _minDistance = 8.0;

    /// 角点检测邻域大小。
    int _blockSize = 7;

    /// LK 光流搜索窗口大小，会在实现中修正为奇数。
    int _windowSize = 21;

    /// LK 金字塔最大层数。
    int _maxLevel = 3;

    /// LK 迭代次数上限。
    int _maxIterations = 30;

    /// LK 收敛阈值。
    double _epsilon = 0.01;

    /// 是否按网格均匀提取角点，降低点过度集中在局部强纹理区域的风险。
    bool _gridEnabled = true;

    /// 网格划分行数。
    int _gridRows = 4;

    /// 网格划分列数。
    int _gridCols = 4;

    /// 是否对检测到的角点做 cornerSubPix 亚像素精修。
    bool _subpixRefine = true;

    /// cornerSubPix 窗口大小。
    int _subpixWindowSize = 5;

    /// cornerSubPix 最大迭代次数。
    int _subpixMaxIterations = 20;

    /// cornerSubPix 收敛阈值。
    double _subpixEpsilon = 0.03;

    /// 是否启用 forward-backward 检查剔除不稳定跟踪。
    bool _forwardBackwardCheck = true;

    /// forward-backward 回投误差阈值，单位为像素。
    double _fbThreshold = 1.5;

    /// 目标点距图像边界的最小安全边距，单位为像素。
    int _borderMargin = 3;

    /// 允许的最大 LK 跟踪误差；小于 0 时表示不启用该过滤。
    double _maxTrackError = -1.0;

    /// 全局几何 RANSAC/重投影内点阈值，单位为像素。
    double _ransacThreshold = 3.0;

    /// 全局几何鲁棒估计方法，可选 RANSAC / LMEDS；HOMOGRAPHY 额外支持 RHO。
    std::string _robustMethod = "RANSAC";

    /// 全局几何鲁棒估计最大迭代次数。
    int _ransacMaxIters = 2000;

    /// 全局几何鲁棒估计置信度。
    double _ransacConfidence = 0.99;

    /// 全局几何鲁棒估计后的 OpenCV refine 迭代次数；仅 affine 族模型生效。
    int _ransacRefineIters = 10;

    /// 全局变换接受所需的最少内点数。
    int _minInliers = 6;

    /// 全局拟合模型，可选 RIGID / SIMILARITY / AFFINE / HOMOGRAPHY。
    std::string _fitModel = "RIGID";

};

} // namespace ir

