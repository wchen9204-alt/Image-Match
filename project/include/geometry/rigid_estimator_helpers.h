#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "core/context.h"

namespace ir::rigid_estimator_helpers {

/// rigid 估计阶段多候选比较时使用的统一评分结构。
/// 这里既保存几何候选本身，也保存其前景 mask 几何评分结果。
struct RigidCandidateScore {
    /// 候选 rigid 变换矩阵。
    cv::Mat transform;
    /// 候选对应的内点掩码。
    std::vector<unsigned char> mask;
    /// 候选内点数量。
    int inliers = 0;
    /// 候选在当前点集上的重投影误差和。
    double reprojError = 0.0;
    /// warped source 与 target 的局部包含率。
    double containment = -1.0;
    /// source 前景 warp 到目标画布后的保留比例。
    double sourceCoverage = -1.0;
    /// target 前景按逆变换 warp 回 source 画布后的保留比例。
    double targetCoverage = -1.0;
    /// 双向 coverage，取 source / target coverage 的较大值。
    double bidirectionalCoverage = -1.0;
    /// 是否通过 containment 门槛。
    bool passedContainment = true;
    /// 是否通过 bidirectional coverage 门槛。
    bool passedBidirectionalCoverage = true;
    /// 是否整体通过前景 mask 几何门槛。
    bool passedMaskGate = true;
};

/// 兼容旧别名与大小写差异，将配置中的 backend 名称归一化。
std::string normalizeRigidEstimatorBackend(const std::string& raw);

/// 计算两点的平方距离，避免仅做比较时重复开方。
double pairDistance2(const cv::Point2f& a, const cv::Point2f& b);

/// 用两对点直接生成一个最小 rigid 假设，并投影回全部点得到候选内点掩码。
bool buildRigidCandidateFromPair(const std::vector<cv::Point2f>& src,
                                 const std::vector<cv::Point2f>& dst,
                                 int first,
                                 int second,
                                 double reprojThreshold,
                                 cv::Mat& candidateA,
                                 std::vector<unsigned char>& candidateMask);

/// 从 filtered 点对中构建混合式候选 seed：
/// 距离池 + 空间分散 + 距离优先补点 + 主旋转峰补点。
std::vector<int> buildMixedCandidateSeedIndices(const std::vector<cv::Point2f>& src,
                                                const std::vector<cv::Point2f>& dst,
                                                const std::vector<float>& distances,
                                                int poolSize,
                                                int topK,
                                                double minPairDistance);

/// 用统一规则在 baseline 和额外 rigid 候选中选择最优结果。
/// 当启用前景 mask 评分时，会优先比较 containment / coverage 门槛；
/// 若所有候选都未过门槛，则自动回退到旧的“内点数 + 误差”比较规则。
bool selectBestRigidCandidate(const std::vector<cv::Mat>& candidateTransforms,
                              const std::vector<std::vector<unsigned char>>& candidateMasks,
                              const RegistrationContext& ctx,
                              const std::vector<cv::Point2f>& src,
                              const std::vector<cv::Point2f>& dst,
                              int minInliers,
                              bool enableCandidateMaskScoring,
                              int candidateMaskForegroundThreshold,
                              double candidateMinContainment,
                              double candidateMinBidirectionalCoverage,
                              double candidateDedupRotationDiffDeg,
                              double candidateDedupTranslationDiff,
                              cv::Mat& bestA,
                              std::vector<unsigned char>& bestMask,
                              int& bestInliers,
                              double& bestError,
                              double& bestContainment,
                              double& bestBidirectionalCoverage);

} // namespace ir::rigid_estimator_helpers
