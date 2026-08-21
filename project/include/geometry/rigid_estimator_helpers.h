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
    /// 候选内点的平均重投影误差。
    double meanReprojError = 0.0;
    /// 候选变换后的前景局部包含率。
    double containment = -1.0;
};

/// 由两个 filtered seed 构成的最小 rigid 假设。
/// 除点索引外，同时缓存稳定的描述子质量排名和两图共同空间间距，
/// 供候选调度阶段避免重复计算。
struct CandidateSeedPair {
    /// 对应点数组中的第一个 seed 下标。
    int first = -1;
    /// 对应点数组中的第二个 seed 下标。
    int second = -1;
    /// 两端 seed 中较差的描述子距离排名，数值越小质量越好。
    int worst_distance_rank = -1;
    /// 两端 seed 的描述子距离排名和，数值越小质量越好。
    int distance_rank_sum = -1;
    /// source / target 两侧点对平方间距中的较小值，数值越大越不易退化。
    double min_spacing2 = 0.0;
};

/// 兼容旧别名与大小写差异，将配置中的 backend 名称归一化。
std::string normalizeRigidEstimatorBackend(const std::string& raw);

/// 将候选点对调度策略归一化；支持 COVERAGE_FIRST / LEGACY_LEXICOGRAPHIC。
std::string normalizeCandidatePairStrategy(const std::string& raw);

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

/// 从 filtered 点对中构建混合式候选 seed。
/// 核心做法是在同一个 filtered 距离池里，按不同排序策略依次选出更好的点，
/// 再把这些点合并成 seed 集合，而不是从别处新增匹配。
/// 整体流程是：距离池 -> 空间分散优先选点 -> 距离最优选点 -> 主旋转峰一致选点。
std::vector<int> buildMixedCandidateSeedIndices(const std::vector<cv::Point2f>& src,
                                                const std::vector<cv::Point2f>& dst,
                                                const std::vector<float>& distances,
                                                int poolSize,
                                                int topK,
                                                double minPairDistance);

/// 从 seed 集合枚举 source / target 两侧均不退化的全部合法两点候选。
/// 返回顺序在 LEGACY_LEXICOGRAPHIC 下保持历史 i/j 嵌套顺序；
/// COVERAGE_FIRST 下由 selectNextCandidateSeedPair 按成功候选覆盖情况动态调度。
std::vector<CandidateSeedPair> buildCandidateSeedPairs(const std::vector<int>& seedIndices,
                                                       const std::vector<cv::Point2f>& src,
                                                       const std::vector<cv::Point2f>& dst,
                                                       const std::vector<float>& distances,
                                                       double minPairDistance);

/// 从尚未尝试的两点候选中选出下一对 seed。
/// COVERAGE_FIRST 优先让成功候选覆盖未使用 seed，再比较描述子质量和空间间距；
/// LEGACY_LEXICOGRAPHIC 则严格返回历史顺序中的下一对。
int selectNextCandidateSeedPair(const std::vector<CandidateSeedPair>& pairs,
                                const std::vector<unsigned char>& attempted,
                                const std::vector<int>& successfulSeedUsage,
                                const std::string& strategy);

/// 在 baseline 与额外 rigid 候选中选择最优结果。
/// 有效候选必须达到 minInliers；先以最高 containment 建立候选窗口，
/// 再以平均重投影误差选择窗口内最优候选。
bool selectBestRigidCandidate(const std::vector<cv::Mat>& candidateTransforms,
                              const std::vector<std::vector<unsigned char>>& candidateMasks,
                              const RegistrationContext& ctx,
                              const std::vector<cv::Point2f>& src,
                              const std::vector<cv::Point2f>& dst,
                              int minInliers,
                              bool enableCandidateMaskScoring,
                              int candidateMaskForegroundThreshold,
                               double candidateContainmentTieMargin,
                              double candidateDedupRotationDiffDeg,
                              double candidateDedupTranslationDiff,
                              cv::Mat& bestA,
                              std::vector<unsigned char>& bestMask,
                              int& bestInliers,
                              double& bestError,
                               double& bestContainment,
                              std::vector<cv::Mat>& validCandidateTransforms);

} // namespace ir::rigid_estimator_helpers
