#include "geometry/rigid_estimator_helpers.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "geometry/partial_affine_utils.h"
#include "pipeline/base_pipeline_helpers.h"
#include "utils/string_utils.h"

namespace ir::rigid_estimator_helpers {
namespace {

// 将角度统一压回 (-180, 180]，便于后续做旋转差比较和直方图投票。
double normalizeAngleDeg(double angleDeg) {
    while (angleDeg <= -180.0) {
        angleDeg += 360.0;
    }
    while (angleDeg > 180.0) {
        angleDeg -= 360.0;
    }
    return angleDeg;
}

// 计算两个角度之间的最小夹角差，供旋转一致性比较使用。
double angleDiffDeg(double lhs, double rhs) {
    return std::abs(normalizeAngleDeg(lhs - rhs));
}

// 计算 from->to 向量的方向角，单位为度。
double vectorAngleDeg(const cv::Point2f& from, const cv::Point2f& to) {
    const double dx = static_cast<double>(to.x) - static_cast<double>(from.x);
    const double dy = static_cast<double>(to.y) - static_cast<double>(from.y);
    return std::atan2(dy, dx) * 180.0 / CV_PI;
}

// 在候选距离池内部估计一个主旋转峰，为后续混合 seed 补点提供方向先验。
double estimateDominantRotationPeakDeg(const std::vector<cv::Point2f>& src,
                                       const std::vector<cv::Point2f>& dst,
                                       const std::vector<int>& poolIndices,
                                       double minPairDistance,
                                       int rotationBins,
                                       double maxDistanceDiff) {
    // 在当前距离池内复用 rigid consistency 的核心思想：
    // 用 pair 的相对旋转角做直方图投票，估一个主旋转峰，
    // 供后面的“主旋转峰补点”阶段优先挑选更一致的 seed。
    if (poolIndices.size() < 2) {
        return 0.0;
    }

    std::vector<int> histogram(static_cast<size_t>(std::max(8, rotationBins)), 0);
    const double binWidth = 360.0 / static_cast<double>(histogram.size());
    const double minPairDistance2 = minPairDistance * minPairDistance;

    for (size_t i = 0; i + 1 < poolIndices.size(); ++i) {
        for (size_t j = i + 1; j < poolIndices.size(); ++j) {
            const int first = poolIndices[i];
            const int second = poolIndices[j];
            if (first < 0 || second < 0 ||
                first >= static_cast<int>(src.size()) || second >= static_cast<int>(src.size()) ||
                first >= static_cast<int>(dst.size()) || second >= static_cast<int>(dst.size())) {
                continue;
            }

            const double srcDistance2 = pairDistance2(src[first], src[second]);
            const double dstDistance2 = pairDistance2(dst[first], dst[second]);
            if (srcDistance2 < minPairDistance2 || dstDistance2 < minPairDistance2) {
                continue;
            }

            const double srcDistance = std::sqrt(srcDistance2);
            const double dstDistance = std::sqrt(dstDistance2);
            if (std::abs(srcDistance - dstDistance) > maxDistanceDiff) {
                continue;
            }

            const double rotationDeg = normalizeAngleDeg(
                vectorAngleDeg(dst[first], dst[second]) - vectorAngleDeg(src[first], src[second]));
            int bin = static_cast<int>(std::floor((rotationDeg + 180.0) / binWidth));
            bin = std::clamp(bin, 0, static_cast<int>(histogram.size()) - 1);
            ++histogram[static_cast<size_t>(bin)];
        }
    }

    const auto peakIt = std::max_element(histogram.begin(), histogram.end());
    if (peakIt == histogram.end() || *peakIt <= 0) {
        return 0.0;
    }

    const int peakBin = static_cast<int>(std::distance(histogram.begin(), peakIt));
    return -180.0 + (static_cast<double>(peakBin) + 0.5) * binWidth;
}

// 向索引列表追加一个值；若已存在则保持原列表不变。
void appendUniqueIndex(std::vector<int>& indices, int value) {
    if (std::find(indices.begin(), indices.end(), value) == indices.end()) {
        indices.push_back(value);
    }
}

// 从距离池中按“空间更分散”这一排序倾向挑点。
// 这一阶段只是在当前候选池内部选点，并不会引入任何新匹配。
std::vector<int> selectSpatiallyDiverseIndices(const std::vector<int>& sortedPoolIndices,
                                               const std::vector<cv::Point2f>& src,
                                               const std::vector<cv::Point2f>& dst,
                                               double minPairDistance,
                                               int maxCount) {
    // 先从距离较好的点里挑一批空间上更分散的 seed，
    // 避免候选全部挤在局部区域，降低 2 点 rigid 假设退化的概率。
    std::vector<int> selected;
    if (maxCount <= 0) {
        return selected;
    }

    const double minDistance2 = minPairDistance * minPairDistance;
    for (const int index : sortedPoolIndices) {
        bool tooClose = false;
        for (const int kept : selected) {
            if (pairDistance2(src[index], src[kept]) < minDistance2 ||
                pairDistance2(dst[index], dst[kept]) < minDistance2) {
                tooClose = true;
                break;
            }
        }
        if (tooClose) {
            continue;
        }
        selected.push_back(index);
        if (static_cast<int>(selected.size()) >= maxCount) {
            break;
        }
    }

    return selected;
}

// 为单个 rigid 候选补算前景局部包含率，用于候选模型的几何质量排序。
bool evaluateRigidCandidateMaskScore(const cv::Mat& sourceMask,
                                     const cv::Mat& targetMask,
                                     const cv::Mat& candidateA,
                                     RigidCandidateScore& score) {
    score.containment = -1.0;
    if (candidateA.empty() || sourceMask.empty() || targetMask.empty()) {
        return false;
    }

    cv::Mat warpedSourceMask;
    if (!base_pipeline_helpers::warpMaskToTargetSize(
            sourceMask, targetMask.size(), candidateA, warpedSourceMask)) {
        return false;
    }

    score.containment = base_pipeline_helpers::computeMaskLocalContainment(
        sourceMask, warpedSourceMask, targetMask);
    return score.containment >= 0.0;
}
// 比较两个 rigid 候选的优先级，用于最终多候选选模。
bool preferRigidCandidateScore(const RigidCandidateScore& lhs, const RigidCandidateScore& rhs) {
    // 候选排序优先级：
    // 1) 是否通过前景 mask 门槛
    // 2) 内点数
    // 3) containment
    // 4) 重投影误差
    if (lhs.passedMaskGate != rhs.passedMaskGate) {
        return lhs.passedMaskGate;
    }
    if (lhs.inliers != rhs.inliers) {
        return lhs.inliers > rhs.inliers;
    }
    if (lhs.containment != rhs.containment) {
        return lhs.containment > rhs.containment;
    }
    return lhs.reprojError < rhs.reprojError;
}

// 从 2x3 刚体矩阵中提取旋转角，供轻量去重比较使用。
double rigidRotationDeg(const cv::Mat& transform) {
    if (transform.empty() || transform.rows < 2 || transform.cols < 3) {
        return 0.0;
    }
    const double r00 = transform.at<double>(0, 0);
    const double r10 = transform.at<double>(1, 0);
    return std::atan2(r10, r00) * 180.0 / CV_PI;
}

// 从 2x3 刚体矩阵中提取平移向量，供轻量去重比较使用。
cv::Point2d rigidTranslation(const cv::Mat& transform) {
    if (transform.empty() || transform.rows < 2 || transform.cols < 3) {
        return cv::Point2d(0.0, 0.0);
    }
    return cv::Point2d(transform.at<double>(0, 2), transform.at<double>(1, 2));
}

// 判断两个 rigid 候选是否可以视为“几乎同一个模型”。
bool isNearDuplicateCandidate(const cv::Mat& lhs,
                              const cv::Mat& rhs,
                              double maxRotationDiffDeg,
                              double maxTranslationDiff) {
    if (lhs.empty() || rhs.empty()) {
        return false;
    }

    const double rotationDiff = angleDiffDeg(rigidRotationDeg(lhs), rigidRotationDeg(rhs));
    if (rotationDiff > maxRotationDiffDeg) {
        return false;
    }

    const cv::Point2d lhsT = rigidTranslation(lhs);
    const cv::Point2d rhsT = rigidTranslation(rhs);
    const double dx = lhsT.x - rhsT.x;
    const double dy = lhsT.y - rhsT.y;
    const double translationDiff = std::sqrt(dx * dx + dy * dy);
    return translationDiff <= maxTranslationDiff;
}

} // namespace

// 兼容旧别名与大小写差异，将刚体估计后端名称归一化。
std::string normalizeRigidEstimatorBackend(const std::string& raw) {
    const std::string key = string_utils::normalizedKey(raw);
    if (key == "CUSTOMRIGIDRANSAC" || key == "CUSTOMRANSAC" || key == "RIGIDRANSAC") {
        return "CUSTOM_RIGID_RANSAC";
    }
    return "OPENCV_PARTIAL_AFFINE";
}

// 计算两点的平方距离，供候选生成阶段做快速距离比较。
double pairDistance2(const cv::Point2f& a, const cv::Point2f& b) {
    const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    return dx * dx + dy * dy;
}

// 用两对点生成一个最小 rigid 候选，并在全部点上回投得到初始内点掩码。
bool buildRigidCandidateFromPair(const std::vector<cv::Point2f>& src,
                                 const std::vector<cv::Point2f>& dst,
                                 int first,
                                 int second,
                                 double reprojThreshold,
                                 cv::Mat& candidateA,
                                 std::vector<unsigned char>& candidateMask) {
    if (first < 0 || second < 0 || first == second ||
        first >= static_cast<int>(src.size()) || second >= static_cast<int>(src.size()) ||
        first >= static_cast<int>(dst.size()) || second >= static_cast<int>(dst.size())) {
        return false;
    }

    std::vector<cv::Point2f> sampleSrc = {src[first], src[second]};
    std::vector<cv::Point2f> sampleDst = {dst[first], dst[second]};
    if (!partial_affine_utils::estimateRigidNoScale2D(sampleSrc, sampleDst, candidateA)) {
        return false;
    }

    candidateMask = partial_affine_utils::maskByReprojection(src, dst, candidateA, reprojThreshold);
    return partial_affine_utils::countInliers(candidateMask) >= 2;
}

// 通过“距离池 + 多种排序策略选点”生成混合式候选 seed。
// 这里后续几轮选点都仍然发生在同一个 sortedIndices 距离池里，
// 只是按不同规则重新排序或筛选，再把选中的好点合并进 seedIndices。
std::vector<int> buildMixedCandidateSeedIndices(const std::vector<cv::Point2f>& src,
                                                const std::vector<cv::Point2f>& dst,
                                                const std::vector<float>& distances,
                                                int poolSize,
                                                int topK,
                                                double minPairDistance) {
    // 第 1 步：先按描述子距离排序，并截出一个比 topK 更大的距离池，
    // 避免“只看前 topK”时过早丢掉几何上可能更有用的点。
    std::vector<int> sortedIndices(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        sortedIndices[i] = static_cast<int>(i);
    }
    std::sort(sortedIndices.begin(),
              sortedIndices.end(),
              [&](int lhs, int rhs) {
                  return distances[lhs] < distances[rhs];
              });

    const int effectivePoolSize = std::min(poolSize, static_cast<int>(sortedIndices.size()));
    sortedIndices.resize(static_cast<size_t>(effectivePoolSize));

    std::vector<int> seedIndices;
    seedIndices.reserve(static_cast<size_t>(std::max(topK, effectivePoolSize)));

    // 第 2 步：从距离池中优先选出空间更分散的点，保证候选 seed 的几何覆盖范围。
    // 这一轮先保“分布别太挤”，避免后续两点构造 rigid 时老是取到局部小对。
    const std::vector<int> spatialSeeds =
        selectSpatiallyDiverseIndices(sortedIndices, src, dst, minPairDistance, topK);
    for (const int index : spatialSeeds) {
        appendUniqueIndex(seedIndices, index);
    }

    // 第 3 步：再从同一个距离池中按 descriptor distance 排序，选一部分距离最优的点。
    // 这样即使某些点因为空间上靠得近，在第 2 步没被选进来，只要匹配质量足够好，
    // 仍然可以进入最终 seed 集合。
    for (int i = 0; i < std::min(topK / 2, effectivePoolSize); ++i) {
        appendUniqueIndex(seedIndices, sortedIndices[static_cast<size_t>(i)]);
    }

    // 第 4 步：根据距离池内估计出的主旋转峰，再从同一个距离池中选一批旋转更一致的点，
    // 缓解弱纹理 / 长方形 / 180 度歧义场景里只靠距离选点的局限。
    // 这一轮本质上也是换一种排序标准，把和主旋转方向更合拍的点优先加入 seed。
    const double dominantRotationDeg = estimateDominantRotationPeakDeg(
        src, dst, sortedIndices, minPairDistance, 36, 8.0);
    std::vector<std::pair<double, int>> rotationRanked;
    rotationRanked.reserve(sortedIndices.size());
    for (const int anchor : sortedIndices) {
        double bestAngleDiff = 180.0;
        for (const int other : sortedIndices) {
            if (anchor == other) {
                continue;
            }
            if (pairDistance2(src[anchor], src[other]) < minPairDistance * minPairDistance ||
                pairDistance2(dst[anchor], dst[other]) < minPairDistance * minPairDistance) {
                continue;
            }
            const double rotationDeg = normalizeAngleDeg(
                vectorAngleDeg(dst[anchor], dst[other]) - vectorAngleDeg(src[anchor], src[other]));
            bestAngleDiff = std::min(bestAngleDiff, angleDiffDeg(rotationDeg, dominantRotationDeg));
        }
        rotationRanked.push_back({bestAngleDiff, anchor});
    }
    std::sort(rotationRanked.begin(),
              rotationRanked.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.first < rhs.first;
              });
    for (const auto& [_, index] : rotationRanked) {
        if (static_cast<int>(seedIndices.size()) >= topK) {
            break;
        }
        appendUniqueIndex(seedIndices, index);
    }

    return seedIndices;
}

// 在 baseline 与额外 rigid 候选之间按统一评分规则选出最优模型。
bool selectBestRigidCandidate(const std::vector<cv::Mat>& candidateTransforms,
                              const std::vector<std::vector<unsigned char>>& candidateMasks,
                              const RegistrationContext& ctx,
                              const std::vector<cv::Point2f>& src,
                              const std::vector<cv::Point2f>& dst,
                              int minInliers,
                              bool enableCandidateMaskScoring,
                              int candidateMaskForegroundThreshold,
                              double candidateMinContainment,
                              double candidateDedupRotationDiffDeg,
                              double candidateDedupTranslationDiff,
                              cv::Mat& bestA,
                              std::vector<unsigned char>& bestMask,
                              int& bestInliers,
                              double& bestError,
                              double& bestContainment) {
    // 第 1 步：先做一次轻量去重，避免几乎相同的 rigid 候选重复参与后续评分。
    bestA.release();
    bestMask.clear();
    bestInliers = 0;
    bestError = std::numeric_limits<double>::infinity();
    bestContainment = -1.0;

    std::vector<RigidCandidateScore> scoredCandidates;
    scoredCandidates.reserve(std::min(candidateTransforms.size(), candidateMasks.size()));
    std::vector<cv::Mat> dedupedTransforms;
    std::vector<std::vector<unsigned char>> dedupedMasks;
    dedupedTransforms.reserve(std::min(candidateTransforms.size(), candidateMasks.size()));
    dedupedMasks.reserve(std::min(candidateTransforms.size(), candidateMasks.size()));

    for (size_t i = 0; i < candidateTransforms.size() && i < candidateMasks.size(); ++i) {
        const cv::Mat& candidateA = candidateTransforms[i];
        const std::vector<unsigned char>& candidateMask = candidateMasks[i];
        if (candidateA.empty()) {
            continue;
        }

        bool duplicate = false;
        for (const auto& kept : dedupedTransforms) {
            if (isNearDuplicateCandidate(candidateA,
                                         kept,
                                         candidateDedupRotationDiffDeg,
                                         candidateDedupTranslationDiff)) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }

        dedupedTransforms.push_back(candidateA);
        dedupedMasks.push_back(candidateMask);
    }

    // 第 2 步：遍历去重后的候选，统一补齐内点、重投影误差和前景 mask 几何评分。
    // Source/target masks are invariant across candidate transforms; build them once per case.
    cv::Mat sourceScoringMask;
    cv::Mat targetScoringMask;
    if (enableCandidateMaskScoring && !dedupedTransforms.empty()) {
        base_pipeline_helpers::buildForegroundMask(
            ctx.images.first, candidateMaskForegroundThreshold, sourceScoringMask);
        base_pipeline_helpers::buildForegroundMask(
            ctx.images.second, candidateMaskForegroundThreshold, targetScoringMask);
    }

    for (size_t i = 0; i < dedupedTransforms.size() && i < dedupedMasks.size(); ++i) {
        const cv::Mat& candidateA = dedupedTransforms[i];
        const std::vector<unsigned char>& candidateMask = dedupedMasks[i];
        if (candidateA.empty()) {
            continue;
        }

        const int candidateInliers = partial_affine_utils::countInliers(candidateMask);
        if (candidateInliers < minInliers) {
            continue;
        }

        RigidCandidateScore score;
        score.transform = candidateA.clone();
        score.mask = candidateMask;
        score.inliers = candidateInliers;
        score.reprojError =
            partial_affine_utils::reprojectionErrorSum(src, dst, candidateMask, candidateA);
        if (enableCandidateMaskScoring) {
            // 启用前景几何评分时，额外标记 containment / coverage 是否达标。
            evaluateRigidCandidateMaskScore(
                sourceScoringMask, targetScoringMask, candidateA, score);
            if (candidateMinContainment >= 0.0) {
                score.passedContainment =
                    score.containment >= 0.0 && score.containment >= candidateMinContainment;
            }
            score.passedMaskGate = score.passedContainment;
        }
        scoredCandidates.push_back(std::move(score));
    }

    if (scoredCandidates.empty()) {
        return false;
    }

    // 第 3 步：如果所有候选都没过前景门槛，则回退到旧比较规则，
    // 避免新评分过严时直接把整个 rigid 结果清空。
    const auto passedMaskGate = std::count_if(scoredCandidates.begin(),
                                              scoredCandidates.end(),
                                              [](const RigidCandidateScore& score) {
                                                  return score.passedMaskGate;
                                              });
    if (enableCandidateMaskScoring && passedMaskGate == 0) {
        for (auto& score : scoredCandidates) {
            score.passedMaskGate = true;
        }
    }

    // 第 4 步：最终按统一优先级选出最优候选，并把关键评分一并返回给上层日志。
    const auto bestIt = std::max_element(scoredCandidates.begin(),
                                         scoredCandidates.end(),
                                         [](const RigidCandidateScore& lhs,
                                            const RigidCandidateScore& rhs) {
                                             return preferRigidCandidateScore(rhs, lhs);
                                         });
    if (bestIt == scoredCandidates.end()) {
        return false;
    }

    bestA = bestIt->transform.clone();
    bestMask = bestIt->mask;
    bestInliers = bestIt->inliers;
    bestError = bestIt->reprojError;
    bestContainment = bestIt->containment;
    return !bestA.empty();
}

} // namespace ir::rigid_estimator_helpers
