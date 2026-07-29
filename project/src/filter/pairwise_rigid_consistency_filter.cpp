#include "filter/pairwise_rigid_consistency_filter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

// 保存一对 match pair 对某个旋转候选的支持关系。
struct PairSupport {
    int first = -1;
    int second = -1;
    double rotationDeg = 0.0;
};

// 计算两点平方距离，先筛掉短向量以减少后续开方与角度计算。
double pairDistanceSquared(const cv::Point2f& a, const cv::Point2f& b) {
    const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    return dx * dx + dy * dy;
}

// 通过叉积与点积直接计算 source 向量旋转到 target 向量的相对角度。
double relativeRotationDeg(const cv::Point2f& srcFrom,
                           const cv::Point2f& srcTo,
                           const cv::Point2f& dstFrom,
                           const cv::Point2f& dstTo) {
    const double srcDx = static_cast<double>(srcTo.x) - static_cast<double>(srcFrom.x);
    const double srcDy = static_cast<double>(srcTo.y) - static_cast<double>(srcFrom.y);
    const double dstDx = static_cast<double>(dstTo.x) - static_cast<double>(dstFrom.x);
    const double dstDy = static_cast<double>(dstTo.y) - static_cast<double>(dstFrom.y);
    const double cross = srcDx * dstDy - srcDy * dstDx;
    const double dot = srcDx * dstDx + srcDy * dstDy;
    const double rotationDeg = std::atan2(cross, dot) * 180.0 / CV_PI;
    return rotationDeg <= -180.0 ? rotationDeg + 360.0 : rotationDeg;
}

// 将角度归一化到 (-180, 180]，便于直方图统计和差值比较。
double normalizeAngleDeg(double angleDeg) {
    while (angleDeg <= -180.0) {
        angleDeg += 360.0;
    }
    while (angleDeg > 180.0) {
        angleDeg -= 360.0;
    }
    return angleDeg;
}

// 计算两个角度之间的最小夹角差。
double angleDiffDeg(double lhs, double rhs) {
    return std::abs(normalizeAngleDeg(lhs - rhs));
}

} // namespace

// 从 YAML 读取 rigid consistency 过滤参数，并做最小合法化约束。
PairwiseRigidConsistencyFilter::PairwiseRigidConsistencyFilter(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    _minPairDistance = yaml_utils::getDouble(params, "min_pair_distance", 20.0);
    _maxDistanceDiff = yaml_utils::getDouble(params, "max_distance_diff", 5.0);
    _maxAngleDiffDeg = yaml_utils::getDouble(params, "max_angle_diff_deg", 8.0);
    _rotationBins = yaml_utils::getInt(params, "rotation_bins", 36);
    _minVotes = yaml_utils::getInt(params, "min_votes", 2);
    _keepTopK = yaml_utils::getInt(params, "keep_top_k", 0);
    _maxInputMatches = yaml_utils::getInt(params, "max_input_matches", 0);
    _fallbackToInputIfEmpty = yaml_utils::getBool(params, "fallback_to_input_if_empty", true);

    _minPairDistance = std::max(0.0, _minPairDistance);
    _maxDistanceDiff = std::max(0.0, _maxDistanceDiff);
    _maxAngleDiffDeg = std::max(0.0, _maxAngleDiffDeg);
    _rotationBins = std::max(8, _rotationBins);
    _minVotes = std::max(0, _minVotes);
    _keepTopK = std::max(0, _keepTopK);
    if (_maxInputMatches > 0) {
        _maxInputMatches = std::max(3, _maxInputMatches);
    }

    IR_LOG_INFO("PairwiseRigidConsistencyFilter: min_pair_distance=",
                _minPairDistance,
                ", max_distance_diff=",
                _maxDistanceDiff,
                ", max_angle_diff_deg=",
                _maxAngleDiffDeg,
                ", rotation_bins=",
                _rotationBins,
                ", min_votes=",
                _minVotes,
                ", keep_top_k=",
                _keepTopK,
                ", max_input_matches=",
                _maxInputMatches,
                ", fallback_to_input_if_empty=",
                _fallbackToInputIfEmpty);
}

// 对当前 filtered keypoint matches 做刚体一致性预过滤。
bool PairwiseRigidConsistencyFilter::apply(RegistrationContext& ctx) {
    // 1. 当前实现只支持点特征匹配；结构法分支先保持透传。
    if (!ctx.structure_match_data.filtered_matches.empty() ||
        !ctx.structure_match_data.raw_matches_knn.empty()) {
        IR_LOG_WARN("PairwiseRigidConsistencyFilter [structure]: not supported, pass-through.");
        return true;
    }

    auto& md = ctx.keypoint_match_data;
    const auto& keypoints1 = ctx.keypoint_data.first.keypoints;
    const auto& keypoints2 = ctx.keypoint_data.second.keypoints;
    const std::vector<cv::DMatch>& input = md.filtered_matches;

    // 2. 少于 3 个匹配时，pairwise vote 几乎没有统计意义，直接保留输入。
    if (input.size() < 3) {
        IR_LOG_WARN("PairwiseRigidConsistencyFilter: too few matches to filter, keeping input.");
        return true;
    }

    std::vector<cv::DMatch> validMatches;
    std::vector<cv::Point2f> srcPoints;
    std::vector<cv::Point2f> dstPoints;
    validMatches.reserve(input.size());
    srcPoints.reserve(input.size());
    dstPoints.reserve(input.size());

    // 3. 先剔除 query/train 索引越界的无效匹配，构造后续 pairwise 分析用的点集。
    for (const auto& match : input) {
        if (match.queryIdx < 0 || match.trainIdx < 0 ||
            match.queryIdx >= static_cast<int>(keypoints1.size()) ||
            match.trainIdx >= static_cast<int>(keypoints2.size())) {
            continue;
        }
        validMatches.push_back(match);
        srcPoints.push_back(keypoints1[match.queryIdx].pt);
        dstPoints.push_back(keypoints2[match.trainIdx].pt);
    }

    // 4. 索引清洗后仍不足以形成稳定 pair 时，根据配置决定回退还是清空。
    if (validMatches.size() < 3) {
        IR_LOG_WARN("PairwiseRigidConsistencyFilter: too few valid matches after index check.");
        if (_fallbackToInputIfEmpty) {
            return true;
        }
        md.filtered_matches.clear();
        return false;
    }
    // 5. 匹配数量超出上限时，先保留质量与空间分布都具代表性的子集，控制后续 O(M^2) 投票成本。
    if (_maxInputMatches > 0 &&
        validMatches.size() > static_cast<size_t>(_maxInputMatches)) {
        // 先固定描述子距离较小的一半匹配，再以 source/target 两侧的最小间距贪心补足其余名额。
        // 5.1 为每条合法匹配建立索引，后续仅移动索引以避免反复复制匹配与坐标数据。
        std::vector<size_t> rankedIndices(validMatches.size());
        for (size_t i = 0; i < rankedIndices.size(); ++i) {
            rankedIndices[i] = i;
        }
        // 5.2 按 DMatch.distance 稳定排序，优先确定描述子距离较小的可靠匹配。
        std::stable_sort(rankedIndices.begin(),
                         rankedIndices.end(),
                         [&](size_t lhs, size_t rhs) {
                             return validMatches[lhs].distance < validMatches[rhs].distance;
                         });

        // 5.3 记录已选状态，并维护每条候选到已选集合的最小双图空间间距。
        std::vector<unsigned char> selected(validMatches.size(), 0);
        std::vector<double> minSeparation(validMatches.size(),
                                          std::numeric_limits<double>::infinity());
        const size_t limit = static_cast<size_t>(_maxInputMatches);
        const size_t qualityCount = std::min((limit + 1) / 2, rankedIndices.size());

        // 5.4 每选中一条匹配，就增量更新其余候选在 source/target 两侧的最小间距。
        const auto updateMinSeparation = [&](size_t selectedIndex) {
            for (size_t candidate = 0; candidate < validMatches.size(); ++candidate) {
                if (selected[candidate] != 0) {
                    continue;
                }
                const double sourceSeparation =
                    pairDistanceSquared(srcPoints[selectedIndex], srcPoints[candidate]);
                const double targetSeparation =
                    pairDistanceSquared(dstPoints[selectedIndex], dstPoints[candidate]);
                minSeparation[candidate] = std::min(
                    minSeparation[candidate], std::min(sourceSeparation, targetSeparation));
            }
        };

        // 5.5 先选取上限约一半的距离优先匹配，保证描述子层面的匹配质量。
        size_t selectedCount = 0;
        for (size_t rank = 0; rank < qualityCount; ++rank) {
            const size_t index = rankedIndices[rank];
            selected[index] = 1;
            ++selectedCount;
            updateMinSeparation(index);
        }

        // 5.6 再以贪心方式补足名额：每轮选择与已选集合在两张图中都最分散的候选。
        while (selectedCount < limit) {
            size_t bestIndex = validMatches.size();
            for (size_t candidate = 0; candidate < validMatches.size(); ++candidate) {
                if (selected[candidate] != 0) {
                    continue;
                }
                // 若空间分散性相同，则以较小描述子距离打破并列，保持结果确定性。
                if (bestIndex == validMatches.size() ||
                    minSeparation[candidate] > minSeparation[bestIndex] ||
                    (minSeparation[candidate] == minSeparation[bestIndex] &&
                     validMatches[candidate].distance < validMatches[bestIndex].distance)) {
                    bestIndex = candidate;
                }
            }
            if (bestIndex == validMatches.size()) {
                break;
            }
            selected[bestIndex] = 1;
            ++selectedCount;
            updateMinSeparation(bestIndex);
        }

        // 按原输入顺序重建子集，保持后续 pair 枚举及同票时的结果确定性。
        // 5.7 根据选中标记重建匹配与两侧坐标子集，作为后续 pairwise 投票的唯一输入。
        std::vector<cv::DMatch> cappedMatches;
        std::vector<cv::Point2f> cappedSrcPoints;
        std::vector<cv::Point2f> cappedDstPoints;
        cappedMatches.reserve(selectedCount);
        cappedSrcPoints.reserve(selectedCount);
        cappedDstPoints.reserve(selectedCount);
        for (size_t i = 0; i < validMatches.size(); ++i) {
            if (selected[i] == 0) {
                continue;
            }
            cappedMatches.push_back(validMatches[i]);
            cappedSrcPoints.push_back(srcPoints[i]);
            cappedDstPoints.push_back(dstPoints[i]);
        }
        // 5.8 记录实际限流规模，便于从运行日志确认该优化是否触发。
        IR_LOG_INFO("PairwiseRigidConsistencyFilter capped input ",
                    validMatches.size(),
                    " -> ",
                    cappedMatches.size(),
                    " matches (quality_spatial selection).");
        // 5.9 将代表性子集替换为当前投票输入，后续流程不再访问被限流排除的匹配。
        validMatches = std::move(cappedMatches);
        srcPoints = std::move(cappedSrcPoints);
        dstPoints = std::move(cappedDstPoints);
    }
    std::vector<std::vector<PairSupport>> supportsByBin(static_cast<size_t>(_rotationBins));
    size_t supportCount = 0;
    std::vector<int> histogram(static_cast<size_t>(_rotationBins), 0);
    const double binWidth = 360.0 / static_cast<double>(_rotationBins);
    const double minPairDistanceSquared = _minPairDistance * _minPairDistance;
// 6. 两两组成 match pair：
    //    - 先过滤过近 pair，避免短向量导致旋转不稳定；
    //    - 再比较 source / target pair 长度差，压掉明显不满足 rigid 的组合；
    //    - 最后把相对旋转角投票到直方图中，估计主旋转峰。
    for (size_t i = 0; i + 1 < validMatches.size(); ++i) {
        for (size_t j = i + 1; j < validMatches.size(); ++j) {
            const double srcDistanceSquared = pairDistanceSquared(srcPoints[i], srcPoints[j]);
            const double dstDistanceSquared = pairDistanceSquared(dstPoints[i], dstPoints[j]);
            if (srcDistanceSquared < minPairDistanceSquared ||
                dstDistanceSquared < minPairDistanceSquared) {
                continue;
            }

            const double srcDistance = std::sqrt(srcDistanceSquared);
            const double dstDistance = std::sqrt(dstDistanceSquared);
            if (std::abs(srcDistance - dstDistance) > _maxDistanceDiff) {
                continue;
            }

            const double rotationDeg = relativeRotationDeg(
                srcPoints[i], srcPoints[j], dstPoints[i], dstPoints[j]);
            const double shifted = rotationDeg + 180.0;
            int bin = static_cast<int>(std::floor(shifted / binWidth));
            bin = std::clamp(bin, 0, _rotationBins - 1);

            supportsByBin[static_cast<size_t>(bin)].push_back(
                PairSupport{static_cast<int>(i), static_cast<int>(j), rotationDeg});
            ++supportCount;
            ++histogram[static_cast<size_t>(bin)];
        }
    }
// 7. 如果没有任何有效支持 pair，说明当前 filtered matches 在 rigid 假设下太散。
    if (supportCount == 0) {
        IR_LOG_WARN("PairwiseRigidConsistencyFilter: no valid support pairs.");
        if (_fallbackToInputIfEmpty) {
            return true;
        }
        md.filtered_matches.clear();
        return false;
    }
// 8. 选出票数最高的旋转 bin，把它当作当前场景的主旋转候选。
    const auto peakIt = std::max_element(histogram.begin(), histogram.end());
    const int peakBin = static_cast<int>(std::distance(histogram.begin(), peakIt));
    const double peakCenterDeg = -180.0 + (static_cast<double>(peakBin) + 0.5) * binWidth;

    std::vector<int> votes(validMatches.size(), 0);
// 9. 只统计“接近主旋转峰”的 pair 支持票数。
    //    这一步的目标不是直接找最终几何模型，而是给每条 match 一个
    //    “它和多少其他 match 能共同支持同一 rigid 方向”的粗评分。
    const double candidateBinRadius = _maxAngleDiffDeg + binWidth * 0.5;
    for (int bin = 0; bin < _rotationBins; ++bin) {
        const double binCenterDeg = -180.0 + (static_cast<double>(bin) + 0.5) * binWidth;
        if (angleDiffDeg(binCenterDeg, peakCenterDeg) > candidateBinRadius) {
            continue;
        }
        for (const auto& support : supportsByBin[static_cast<size_t>(bin)]) {
            if (angleDiffDeg(support.rotationDeg, peakCenterDeg) > _maxAngleDiffDeg) {
                continue;
            }
            ++votes[static_cast<size_t>(support.first)];
            ++votes[static_cast<size_t>(support.second)];
        }
    }

    std::vector<int> keptIndices;
    keptIndices.reserve(validMatches.size());
// 10. 按最少支持票数筛掉几乎没有几何同伴支持的匹配。
    for (size_t i = 0; i < votes.size(); ++i) {
        if (votes[i] >= _minVotes) {
            keptIndices.push_back(static_cast<int>(i));
        }
    }
// 11. 如果配置了 keepTopK，则在通过票数门槛的匹配中继续保留最稳定的一批。
    //     优先看 votes，其次再用原始描述子距离打破并列。
    if (_keepTopK > 0 && static_cast<int>(keptIndices.size()) > _keepTopK) {
        std::stable_sort(keptIndices.begin(),
                         keptIndices.end(),
                         [&](int lhs, int rhs) {
                             if (votes[static_cast<size_t>(lhs)] != votes[static_cast<size_t>(rhs)]) {
                                 return votes[static_cast<size_t>(lhs)] > votes[static_cast<size_t>(rhs)];
                             }
                             return validMatches[static_cast<size_t>(lhs)].distance <
                                    validMatches[static_cast<size_t>(rhs)].distance;
                         });
        keptIndices.resize(static_cast<size_t>(_keepTopK));
    }

    std::vector<cv::DMatch> kept;
    kept.reserve(keptIndices.size());
    for (const int index : keptIndices) {
        kept.push_back(validMatches[static_cast<size_t>(index)]);
    }
// 12. 过滤结果为空时，按配置决定回退到输入，避免过严过滤把后续几何彻底饿死。
    if (kept.empty()) {
        IR_LOG_WARN("PairwiseRigidConsistencyFilter kept 0 / ",
                    validMatches.size(),
                    " matches (peak_rotation_deg=",
                    peakCenterDeg,
                    ")");
        if (_fallbackToInputIfEmpty) {
            md.filtered_matches = validMatches;
            return true;
        }
        md.filtered_matches.clear();
        return false;
    }
// 13. 写回最终保留的 matches，后续由 rigid_estimator 继续求解。
    IR_LOG_INFO("PairwiseRigidConsistencyFilter kept ",
                kept.size(),
                " / ",
                validMatches.size(),
                " matches (pairs=",
                supportCount,
                ", peak_rotation_deg=",
                peakCenterDeg,
                ")");
    md.filtered_matches = std::move(kept);
    return true;
}

} // namespace ir
