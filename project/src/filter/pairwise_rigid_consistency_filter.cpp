#include "filter/pairwise_rigid_consistency_filter.h"

#include <algorithm>
#include <cmath>
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

// 计算两点欧氏距离，供 pair 长度一致性判断使用。
double pairDistance(const cv::Point2f& a, const cv::Point2f& b) {
    const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    return std::sqrt(dx * dx + dy * dy);
}

// 计算 from->to 向量方向角，单位为度。
double vectorAngleDeg(const cv::Point2f& from, const cv::Point2f& to) {
    const double dx = static_cast<double>(to.x) - static_cast<double>(from.x);
    const double dy = static_cast<double>(to.y) - static_cast<double>(from.y);
    return std::atan2(dy, dx) * 180.0 / CV_PI;
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
    _fallbackToInputIfEmpty = yaml_utils::getBool(params, "fallback_to_input_if_empty", true);

    _minPairDistance = std::max(0.0, _minPairDistance);
    _maxDistanceDiff = std::max(0.0, _maxDistanceDiff);
    _maxAngleDiffDeg = std::max(0.0, _maxAngleDiffDeg);
    _rotationBins = std::max(8, _rotationBins);
    _minVotes = std::max(0, _minVotes);
    _keepTopK = std::max(0, _keepTopK);

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
    const std::vector<cv::DMatch> input = md.filtered_matches;

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

    std::vector<PairSupport> supports;
    supports.reserve(validMatches.size() * 2);
    std::vector<int> histogram(static_cast<size_t>(_rotationBins), 0);
    const double binWidth = 360.0 / static_cast<double>(_rotationBins);

    // 5. 两两组成 match pair：
    //    - 先过滤过近 pair，避免短向量导致旋转不稳定；
    //    - 再比较 source / target pair 长度差，压掉明显不满足 rigid 的组合；
    //    - 最后把相对旋转角投票到直方图中，估计主旋转峰。
    for (size_t i = 0; i + 1 < validMatches.size(); ++i) {
        for (size_t j = i + 1; j < validMatches.size(); ++j) {
            const double srcDistance = pairDistance(srcPoints[i], srcPoints[j]);
            const double dstDistance = pairDistance(dstPoints[i], dstPoints[j]);
            if (srcDistance < _minPairDistance || dstDistance < _minPairDistance) {
                continue;
            }

            const double distanceDiff = std::abs(srcDistance - dstDistance);
            if (distanceDiff > _maxDistanceDiff) {
                continue;
            }

            const double srcAngle = vectorAngleDeg(srcPoints[i], srcPoints[j]);
            const double dstAngle = vectorAngleDeg(dstPoints[i], dstPoints[j]);
            const double rotationDeg = normalizeAngleDeg(dstAngle - srcAngle);
            const double shifted = rotationDeg + 180.0;
            int bin = static_cast<int>(std::floor(shifted / binWidth));
            bin = std::clamp(bin, 0, _rotationBins - 1);

            supports.push_back(PairSupport{static_cast<int>(i), static_cast<int>(j), rotationDeg});
            ++histogram[static_cast<size_t>(bin)];
        }
    }

    // 6. 如果没有任何有效支持 pair，说明当前 filtered matches 在 rigid 假设下太散。
    if (supports.empty()) {
        IR_LOG_WARN("PairwiseRigidConsistencyFilter: no valid support pairs.");
        if (_fallbackToInputIfEmpty) {
            return true;
        }
        md.filtered_matches.clear();
        return false;
    }

    // 7. 选出票数最高的旋转 bin，把它当作当前场景的主旋转候选。
    const auto peakIt = std::max_element(histogram.begin(), histogram.end());
    const int peakBin = static_cast<int>(std::distance(histogram.begin(), peakIt));
    const double peakCenterDeg = -180.0 + (static_cast<double>(peakBin) + 0.5) * binWidth;

    std::vector<int> votes(validMatches.size(), 0);
    // 8. 只统计“接近主旋转峰”的 pair 支持票数。
    //    这一步的目标不是直接找最终几何模型，而是给每条 match 一个
    //    “它和多少其他 match 能共同支持同一 rigid 方向”的粗评分。
    for (const auto& support : supports) {
        if (angleDiffDeg(support.rotationDeg, peakCenterDeg) > _maxAngleDiffDeg) {
            continue;
        }
        ++votes[static_cast<size_t>(support.first)];
        ++votes[static_cast<size_t>(support.second)];
    }

    std::vector<int> keptIndices;
    keptIndices.reserve(validMatches.size());
    // 9. 按最少支持票数筛掉几乎没有几何同伴支持的匹配。
    for (size_t i = 0; i < votes.size(); ++i) {
        if (votes[i] >= _minVotes) {
            keptIndices.push_back(static_cast<int>(i));
        }
    }

    // 10. 如果配置了 keepTopK，则在通过票数门槛的匹配中继续保留最稳定的一批。
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

    // 11. 过滤结果为空时，按配置决定回退到输入，避免过严过滤把后续几何彻底饿死。
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

    // 12. 写回最终保留的 matches，后续由 rigid_estimator 继续求解。
    IR_LOG_INFO("PairwiseRigidConsistencyFilter kept ",
                kept.size(),
                " / ",
                validMatches.size(),
                " matches (pairs=",
                supports.size(),
                ", peak_rotation_deg=",
                peakCenterDeg,
                ")");
    md.filtered_matches = std::move(kept);
    return true;
}

} // namespace ir
