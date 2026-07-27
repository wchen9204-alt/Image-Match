#include "matcher/structure/line_segment_associator.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/types.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

constexpr double kPi = 3.14159265358979323846;

/// 线段匹配时使用的轻量几何描述。
struct LineInfo {
    /// 线段中心点，用于几何 baseline 的候选排序。
    cv::Point2d center;

    /// 线段方向，归一化到 [0, pi)，按无向直线处理。
    double angle = 0.0;

    /// 线段长度，用于过滤尺度差异过大的候选。
    double length = 0.0;
};

/// 将 OpenCV 的 `(x1, y1, x2, y2)` 线段转换为匹配所需的几何描述。
LineInfo describeLine(const cv::Vec4i& line) {
    const cv::Point2d p1(static_cast<double>(line[0]), static_cast<double>(line[1]));
    const cv::Point2d p2(static_cast<double>(line[2]), static_cast<double>(line[3]));
    const cv::Point2d delta = p2 - p1;

    LineInfo out;
    out.center = (p1 + p2) * 0.5;
    out.angle = std::atan2(delta.y, delta.x);
    if (out.angle < 0.0) {
        out.angle += kPi;
    }
    if (out.angle >= kPi) {
        out.angle -= kPi;
    }
    out.length = cv::norm(delta);
    return out;
}

/// 计算两条无向线段的最小方向差。
double angleDistance(double a, double b) {
    const double d = std::abs(a - b);
    return std::min(d, kPi - d);
}

cv::Point2d matchShift(const std::vector<LineInfo>& src,
                       const std::vector<LineInfo>& dst,
                       const cv::DMatch& m) {
    return dst[static_cast<size_t>(m.trainIdx)].center -
           src[static_cast<size_t>(m.queryIdx)].center;
}

std::vector<cv::DMatch> selectConsistentUniqueMatches(const std::vector<cv::DMatch>& candidates,
                                                      const std::vector<LineInfo>& src,
                                                      const std::vector<LineInfo>& dst,
                                                      double threshold) {
    int bestCount = 0;
    double bestCost = 0.0;
    cv::Point2d bestShift(0.0, 0.0);

    // 1. 用每个候选的中心位移做轻量投票，只筛匹配，不产出最终几何模型。
    for (const auto& seed : candidates) {
        const cv::Point2d shift = matchShift(src, dst, seed);
        int count = 0;
        double cost = 0.0;
        for (const auto& m : candidates) {
            const double error = cv::norm(matchShift(src, dst, m) - shift);
            if (error <= threshold) {
                ++count;
                cost += error;
            }
        }

        if (count > bestCount || (count == bestCount && cost < bestCost)) {
            bestCount = count;
            bestCost = cost;
            bestShift = shift;
        }
    }

    std::vector<cv::DMatch> compatible;
    compatible.reserve(candidates.size());
    for (const auto& m : candidates) {
        if (cv::norm(matchShift(src, dst, m) - bestShift) <= threshold) {
            compatible.push_back(m);
        }
    }

    std::stable_sort(compatible.begin(), compatible.end(), [](const cv::DMatch& a,
                                                              const cv::DMatch& b) {
        return a.distance < b.distance;
    });

    // 2. 保证一条 source / target 线段只进入一次几何估计，避免端点点对重复轰炸 RANSAC。
    std::vector<cv::DMatch> selected;
    std::vector<unsigned char> usedSrc(src.size(), 0);
    std::vector<unsigned char> usedDst(dst.size(), 0);
    for (const auto& m : compatible) {
        if (m.queryIdx < 0 || m.trainIdx < 0) {
            continue;
        }
        const size_t qi = static_cast<size_t>(m.queryIdx);
        const size_t ti = static_cast<size_t>(m.trainIdx);
        if (qi >= usedSrc.size() || ti >= usedDst.size() || usedSrc[qi] || usedDst[ti]) {
            continue;
        }
        usedSrc[qi] = 1;
        usedDst[ti] = 1;
        selected.push_back(m);
    }
    return selected;
}

cv::Point2d averageShift(const std::vector<cv::DMatch>& matches,
                         const std::vector<LineInfo>& src,
                         const std::vector<LineInfo>& dst) {
    if (matches.empty()) {
        return cv::Point2d(0.0, 0.0);
    }

    cv::Point2d sum(0.0, 0.0);
    for (const auto& m : matches) {
        sum += matchShift(src, dst, m);
    }
    const double inv = 1.0 / static_cast<double>(matches.size());
    return cv::Point2d(sum.x * inv, sum.y * inv);
}

} // namespace

LineSegmentAssociator::LineSegmentAssociator(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    _angleThresholdDeg = yaml_utils::getDouble(params, "angleThresholdDeg", 10.0);
    _minLengthRatio = yaml_utils::getDouble(params, "minLengthRatio", 0.60);
    _maxShiftDistance = yaml_utils::getDouble(params, "maxShiftDistance", 100000.0);
    _shiftConsistencyThreshold = yaml_utils::getDouble(params, "shiftConsistencyThreshold", 8.0);
    _minMatches = yaml_utils::getInt(params, "minMatches", 4);
    _maxCandidatesPerLine = yaml_utils::getInt(params, "maxCandidatesPerLine", 5);

    IR_LOG_DEBUG("LineSegmentAssociator: angleThresholdDeg=",
                _angleThresholdDeg,
                ", minLengthRatio=",
                _minLengthRatio,
                ", shiftConsistencyThreshold=",
                _shiftConsistencyThreshold,
                ", minMatches=",
                _minMatches,
                ", maxCandidatesPerLine=",
                _maxCandidatesPerLine);
}

bool LineSegmentAssociator::associate(RegistrationContext& ctx) {
    auto& md = ctx.structure_match_data;
    md.clear();
    md.method = name();

    // 1. 检查输入必须是直线结构，并且 source / target 都有可用线段。
    if (ctx.structure_data.type != StructureType::LINE) {
        md.message = "LineSegmentAssociator requires LINE structure data";
        return false;
    }

    const auto& srcLines = ctx.structure_data.first.lines;
    const auto& dstLines = ctx.structure_data.second.lines;
    if (srcLines.empty() || dstLines.empty()) {
        md.message = "line sets are empty";
        return false;
    }

    // 2. 预计算线段中心、方向和长度，避免候选匹配阶段重复计算几何量。
    std::vector<LineInfo> src;
    std::vector<LineInfo> dst;
    src.reserve(srcLines.size());
    dst.reserve(dstLines.size());
    for (const auto& line : srcLines) {
        src.push_back(describeLine(line));
    }
    for (const auto& line : dstLines) {
        dst.push_back(describeLine(line));
    }

    // 3. 按方向差、长度比例和中心位移生成候选线段匹配。
    const double angleThreshold = _angleThresholdDeg * kPi / 180.0;
    std::vector<cv::DMatch> candidates;
    for (size_t i = 0; i < src.size(); ++i) {
        std::vector<cv::DMatch> perLine;
        for (size_t j = 0; j < dst.size(); ++j) {
            if (src[i].length <= 0.0 || dst[j].length <= 0.0) {
                continue;
            }

            const double angle = angleDistance(src[i].angle, dst[j].angle);
            if (angle > angleThreshold) {
                continue;
            }

            const double lengthRatio =
                std::min(src[i].length, dst[j].length) / std::max(src[i].length, dst[j].length);
            if (lengthRatio < _minLengthRatio) {
                continue;
            }

            const cv::Point2d shift = dst[j].center - src[i].center;
            const double shiftDistance = cv::norm(shift);
            if (shiftDistance > _maxShiftDistance) {
                continue;
            }

            const double angleCost = angle / std::max(angleThreshold, 1e-6);
            const double lengthCost = 1.0 - lengthRatio;
            const double shiftCost = shiftDistance / std::max(_maxShiftDistance, 1.0);
            const float distance = static_cast<float>(angleCost + lengthCost + 0.1 * shiftCost);
            perLine.emplace_back(static_cast<int>(i), static_cast<int>(j), distance);
        }

        std::stable_sort(perLine.begin(), perLine.end(), [](const cv::DMatch& a,
                                                            const cv::DMatch& b) {
            return a.distance < b.distance;
        });
        const int keep = _maxCandidatesPerLine <= 0
                             ? static_cast<int>(perLine.size())
                             : std::min(_maxCandidatesPerLine, static_cast<int>(perLine.size()));
        candidates.insert(candidates.end(), perLine.begin(), perLine.begin() + keep);
    }

    // 4. 用中心位移一致性做 baseline 预筛，只输出更干净的线段匹配集合。
    md.line_matches =
        selectConsistentUniqueMatches(candidates, src, dst, _shiftConsistencyThreshold);
    if (static_cast<int>(candidates.size()) < _minMatches) {
        md.message = "not enough candidate line matches: " + std::to_string(candidates.size());
        IR_LOG_WARN("LineSegmentAssociator rejected match: ", md.message);
        return false;
    }
    if (static_cast<int>(md.line_matches.size()) < _minMatches) {
        md.message = "not enough consistent line matches: " +
                     std::to_string(md.line_matches.size());
        IR_LOG_WARN("LineSegmentAssociator rejected match: ", md.message);
        return false;
    }

    // 5. 线段几何 baseline 输出候选匹配；最终几何模型由 StructurePipeline 的
    // geometry estimator（由 YAML 配置决定）统一负责。
    md.translation = averageShift(md.line_matches, src, dst);
    md.inlier_line_matches = md.line_matches;
    md.valid = true;
    md.score = candidates.empty()
                   ? 0.0
                   : static_cast<double>(md.line_matches.size()) /
                         static_cast<double>(candidates.size());
    IR_LOG_DEBUG("LineSegmentAssociator produced baseline matches: ",
                md.line_matches.size(),
                " / ",
                candidates.size(),
                ", dx=",
                md.translation.x,
                ", dy=",
                md.translation.y);
    return true;
}

} // namespace ir


