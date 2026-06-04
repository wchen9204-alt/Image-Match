#include "matcher/structure/line_descriptor_associator.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>

#ifdef IR_HAS_OPENCV_LINE_DESCRIPTOR
#include <opencv2/line_descriptor.hpp>
#endif

#include "core/types.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

constexpr double kPi = 3.14159265358979323846;

struct LineInfo {
    /// 线段中心点，用于估计匹配线段之间的平移。
    cv::Point2d center;

    /// 无向线段角度，范围归一化到 [0, pi)。
    double angle = 0.0;

    /// 线段长度，用于过滤尺度差异过大的候选。
    double length = 0.0;
};

/// 将字符串转为大写，便于兼容 YAML 中不同大小写的枚举值。
std::string upperAscii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

/// 将 OpenCV `(x1, y1, x2, y2)` 线段转为轻量几何描述。
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

/// 批量生成线段几何描述，避免在筛选循环中重复计算。
std::vector<LineInfo> describeLines(const std::vector<cv::Vec4i>& lines) {
    std::vector<LineInfo> out;
    out.reserve(lines.size());
    for (const auto& line : lines) {
        out.push_back(describeLine(line));
    }
    return out;
}

/// 计算两条无向线段之间的最小方向差。
double angleDistance(double a, double b) {
    const double d = std::abs(a - b);
    return std::min(d, kPi - d);
}

/// 返回单个线段匹配对应的中心平移向量。
cv::Point2d matchShift(const std::vector<LineInfo>& src,
                       const std::vector<LineInfo>& dst,
                       const cv::DMatch& m) {
    return dst[static_cast<size_t>(m.trainIdx)].center -
           src[static_cast<size_t>(m.queryIdx)].center;
}

/// 检查单个描述子匹配是否满足方向、长度和中心位移约束。
bool passLineGeometry(const std::vector<LineInfo>& src,
                      const std::vector<LineInfo>& dst,
                      const cv::DMatch& m,
                      double angleThreshold,
                      double minLengthRatio,
                      double maxShiftDistance) {
    if (m.queryIdx < 0 || m.trainIdx < 0 || m.queryIdx >= static_cast<int>(src.size()) ||
        m.trainIdx >= static_cast<int>(dst.size())) {
        return false;
    }

    const auto& s = src[static_cast<size_t>(m.queryIdx)];
    const auto& d = dst[static_cast<size_t>(m.trainIdx)];
    if (s.length <= 0.0 || d.length <= 0.0) {
        return false;
    }
    if (angleDistance(s.angle, d.angle) > angleThreshold) {
        return false;
    }

    const double lengthRatio = std::min(s.length, d.length) / std::max(s.length, d.length);
    if (lengthRatio < minLengthRatio) {
        return false;
    }
    return cv::norm(d.center - s.center) <= maxShiftDistance;
}

/// 对 LBD 原始匹配做几何一致性筛选，并输出一对一的线段匹配集合。
std::vector<cv::DMatch> filterGeometryConsistentMatches(
    const std::vector<cv::DMatch>& rawMatches,
    const std::vector<cv::Vec4i>& srcLines,
    const std::vector<cv::Vec4i>& dstLines,
    double angleThresholdDeg,
    double minLengthRatio,
    double maxShiftDistance,
    double shiftConsistencyThreshold) {
    const std::vector<LineInfo> src = describeLines(srcLines);
    const std::vector<LineInfo> dst = describeLines(dstLines);
    const double angleThreshold = angleThresholdDeg * kPi / 180.0;

    // 1. 先按单条线段的局部几何关系粗筛候选，去掉明显不可能的匹配。
    std::vector<cv::DMatch> candidates;
    candidates.reserve(rawMatches.size());
    for (const auto& m : rawMatches) {
        if (passLineGeometry(src, dst, m, angleThreshold, minLengthRatio, maxShiftDistance)) {
            candidates.push_back(m);
        }
    }
    if (candidates.empty() || shiftConsistencyThreshold <= 0.0) {
        return candidates;
    }

    // 2. 以每个候选的中心平移为种子投票，选择支持数最多的整体平移。
    int bestCount = 0;
    double bestCost = 0.0;
    cv::Point2d bestShift(0.0, 0.0);
    for (const auto& seed : candidates) {
        const cv::Point2d shift = matchShift(src, dst, seed);
        int count = 0;
        double cost = 0.0;
        for (const auto& m : candidates) {
            const double error = cv::norm(matchShift(src, dst, m) - shift);
            if (error <= shiftConsistencyThreshold) {
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

    // 3. 只保留与最佳平移一致的候选。
    std::vector<cv::DMatch> filtered;
    filtered.reserve(candidates.size());
    for (const auto& m : candidates) {
        if (cv::norm(matchShift(src, dst, m) - bestShift) <= shiftConsistencyThreshold) {
            filtered.push_back(m);
        }
    }

    std::stable_sort(filtered.begin(), filtered.end(), [](const cv::DMatch& a,
                                                          const cv::DMatch& b) {
        return a.distance < b.distance;
    });

    // 4. 描述子阶段允许多邻居候选，这里再做 source/target 线段的一对一去重。
    std::vector<cv::DMatch> unique;
    std::vector<unsigned char> usedQuery(srcLines.size(), 0);
    std::vector<unsigned char> usedTrain(dstLines.size(), 0);
    for (const auto& m : filtered) {
        if (m.queryIdx < 0 || m.trainIdx < 0) {
            continue;
        }
        const size_t qi = static_cast<size_t>(m.queryIdx);
        const size_t ti = static_cast<size_t>(m.trainIdx);
        if (qi >= usedQuery.size() || ti >= usedTrain.size() || usedQuery[qi] || usedTrain[ti]) {
            continue;
        }
        usedQuery[qi] = 1;
        usedTrain[ti] = 1;
        unique.push_back(m);
    }
    return unique;
}

/// 计算几何一致线段匹配的平均中心平移，作为结构法的平移仿射结果。
cv::Point2d averageShift(const std::vector<cv::DMatch>& matches,
                         const std::vector<cv::Vec4i>& srcLines,
                         const std::vector<cv::Vec4i>& dstLines) {
    if (matches.empty()) {
        return cv::Point2d(0.0, 0.0);
    }

    const std::vector<LineInfo> src = describeLines(srcLines);
    const std::vector<LineInfo> dst = describeLines(dstLines);
    cv::Point2d sum(0.0, 0.0);
    int count = 0;
    for (const auto& m : matches) {
        if (m.queryIdx < 0 || m.trainIdx < 0 || m.queryIdx >= static_cast<int>(src.size()) ||
            m.trainIdx >= static_cast<int>(dst.size())) {
            continue;
        }
        sum += matchShift(src, dst, m);
        ++count;
    }
    if (count == 0) {
        return cv::Point2d(0.0, 0.0);
    }
    const double inv = 1.0 / static_cast<double>(count);
    return cv::Point2d(sum.x * inv, sum.y * inv);
}

#ifdef IR_HAS_OPENCV_LINE_DESCRIPTOR

/// 将框架内部线段转换为 OpenCV line_descriptor 模块需要的 KeyLine。
cv::line_descriptor::KeyLine toKeyLine(const cv::Vec4i& line, int classId) {
    const float x1 = static_cast<float>(line[0]);
    const float y1 = static_cast<float>(line[1]);
    const float x2 = static_cast<float>(line[2]);
    const float y2 = static_cast<float>(line[3]);
    const cv::Point2f p1(x1, y1);
    const cv::Point2f p2(x2, y2);
    const cv::Point2f delta = p2 - p1;

    cv::line_descriptor::KeyLine keyLine;
    keyLine.startPointX = x1;
    keyLine.startPointY = y1;
    keyLine.endPointX = x2;
    keyLine.endPointY = y2;
    keyLine.sPointInOctaveX = x1;
    keyLine.sPointInOctaveY = y1;
    keyLine.ePointInOctaveX = x2;
    keyLine.ePointInOctaveY = y2;
    keyLine.lineLength = cv::norm(delta);
    keyLine.angle = std::atan2(delta.y, delta.x);
    keyLine.class_id = classId;
    keyLine.octave = 0;
    keyLine.numOfPixels = static_cast<int>(std::max(1.0f, keyLine.lineLength));
    keyLine.response = keyLine.lineLength;
    keyLine.size = keyLine.lineLength;
    keyLine.pt = (p1 + p2) * 0.5f;
    return keyLine;
}

/// 批量转换上游线提取器输出的线段，描述子阶段不再重新检测线段。
std::vector<cv::line_descriptor::KeyLine> toKeyLines(const std::vector<cv::Vec4i>& lines) {
    std::vector<cv::line_descriptor::KeyLine> out;
    out.reserve(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) {
        out.push_back(toKeyLine(lines[i], static_cast<int>(i)));
    }
    return out;
}

/// 调用 OpenCV LBD 实现，为 KeyLine 集合计算二进制描述子矩阵。
bool computeLbd(const cv::Mat& gray,
                std::vector<cv::line_descriptor::KeyLine>& keyLines,
                cv::Mat& descriptors,
                std::string& message) {
    descriptors.release();
    if (gray.empty()) {
        message = "input gray image is empty";
        return false;
    }
    if (keyLines.empty()) {
        message = "line set is empty";
        return false;
    }

    // 1. 调用 OpenCV contrib LBD 实现，输出二进制描述子。
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> descriptor =
        cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    descriptor->compute(gray, keyLines, descriptors);
    if (descriptors.empty()) {
        message = "LBD descriptor matrix is empty";
        return false;
    }
    return true;
}

/// 执行 LBD 描述子 KNN 匹配，ratio 接近 1.0 时保留多邻居供几何投票筛选。
std::vector<cv::DMatch> ratioMatch(const cv::Mat& srcDescriptors,
                                   const cv::Mat& dstDescriptors,
                                   const std::vector<cv::line_descriptor::KeyLine>& srcKeys,
                                   const std::vector<cv::line_descriptor::KeyLine>& dstKeys,
                                   double ratio,
                                   int knnK) {
    cv::Ptr<cv::line_descriptor::BinaryDescriptorMatcher> matcher =
        cv::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    std::vector<std::vector<cv::DMatch>> knn;
    matcher->knnMatch(srcDescriptors, dstDescriptors, knn, std::max(2, knnK));

    // 1. 对每条源线段保留通过 ratio test 的候选；宽松模式保留前 knnK 个邻居。
    std::vector<cv::DMatch> accepted;
    for (const auto& neighbours : knn) {
        if (neighbours.empty()) {
            continue;
        }

        int keep = 1;
        if (ratio >= 0.999) {
            keep = std::min(static_cast<int>(neighbours.size()), std::max(1, knnK));
        } else if (neighbours.size() >= 2 &&
                   neighbours[0].distance > static_cast<float>(ratio) * neighbours[1].distance) {
            keep = 0;
        }

        // 2. 将描述子行号映射回框架线段下标，保证后续几何过滤使用正确索引。
        for (int i = 0; i < keep; ++i) {
            const cv::DMatch& candidate = neighbours[static_cast<size_t>(i)];
            if (candidate.queryIdx < 0 || candidate.trainIdx < 0 ||
                candidate.queryIdx >= static_cast<int>(srcKeys.size()) ||
                candidate.trainIdx >= static_cast<int>(dstKeys.size())) {
                continue;
            }

            const int queryLine = srcKeys[static_cast<size_t>(candidate.queryIdx)].class_id;
            const int trainLine = dstKeys[static_cast<size_t>(candidate.trainIdx)].class_id;
            if (queryLine < 0 || trainLine < 0) {
                continue;
            }
            accepted.emplace_back(queryLine, trainLine, candidate.distance);
        }
    }
    return accepted;
}

#endif

} // namespace

/// 构造线描述子关联器，并从 YAML 中读取 LBD 检测、匹配与几何筛选参数。
LineDescriptorAssociator::LineDescriptorAssociator(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    _descriptor = yaml_utils::getString(params, "descriptor", "LBD");
    _ratio = yaml_utils::getDouble(params, "ratio", 0.75);
    _knnK = yaml_utils::getInt(params, "knnK", 8);
    _angleThresholdDeg = yaml_utils::getDouble(params, "angleThresholdDeg", 15.0);
    _minLengthRatio = yaml_utils::getDouble(params, "minLengthRatio", 0.50);
    _maxShiftDistance = yaml_utils::getDouble(params, "maxShiftDistance", 100000.0);
    _shiftConsistencyThreshold =
        yaml_utils::getDouble(params, "shiftConsistencyThreshold", 12.0);
    _minMatches = yaml_utils::getInt(params, "minMatches", 4);

    IR_LOG_INFO("LineDescriptorAssociator: descriptor=",
                _descriptor,
                ", matcher=OpenCV BinaryDescriptorMatcher",
                ", ratio=",
                _ratio,
                ", knnK=",
                _knnK,
                ", angleThresholdDeg=",
                _angleThresholdDeg,
                ", minLengthRatio=",
                _minLengthRatio,
                ", shiftConsistencyThreshold=",
                _shiftConsistencyThreshold,
                ", minMatches=",
                _minMatches);
}

/// 执行 LBD 线描述子关联，并用几何一致线段中心位移生成平移仿射。
bool LineDescriptorAssociator::associate(RegistrationContext& ctx) {
    auto& md = ctx.structure_match_data;
    md.clear();
    md.method = name();

#ifndef IR_HAS_OPENCV_LINE_DESCRIPTOR
    md.message =
        "OpenCV line_descriptor module is not available; rebuild OpenCV contrib with "
        "line_descriptor to enable LBD";
    IR_LOG_ERROR("LineDescriptorAssociator: ", md.message);
    return false;
#else
    // 1. 校验输入结构类型和当前实现支持的描述子。
    if (ctx.structure_data.type != StructureType::LINE) {
        md.message = "LineDescriptorAssociator requires LINE structure data";
        return false;
    }

    const std::string descriptor = upperAscii(_descriptor);
    if (descriptor != "LBD") {
        md.message = "unsupported line descriptor: " + _descriptor;
        return false;
    }

    const std::vector<cv::Vec4i>& srcLines = ctx.structure_data.first.lines;
    const std::vector<cv::Vec4i>& dstLines = ctx.structure_data.second.lines;
    std::vector<cv::line_descriptor::KeyLine> srcKeys = toKeyLines(srcLines);
    std::vector<cv::line_descriptor::KeyLine> dstKeys = toKeyLines(dstLines);

    // 2. 将上游提取器输出的 Vec4i 线段转为 KeyLine，计算两张图像的 LBD 描述子矩阵。
    cv::Mat srcDescriptors;
    cv::Mat dstDescriptors;
    if (!computeLbd(ctx.images.first_gray, srcKeys, srcDescriptors, md.message) ||
        !computeLbd(ctx.images.second_gray, dstKeys, dstDescriptors, md.message)) {
        IR_LOG_WARN("LineDescriptorAssociator rejected match: ", md.message);
        return false;
    }

    // 3. LBD 多邻居匹配 + 几何一致性筛选（方向、长度比、中心位移投票），得到一对一匹配。
    const std::vector<cv::DMatch> rawMatches =
        ratioMatch(srcDescriptors, dstDescriptors, srcKeys, dstKeys, _ratio, _knnK);
    md.line_matches = filterGeometryConsistentMatches(rawMatches,
                                                      srcLines,
                                                      dstLines,
                                                      _angleThresholdDeg,
                                                      _minLengthRatio,
                                                      _maxShiftDistance,
                                                      _shiftConsistencyThreshold);
    md.valid = static_cast<int>(md.line_matches.size()) >= _minMatches;
    md.score = srcLines.empty()
                   ? 0.0
                   : static_cast<double>(md.line_matches.size()) /
                         static_cast<double>(srcLines.size());
    if (!md.valid) {
        md.message = "not enough geometry-consistent LBD line matches: " +
                     std::to_string(md.line_matches.size()) +
                     " from raw=" + std::to_string(rawMatches.size());
        IR_LOG_WARN("LineDescriptorAssociator rejected match: ", md.message);
        return false;
    }

    // 4. 记录匹配线段的平均中心平移（作为信息输出，最终几何模型由 geometry estimator 决定）。
    md.translation = averageShift(md.line_matches, srcLines, dstLines);
    md.inlier_line_matches = md.line_matches;

    IR_LOG_INFO("LineDescriptorAssociator produced LBD matches: ",
                md.line_matches.size(),
                " / raw=",
                rawMatches.size(),
                ", dx=",
                md.translation.x,
                ", dy=",
                md.translation.y,
                ", srcDescriptors=",
                srcDescriptors.rows,
                ", dstDescriptors=",
                dstDescriptors.rows,
                ", srcLines=",
                srcLines.size(),
                ", dstLines=",
                dstLines.size());
    return true;
#endif
}

} // namespace ir
