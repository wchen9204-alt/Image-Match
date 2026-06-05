#include "matcher/structure/line_descriptor_associator.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#ifdef IR_HAS_OPENCV_LINE_DESCRIPTOR
#include <opencv2/line_descriptor.hpp>
#endif

#include "core/types.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

// ========================================================================
// 几何一致性筛选
// ========================================================================

constexpr double kPi = 3.14159265358979323846;

struct LineInfo {
    cv::Point2d center;
    double angle = 0.0;
    double length = 0.0;
};

/// 将字符串转为大写，便于兼容 YAML 中不同大小写的枚举值。
std::string upperAscii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

LineInfo describeLineGeom(const cv::Vec4i& line) {
    const cv::Point2d p1(static_cast<double>(line[0]), static_cast<double>(line[1]));
    const cv::Point2d p2(static_cast<double>(line[2]), static_cast<double>(line[3]));
    const cv::Point2d delta = p2 - p1;
    LineInfo out;
    out.center = (p1 + p2) * 0.5;
    out.angle = std::atan2(delta.y, delta.x);
    if (out.angle < 0.0) out.angle += kPi;
    if (out.angle >= kPi) out.angle -= kPi;
    out.length = cv::norm(delta);
    return out;
}

std::vector<LineInfo> describeLinesGeom(const std::vector<cv::Vec4i>& lines) {
    std::vector<LineInfo> out;
    out.reserve(lines.size());
    for (const auto& line : lines) out.push_back(describeLineGeom(line));
    return out;
}

double angleDistanceGeom(double a, double b) {
    const double d = std::abs(a - b);
    return std::min(d, kPi - d);
}

cv::Point2d matchShiftGeom(const std::vector<LineInfo>& src,
                           const std::vector<LineInfo>& dst,
                           const cv::DMatch& m) {
    return dst[static_cast<size_t>(m.trainIdx)].center -
           src[static_cast<size_t>(m.queryIdx)].center;
}

/// 几何一致性筛选：方向 + 长度比 + 中心位移投票 + 一对一去重。
std::vector<cv::DMatch> filterGeometricConsistent(
    const std::vector<cv::DMatch>& rawMatches,
    const std::vector<cv::Vec4i>& srcLines,
    const std::vector<cv::Vec4i>& dstLines,
    double angleThresholdDeg,
    double minLengthRatio,
    double shiftConsistencyThreshold) {
    const std::vector<LineInfo> src = describeLinesGeom(srcLines);
    const std::vector<LineInfo> dst = describeLinesGeom(dstLines);
    const double angleThreshold = angleThresholdDeg * kPi / 180.0;

    // 1. 粗筛：方向差 + 长度比
    std::vector<cv::DMatch> candidates;
    candidates.reserve(rawMatches.size());
    for (const auto& m : rawMatches) {
        if (m.queryIdx < 0 || m.trainIdx < 0 ||
            m.queryIdx >= static_cast<int>(src.size()) ||
            m.trainIdx >= static_cast<int>(dst.size()))
            continue;
        const auto& s = src[static_cast<size_t>(m.queryIdx)];
        const auto& d = dst[static_cast<size_t>(m.trainIdx)];
        if (s.length <= 0.0 || d.length <= 0.0) continue;
        if (angleDistanceGeom(s.angle, d.angle) > angleThreshold) continue;
        const double lr = std::min(s.length, d.length) / std::max(s.length, d.length);
        if (lr < minLengthRatio) continue;
        candidates.push_back(m);
    }
    if (candidates.empty()) return {};

    // 2. 中心位移一致性投票
    int bestCount = 0;
    double bestCost = 0.0;
    cv::Point2d bestShift(0.0, 0.0);
    for (const auto& seed : candidates) {
        const cv::Point2d shift = matchShiftGeom(src, dst, seed);
        int count = 0;
        double cost = 0.0;
        for (const auto& m : candidates) {
            const double err = cv::norm(matchShiftGeom(src, dst, m) - shift);
            if (err <= shiftConsistencyThreshold) { ++count; cost += err; }
        }
        if (count > bestCount || (count == bestCount && cost < bestCost)) {
            bestCount = count; bestCost = cost; bestShift = shift;
        }
    }

    // 3. 只保留与最佳平移一致的候选
    std::vector<cv::DMatch> filtered;
    for (const auto& m : candidates) {
        if (cv::norm(matchShiftGeom(src, dst, m) - bestShift) <= shiftConsistencyThreshold)
            filtered.push_back(m);
    }
    std::stable_sort(filtered.begin(), filtered.end(),
                     [](const cv::DMatch& a, const cv::DMatch& b) {
                         return a.distance < b.distance;
                     });

    // 4. 一对一去重
    std::vector<cv::DMatch> unique;
    std::vector<unsigned char> usedQ(srcLines.size(), 0);
    std::vector<unsigned char> usedT(dstLines.size(), 0);
    for (const auto& m : filtered) {
        const size_t qi = static_cast<size_t>(m.queryIdx);
        const size_t ti = static_cast<size_t>(m.trainIdx);
        if (qi >= usedQ.size() || ti >= usedT.size() || usedQ[qi] || usedT[ti]) continue;
        usedQ[qi] = 1; usedT[ti] = 1;
        unique.push_back(m);
    }
    return unique;
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
    // 不设置 size：让 OpenCV 内部默认使用 lineLength * 0.5 作为带宽
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

/// 通用 KNN 匹配：根据描述子类型自动选择匹配器（二进制→BinaryDescriptorMatcher，浮点→BFMatcher L2）。
/// 输出重映射后的 KNN 分组（描述子索引→线段 class_id）。
std::vector<std::vector<cv::DMatch>> knnMatchDescriptors(
    const cv::Mat& srcDescriptors,
    const cv::Mat& dstDescriptors,
    const std::vector<cv::line_descriptor::KeyLine>& srcKeys,
    const std::vector<cv::line_descriptor::KeyLine>& dstKeys,
    int knnK,
    bool binaryDescriptor,
    bool useFlann) {
    std::vector<std::vector<cv::DMatch>> knn;

    if (binaryDescriptor) {
#ifdef IR_HAS_OPENCV_LINE_DESCRIPTOR
        cv::Ptr<cv::line_descriptor::BinaryDescriptorMatcher> matcher =
            cv::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
        matcher->knnMatch(srcDescriptors, dstDescriptors, knn, std::max(2, knnK));
#else
        return {};
#endif
    } else if (useFlann) {
        cv::Ptr<cv::FlannBasedMatcher> matcher =
            cv::FlannBasedMatcher::create();
        matcher->knnMatch(srcDescriptors, dstDescriptors, knn, std::max(2, knnK));
    } else {
        cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, false);
        matcher->knnMatch(srcDescriptors, dstDescriptors, knn, std::max(2, knnK));
    }

    // 将描述子行号重映射为框架线段下标（class_id）
    std::vector<std::vector<cv::DMatch>> out;
    out.reserve(knn.size());
    for (const auto& neighbours : knn) {
        std::vector<cv::DMatch> remapped;
        remapped.reserve(neighbours.size());
        for (const auto& candidate : neighbours) {
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
            remapped.emplace_back(queryLine, trainLine, candidate.distance);
        }
        if (!remapped.empty()) {
            out.push_back(std::move(remapped));
        }
    }
    return out;
}

#endif

} // namespace

// ========================================================================
// 描述子计算（在 ir 命名空间中，被 associate() 调用）
// ========================================================================

bool computeLbd(const cv::Mat& gray,
                std::vector<cv::line_descriptor::KeyLine>& keyLines,
                cv::Mat& descriptors,
                std::string& message) {
#ifdef IR_HAS_OPENCV_LINE_DESCRIPTOR
    descriptors.release();
    if (gray.empty()) { message = "input gray image is empty"; return false; }
    if (keyLines.empty()) { message = "line set is empty"; return false; }
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> descriptor =
        cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    descriptor->compute(gray, keyLines, descriptors);
    if (descriptors.empty()) { message = "LBD descriptor matrix is empty"; return false; }
    return true;
#else
    message = "OpenCV line_descriptor module not available"; return false;
#endif
}

namespace {

int orientationBin(float dx, float dy) {
    float angle = std::atan2(dy, dx);
    if (angle < 0.0f) angle += static_cast<float>(CV_PI);
    if (angle >= static_cast<float>(CV_PI)) angle -= static_cast<float>(CV_PI);
    const float halfBin = static_cast<float>(CV_PI) / 8.0f;
    int bin = static_cast<int>(std::floor((angle + halfBin) / (static_cast<float>(CV_PI) / 4.0f)));
    if (bin >= 4) bin = 0;
    return bin;
}

bool computeMsldForLine(const cv::Mat& gray,
                         const cv::Vec4i& line,
                         int bandWidth, int strips,
                         std::vector<float>& desc) {
    const float x1 = static_cast<float>(line[0]), y1 = static_cast<float>(line[1]);
    const float x2 = static_cast<float>(line[2]), y2 = static_cast<float>(line[3]);
    const float cx = (x1 + x2) * 0.5f, cy = (y1 + y2) * 0.5f;
    const float dx = x2 - x1, dy = y2 - y1;
    const float lineLen = std::sqrt(dx * dx + dy * dy);
    if (lineLen < 2.0f || bandWidth < 2 || strips < 1) return false;

    const float angle = std::atan2(dy, dx);
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point2f(cx, cy),
                                           static_cast<double>(angle * 180.0 / CV_PI), 1.0);
    const int patchW = std::max(2, static_cast<int>(lineLen));
    const int patchH = std::max(2, bandWidth);
    cv::Mat patch;
    rot.at<double>(0, 2) += patchW * 0.5 - cx;
    rot.at<double>(1, 2) += patchH * 0.5 - cy;
    cv::warpAffine(gray, patch, rot, cv::Size(patchW, patchH),
                   cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    cv::Mat gradX, gradY;
    cv::Sobel(patch, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(patch, gradY, CV_32F, 0, 1, 3);

    const int numBins = 4, descLen = strips * 2 * numBins;
    desc.assign(static_cast<size_t>(descLen), 0.0f);
    const float cellW = static_cast<float>(patchW) / strips;
    const float cellH = static_cast<float>(patchH) * 0.5f;

    for (int s = 0; s < strips; ++s) {
        const int x0 = std::max(0, static_cast<int>(s * cellW));
        const int x1 = std::min(patchW, static_cast<int>((s + 1) * cellW));
        if (x1 <= x0) continue;
        for (int b = 0; b < 2; ++b) {
            const int y0 = std::max(0, static_cast<int>(b * cellH));
            const int y1 = std::min(patchH, static_cast<int>((b + 1) * cellH));
            if (y1 <= y0) continue;
            const int histOff = (s * 2 + b) * numBins;
            for (int y = y0; y < y1; ++y) {
                const float* rowGx = gradX.ptr<float>(y);
                const float* rowGy = gradY.ptr<float>(y);
                for (int x = x0; x < x1; ++x) {
                    const float gx = rowGx[x], gy = rowGy[x];
                    const float mag = std::sqrt(gx * gx + gy * gy);
                    if (mag < 1e-6f) continue;
                    desc[static_cast<size_t>(histOff + orientationBin(gx, gy))] += mag;
                }
            }
        }
    }

    float norm = 0.0f;
    for (float v : desc) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
        const float inv = 1.0f / norm;
        for (float& v : desc) v *= inv;
        bool reclamp = false;
        for (float& v : desc) { if (v > 0.2f) { v = 0.2f; reclamp = true; } }
        if (reclamp) {
            float norm2 = 0.0f;
            for (float v : desc) norm2 += v * v;
            norm2 = std::sqrt(norm2);
            if (norm2 > 1e-8f) { const float inv2 = 1.0f / norm2; for (float& v : desc) v *= inv2; }
        }
    }
    return true;
}

// line-SIFT 单线段描述子
bool computeLineSiftForLine(const cv::Mat& gray,
                             const cv::Vec4i& line,
                             int bandWidth, int strips, int bands, int numBins,
                             std::vector<float>& desc) {
    const float x1 = static_cast<float>(line[0]), y1 = static_cast<float>(line[1]);
    const float x2 = static_cast<float>(line[2]), y2 = static_cast<float>(line[3]);
    const float cx = (x1 + x2) * 0.5f, cy = (y1 + y2) * 0.5f;
    const float dx = x2 - x1, dy = y2 - y1;
    const float lineLen = std::sqrt(dx * dx + dy * dy);
    if (lineLen < 2.0f || bandWidth < 2 || strips < 1 || bands < 1) return false;

    // warpAffine 提取对齐矩形带
    const float angle = std::atan2(dy, dx);
    cv::Mat rot = cv::getRotationMatrix2D(cv::Point2f(cx, cy),
                                           static_cast<double>(angle * 180.0 / CV_PI), 1.0);
    const int patchW = std::max(2, static_cast<int>(lineLen));
    const int patchH = std::max(2, bandWidth);
    cv::Mat patch;
    rot.at<double>(0, 2) += patchW * 0.5 - cx;
    rot.at<double>(1, 2) += patchH * 0.5 - cy;
    cv::warpAffine(gray, patch, rot, cv::Size(patchW, patchH),
                   cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    // 梯度
    cv::Mat gradX, gradY;
    cv::Sobel(patch, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(patch, gradY, CV_32F, 0, 1, 3);

    const int descDim = strips * bands * numBins;
    desc.assign(static_cast<size_t>(descDim), 0.0f);
    const float cellW = static_cast<float>(patchW) / strips;
    const float cellH = static_cast<float>(patchH) / bands;

    for (int s = 0; s < strips; ++s) {
        const int x0 = std::max(0, static_cast<int>(s * cellW));
        const int x1 = std::min(patchW, static_cast<int>((s + 1) * cellW));
        if (x1 <= x0) continue;
        const float cxCell = (x0 + x1) * 0.5f;
        for (int b = 0; b < bands; ++b) {
            const int y0 = std::max(0, static_cast<int>(b * cellH));
            const int y1 = std::min(patchH, static_cast<int>((b + 1) * cellH));
            if (y1 <= y0) continue;
            const float cyCell = (y0 + y1) * 0.5f;
            const int off = (s * bands + b) * numBins;

            for (int y = y0; y < y1; ++y) {
                const float* rowGx = gradX.ptr<float>(y);
                const float* rowGy = gradY.ptr<float>(y);
                for (int x = x0; x < x1; ++x) {
                    const float gx = rowGx[x], gy = rowGy[x];
                    const float mag = std::sqrt(gx * gx + gy * gy);
                    if (mag < 1e-6f) continue;
                    // 高斯加权（距离子区域中心）
                    const float dxc = static_cast<float>(x) - cxCell;
                    const float dyc = static_cast<float>(y) - cyCell;
                    const float sigma = std::max(cellW, cellH) * 0.5f;
                    const float w = mag * std::exp(-(dxc * dxc + dyc * dyc) / (2.0f * sigma * sigma));
                    float angleVal = std::atan2(gy, gx);
                    if (angleVal < 0.0f) angleVal += 2.0f * static_cast<float>(CV_PI);
                    int bin = static_cast<int>(angleVal / (2.0f * static_cast<float>(CV_PI) / numBins));
                    if (bin >= numBins) bin = 0;
                    desc[static_cast<size_t>(off + bin)] += w;
                }
            }
        }
    }

    // L2 归一化 + clamp(0.2) + 再归一化
    float norm = 0.0f;
    for (float v : desc) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
        const float inv = 1.0f / norm;
        for (float& v : desc) v *= inv;
        bool reclamp = false;
        for (float& v : desc) { if (v > 0.2f) { v = 0.2f; reclamp = true; } }
        if (reclamp) {
            float norm2 = 0.0f;
            for (float v : desc) norm2 += v * v;
            norm2 = std::sqrt(norm2);
            if (norm2 > 1e-8f) { const float inv2 = 1.0f / norm2; for (float& v : desc) v *= inv2; }
        }
    }
    return true;
}

} // namespace

bool computeMsld(const cv::Mat& gray,
                 const std::vector<cv::Vec4i>& lines,
                 cv::Mat& descriptors,
                 int bandWidth, int strips,
                 std::string& message) {
    descriptors.release();
    if (gray.empty()) { message = "input gray image is empty"; return false; }
    if (lines.empty()) { message = "line set is empty"; return false; }
    const int descDim = strips * 2 * 4;
    descriptors = cv::Mat::zeros(static_cast<int>(lines.size()), descDim, CV_32F);
    int validCount = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        std::vector<float> desc;
        if (computeMsldForLine(gray, lines[i], bandWidth, strips, desc)) {
            float* row = descriptors.ptr<float>(static_cast<int>(i));
            for (int j = 0; j < descDim; ++j) row[j] = desc[static_cast<size_t>(j)];
            ++validCount;
        }
    }
    if (validCount == 0) { message = "MSLD failed to compute any descriptors"; return false; }
    IR_LOG_INFO("MSLD computed ", validCount, " / ", lines.size(),
                " descriptors, dim=", descDim);
    return true;
}

bool computeLineSift(const cv::Mat& gray,
                     const std::vector<cv::Vec4i>& lines,
                     cv::Mat& descriptors,
                     int bandWidth, int strips, int bands, int bins,
                     std::string& message) {
    descriptors.release();
    if (gray.empty()) { message = "input gray image is empty"; return false; }
    if (lines.empty()) { message = "line set is empty"; return false; }
    const int descDim = strips * bands * bins;
    descriptors = cv::Mat::zeros(static_cast<int>(lines.size()), descDim, CV_32F);
    int validCount = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        std::vector<float> desc;
        if (computeLineSiftForLine(gray, lines[i], bandWidth, strips, bands, bins, desc)) {
            float* row = descriptors.ptr<float>(static_cast<int>(i));
            for (int j = 0; j < descDim; ++j) row[j] = desc[static_cast<size_t>(j)];
            ++validCount;
        }
    }
    if (validCount == 0) { message = "line-SIFT failed to compute any descriptors"; return false; }
    IR_LOG_INFO("line-SIFT computed ", validCount, " / ", lines.size(),
                " descriptors, dim=", descDim);
    return true;
}

/// 构造线描述子关联器。
LineDescriptorAssociator::LineDescriptorAssociator(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    _descriptor = yaml_utils::getString(params, "descriptor", "LBD");
    _matcher = yaml_utils::getString(params, "matcher", "BF");
    _knnK = yaml_utils::getInt(params, "knn_k", 8);
    _minMatches = yaml_utils::getInt(params, "min_matches", 2);
    _geometricFilter = yaml_utils::getBool(params, "geometric_filter", true);
    _angleThresholdDeg = yaml_utils::getDouble(params, "angle_threshold_deg", 30.0);
    _minLengthRatio = yaml_utils::getDouble(params, "min_length_ratio", 0.30);
    _shiftConsistencyThreshold =
        yaml_utils::getDouble(params, "shift_consistency_threshold", 30.0);
    _msldBandWidth = yaml_utils::getInt(params, "msld_band_width", 12);
    _msldStrips = yaml_utils::getInt(params, "msld_strips", 9);
    _lineSiftBandWidth = yaml_utils::getInt(params, "linesift_band_width", 20);
    _lineSiftStrips = yaml_utils::getInt(params, "linesift_strips", 4);
    _lineSiftBands = yaml_utils::getInt(params, "linesift_bands", 4);
    _lineSiftBins = yaml_utils::getInt(params, "linesift_bins", 8);

    IR_LOG_INFO("LineDescriptorAssociator: descriptor=",
                _descriptor,
                ", matcher=", _matcher,
                ", knnK=", _knnK,
                ", minMatches=", _minMatches,
                ", geometricFilter=", _geometricFilter,
                ", angleThrDeg=", _angleThresholdDeg,
                ", minLenRatio=", _minLengthRatio,
                ", shiftThr=", _shiftConsistencyThreshold);
    if (_descriptor == "MSLD") {
        IR_LOG_INFO("LineDescriptorAssociator MSLD: bandWidth=",
                    _msldBandWidth, ", strips=", _msldStrips);
    }
    if (_descriptor == "LINE_SIFT") {
        IR_LOG_INFO("LineDescriptorAssociator line-SIFT: bandWidth=",
                    _lineSiftBandWidth, ", strips=", _lineSiftStrips,
                    ", bands=", _lineSiftBands, ", bins=", _lineSiftBins);
    }
}

/// 执行描述子计算（LBD/MSLD）、KNN 匹配与几何一致性筛选。
bool LineDescriptorAssociator::associate(RegistrationContext& ctx) {
    auto& md = ctx.structure_match_data;
    md.clear();
    md.method = name();

    // 1. 校验输入结构类型。
    if (ctx.structure_data.type != StructureType::LINE) {
        md.message = "LineDescriptorAssociator requires LINE structure data";
        return false;
    }

    const std::string descriptor = upperAscii(_descriptor);
    const bool isLbd = (descriptor == "LBD");
    const bool isFloat = (descriptor == "MSLD" || descriptor == "LINE_SIFT");

    if (!isLbd && !isFloat) {
        md.message = "unsupported line descriptor: " + _descriptor;
        return false;
    }

    const std::vector<cv::Vec4i>& srcLines = ctx.structure_data.first.lines;
    const std::vector<cv::Vec4i>& dstLines = ctx.structure_data.second.lines;
    IR_LOG_INFO("LineDescriptorAssociator input: srcLines=", srcLines.size(),
                ", dstLines=", dstLines.size(), ", descriptor=", descriptor);

    // 2. 构建 KeyLine 索引（LBD 需要完整 KeyLine 做描述子计算，MSLD 仅需 class_id 映射）
    std::vector<cv::line_descriptor::KeyLine> srcKeys;
    std::vector<cv::line_descriptor::KeyLine> dstKeys;
    if (isLbd) {
        srcKeys = toKeyLines(srcLines);
        dstKeys = toKeyLines(dstLines);
    } else {
        srcKeys.reserve(srcLines.size());
        for (size_t i = 0; i < srcLines.size(); ++i) {
            cv::line_descriptor::KeyLine kl;
            kl.class_id = static_cast<int>(i);
            srcKeys.push_back(kl);
        }
        dstKeys.reserve(dstLines.size());
        for (size_t i = 0; i < dstLines.size(); ++i) {
            cv::line_descriptor::KeyLine kl;
            kl.class_id = static_cast<int>(i);
            dstKeys.push_back(kl);
        }
    }

    // 3. 计算描述子
    cv::Mat srcDescriptors;
    cv::Mat dstDescriptors;

    if (isLbd) {
#ifndef IR_HAS_OPENCV_LINE_DESCRIPTOR
        md.message =
            "OpenCV line_descriptor module not available; rebuild with contrib to enable LBD";
        IR_LOG_ERROR("LineDescriptorAssociator: ", md.message);
        return false;
#else
        if (!computeLbd(ctx.images.first_gray, srcKeys, srcDescriptors, md.message) ||
            !computeLbd(ctx.images.second_gray, dstKeys, dstDescriptors, md.message)) {
            IR_LOG_WARN("LineDescriptorAssociator LBD failed: ", md.message);
            return false;
        }
        IR_LOG_INFO("LineDescriptorAssociator LBD: srcDesc=", srcDescriptors.rows,
                    "x", srcDescriptors.cols, " (keys=", srcKeys.size(), ")",
                    ", dstDesc=", dstDescriptors.rows,
                    "x", dstDescriptors.cols, " (keys=", dstKeys.size(), ")");
#endif
    } else if (descriptor == "MSLD") {
        if (!computeMsld(ctx.images.first_gray, srcLines, srcDescriptors,
                         _msldBandWidth, _msldStrips, md.message) ||
            !computeMsld(ctx.images.second_gray, dstLines, dstDescriptors,
                         _msldBandWidth, _msldStrips, md.message)) {
            IR_LOG_WARN("LineDescriptorAssociator MSLD failed: ", md.message);
            return false;
        }
        IR_LOG_INFO("LineDescriptorAssociator MSLD: srcDesc=", srcDescriptors.rows,
                    "x", srcDescriptors.cols,
                    ", dstDesc=", dstDescriptors.rows, "x", dstDescriptors.cols);
    } else if (descriptor == "LINE_SIFT") {
        if (!computeLineSift(ctx.images.first_gray, srcLines, srcDescriptors,
                             _lineSiftBandWidth, _lineSiftStrips,
                             _lineSiftBands, _lineSiftBins, md.message) ||
            !computeLineSift(ctx.images.second_gray, dstLines, dstDescriptors,
                             _lineSiftBandWidth, _lineSiftStrips,
                             _lineSiftBands, _lineSiftBins, md.message)) {
            IR_LOG_WARN("LineDescriptorAssociator line-SIFT failed: ", md.message);
            return false;
        }
        IR_LOG_INFO("LineDescriptorAssociator line-SIFT: srcDesc=", srcDescriptors.rows,
                    "x", srcDescriptors.cols,
                    ", dstDesc=", dstDescriptors.rows, "x", dstDescriptors.cols);
    }

    // 4. KNN 匹配：二进制→BinaryDescriptorMatcher，浮点→BFMatcher L2

    const bool useFlann = (upperAscii(_matcher) == "FLANN") && !isLbd;
    md.raw_matches_knn =
        knnMatchDescriptors(srcDescriptors, dstDescriptors, srcKeys, dstKeys,
                            _knnK, isLbd, useFlann);
    IR_LOG_INFO("LineDescriptorAssociator KNN: raw_matches_knn groups=",
                md.raw_matches_knn.size());
    if (md.raw_matches_knn.empty()) {
        md.message = "KNN matching produced no matches";
        IR_LOG_WARN("LineDescriptorAssociator rejected match: ", md.message);
        return false;
    }

    // 4. 展开 KNN 为扁平候选列表
    std::vector<cv::DMatch> rawFlat;
    for (const auto& nb : md.raw_matches_knn) {
        for (const auto& m : nb)
            rawFlat.push_back(m);
    }

    // 5. 几何一致性筛选
    std::vector<cv::DMatch> selected;
    if (_geometricFilter) {
        selected = filterGeometricConsistent(rawFlat, srcLines, dstLines,
                                             _angleThresholdDeg,
                                             _minLengthRatio,
                                             _shiftConsistencyThreshold);
    } else {
        selected.reserve(md.raw_matches_knn.size());
        std::vector<unsigned char> usedT(dstLines.size(), 0);
        for (const auto& nb : md.raw_matches_knn) {
            for (const auto& m : nb) {
                if (m.trainIdx >= 0 && m.trainIdx < static_cast<int>(usedT.size()) &&
                    !usedT[static_cast<size_t>(m.trainIdx)]) {
                    usedT[static_cast<size_t>(m.trainIdx)] = 1;
                    selected.push_back(m);
                    break;
                }
            }
        }
    }

    // 6. 输出
    md.filtered_matches = selected;
    md.line_matches = md.filtered_matches;
    md.inlier_line_matches = md.line_matches;
    md.valid = static_cast<int>(md.line_matches.size()) >= _minMatches;
    md.score = srcLines.empty()
                   ? 0.0
                   : static_cast<double>(md.line_matches.size()) /
                         static_cast<double>(srcLines.size());

    if (!md.valid) {
        md.message = "not enough " + descriptor + " matches after filtering: " +
                     std::to_string(md.line_matches.size()) +
                     " (min=" + std::to_string(_minMatches) + ")";
        IR_LOG_WARN("LineDescriptorAssociator rejected match: ", md.message);
        return false;
    }

    IR_LOG_INFO("LineDescriptorAssociator produced ", descriptor, " matches: ",
                md.line_matches.size(), " / ", rawFlat.size(), " raw",
                ", geometricFilter=", _geometricFilter,
                ", srcLines=", srcLines.size(),
                ", dstLines=", dstLines.size());
    return true;
}

} // namespace ir
