#include "matcher/structure/contour_descriptor_associator.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include "core/types.h"
#include "utils/logger.h"
#include "utils/string_utils.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

constexpr float kPiF = 3.14159265358979323846f;

cv::Point2d contourCentroid(const std::vector<cv::Point>& contour) {
    cv::Moments m = cv::moments(contour);
    if (m.m00 < 1e-9) {
        return {0.0, 0.0};
    }
    return {m.m10 / m.m00, m.m01 / m.m00};
}

double contourAreaVal(const std::vector<cv::Point>& contour) {
    return cv::contourArea(contour);
}

double contourPerimeter(const std::vector<cv::Point>& contour) {
    return cv::arcLength(contour, true);
}

bool isValidContourForDescriptor(const std::vector<cv::Point>& contour) {
    return contour.size() >= 3 && contourAreaVal(contour) > 0.0 &&
           contourPerimeter(contour) > 1e-6;
}

// 按弧长将轮廓重采样为固定点数，供 FD / Shape Context 这类轮廓级描述子复用。
std::vector<cv::Point2f> resampleContour(const std::vector<cv::Point>& contour,
                                         int samplePoints) {
    std::vector<cv::Point2f> sampled;
    if (samplePoints < 3 || contour.size() < 2) {
        return sampled;
    }

    std::vector<cv::Point2f> pts;
    pts.reserve(contour.size());
    for (const auto& p : contour) {
        pts.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
    }
    if (pts.front() != pts.back()) {
        pts.push_back(pts.front());
    }
    if (pts.size() < 2) {
        return sampled;
    }

    std::vector<float> cumulative(pts.size(), 0.0f);
    for (size_t i = 1; i < pts.size(); ++i) {
        cumulative[i] = cumulative[i - 1] + cv::norm(pts[i] - pts[i - 1]);
    }
    const float totalLength = cumulative.back();
    if (totalLength <= 1e-6f) {
        return sampled;
    }

    sampled.reserve(static_cast<size_t>(samplePoints));
    const float step = totalLength / static_cast<float>(samplePoints);
    size_t segment = 1;
    for (int i = 0; i < samplePoints; ++i) {
        const float target = step * static_cast<float>(i);
        while (segment < cumulative.size() - 1 && cumulative[segment] < target) {
            ++segment;
        }
        const float prevLen = cumulative[segment - 1];
        const float nextLen = cumulative[segment];
        const float denom = std::max(1e-6f, nextLen - prevLen);
        const float alpha = (target - prevLen) / denom;
        sampled.push_back(pts[segment - 1] + alpha * (pts[segment] - pts[segment - 1]));
    }
    return sampled;
}

// 为单条采样轮廓构建简化版 Shape Context 直方图描述子。
cv::Mat computeShapeContextForSampledContour(const std::vector<cv::Point2f>& points,
                                             int radialBins,
                                             int angularBins,
                                             float innerRadius,
                                             float outerRadius) {
    const int descriptorDim = radialBins * angularBins;
    cv::Mat descriptor = cv::Mat::zeros(1, descriptorDim, CV_32F);
    if (points.size() < 3 || descriptorDim <= 0 || innerRadius <= 0.0f ||
        outerRadius <= innerRadius) {
        return descriptor;
    }

    float meanDistance = 0.0f;
    int distanceCount = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = i + 1; j < points.size(); ++j) {
            const float d = cv::norm(points[i] - points[j]);
            if (d > 1e-6f) {
                meanDistance += d;
                ++distanceCount;
            }
        }
    }
    if (distanceCount == 0) {
        return descriptor;
    }
    meanDistance /= static_cast<float>(distanceCount);
    if (meanDistance <= 1e-6f) {
        return descriptor;
    }

    const float logInner = std::log(innerRadius);
    const float logOuter = std::log(outerRadius);
    const float logSpan = std::max(1e-6f, logOuter - logInner);

    for (size_t i = 0; i < points.size(); ++i) {
        for (size_t j = 0; j < points.size(); ++j) {
            if (i == j) {
                continue;
            }
            const cv::Point2f delta = points[j] - points[i];
            const float distance = cv::norm(delta) / meanDistance;
            if (distance < innerRadius || distance > outerRadius) {
                continue;
            }

            const float logDistance = std::log(std::max(distance, innerRadius));
            int radialBin = static_cast<int>(std::floor(
                (logDistance - logInner) / logSpan * static_cast<float>(radialBins)));
            radialBin = std::clamp(radialBin, 0, radialBins - 1);

            float angle = std::atan2(delta.y, delta.x);
            if (angle < 0.0f) {
                angle += 2.0f * kPiF;
            }
            int angularBin = static_cast<int>(
                std::floor(angle / (2.0f * kPiF) * static_cast<float>(angularBins)));
            angularBin = std::clamp(angularBin, 0, angularBins - 1);

            descriptor.at<float>(0, radialBin * angularBins + angularBin) += 1.0f;
        }
    }

    const float normalizer = static_cast<float>(points.size());
    if (normalizer > 0.0f) {
        descriptor /= normalizer;
    }
    cv::normalize(descriptor, descriptor);
    return descriptor;
}

// 为单条采样轮廓构建 Fourier Descriptor。
// 步骤：去质心 -> 尺度归一化 -> DFT -> 取低频幅值。
cv::Mat computeFourierDescriptorForSampledContour(const std::vector<cv::Point2f>& points,
                                                  int coefficientCount) {
    cv::Mat descriptor;
    if (points.size() < 8 || coefficientCount <= 0) {
        return descriptor;
    }

    cv::Point2f centroid(0.0f, 0.0f);
    for (const auto& p : points) {
        centroid += p;
    }
    centroid *= (1.0f / static_cast<float>(points.size()));

    cv::Mat sequence(static_cast<int>(points.size()), 1, CV_32FC2);
    float energy = 0.0f;
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        const cv::Point2f centered = points[static_cast<size_t>(i)] - centroid;
        sequence.at<cv::Vec2f>(i, 0) = cv::Vec2f(centered.x, centered.y);
        energy += centered.x * centered.x + centered.y * centered.y;
    }

    const float scale = std::sqrt(energy / static_cast<float>(points.size()));
    if (scale <= 1e-6f) {
        return descriptor;
    }
    sequence /= scale;

    cv::Mat spectrum;
    cv::dft(sequence, spectrum, cv::DFT_COMPLEX_OUTPUT);

    const int usableCoefficients =
        std::min(coefficientCount, std::max(0, static_cast<int>(points.size()) / 2));
    if (usableCoefficients <= 0) {
        return descriptor;
    }

    descriptor = cv::Mat::zeros(1, usableCoefficients, CV_32F);
    for (int k = 0; k < usableCoefficients; ++k) {
        const cv::Vec2f coeff = spectrum.at<cv::Vec2f>(k + 1, 0);
        descriptor.at<float>(0, k) = std::sqrt(coeff[0] * coeff[0] + coeff[1] * coeff[1]);
    }

    const double normValue = cv::norm(descriptor);
    if (!(normValue > 1e-9)) {
        descriptor.release();
        return descriptor;
    }
    descriptor /= normValue;
    return descriptor;
}

// 为单条闭合轮廓计算 EFD：
// 1. 视轮廓为按弧长参数化的闭合曲线。
// 2. 分别对 x(t)、y(t) 的一阶差分做椭圆傅里叶展开。
// 3. 将每个谐波的 a/b/c/d 系数拼接成描述子，并做可选归一化。
cv::Mat computeEfdForContour(const std::vector<cv::Point>& contour,
                             int harmonics,
                             bool normalizeRotation,
                             bool normalizeScale) {
    cv::Mat descriptor;
    if (harmonics <= 0 || contour.size() < 3) {
        return descriptor;
    }

    std::vector<cv::Point2f> pts;
    pts.reserve(contour.size() + 1);
    for (const auto& p : contour) {
        pts.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
    }
    if (pts.front() != pts.back()) {
        pts.push_back(pts.front());
    }
    if (pts.size() < 4) {
        return descriptor;
    }

    const size_t segments = pts.size() - 1;
    std::vector<double> dt(segments, 0.0);
    std::vector<double> cumulative(segments + 1, 0.0);
    for (size_t i = 0; i < segments; ++i) {
        dt[i] = static_cast<double>(cv::norm(pts[i + 1] - pts[i]));
        if (dt[i] <= 1e-9) {
            return descriptor;
        }
        cumulative[i + 1] = cumulative[i] + dt[i];
    }

    const double period = cumulative.back();
    if (period <= 1e-9) {
        return descriptor;
    }

    descriptor = cv::Mat::zeros(1, harmonics * 4, CV_32F);
    for (int n = 1; n <= harmonics; ++n) {
        const double coeff = period / (2.0 * kPiF * kPiF * n * n);
        double an = 0.0;
        double bn = 0.0;
        double cn = 0.0;
        double dn = 0.0;

        for (size_t i = 0; i < segments; ++i) {
            const double dx =
                static_cast<double>(pts[i + 1].x) - static_cast<double>(pts[i].x);
            const double dy =
                static_cast<double>(pts[i + 1].y) - static_cast<double>(pts[i].y);
            const double invDt = 1.0 / dt[i];
            const double t0 = cumulative[i];
            const double t1 = cumulative[i + 1];
            const double angle0 = 2.0 * kPiF * n * t0 / period;
            const double angle1 = 2.0 * kPiF * n * t1 / period;
            const double cosDiff = std::cos(angle1) - std::cos(angle0);
            const double sinDiff = std::sin(angle1) - std::sin(angle0);

            an += dx * invDt * cosDiff;
            bn += dx * invDt * sinDiff;
            cn += dy * invDt * cosDiff;
            dn += dy * invDt * sinDiff;
        }

        an *= coeff;
        bn *= coeff;
        cn *= coeff;
        dn *= coeff;

        descriptor.at<float>(0, (n - 1) * 4 + 0) = static_cast<float>(an);
        descriptor.at<float>(0, (n - 1) * 4 + 1) = static_cast<float>(bn);
        descriptor.at<float>(0, (n - 1) * 4 + 2) = static_cast<float>(cn);
        descriptor.at<float>(0, (n - 1) * 4 + 3) = static_cast<float>(dn);
    }

    if (normalizeRotation) {
        const float a1 = descriptor.at<float>(0, 0);
        const float b1 = descriptor.at<float>(0, 1);
        const float c1 = descriptor.at<float>(0, 2);
        const float d1 = descriptor.at<float>(0, 3);
        const double theta = 0.5 * std::atan2(
            2.0 * (static_cast<double>(a1) * b1 + static_cast<double>(c1) * d1),
            static_cast<double>(a1) * a1 + static_cast<double>(c1) * c1 -
                static_cast<double>(b1) * b1 - static_cast<double>(d1) * d1);
        const float cosTheta = static_cast<float>(std::cos(theta));
        const float sinTheta = static_cast<float>(std::sin(theta));

        for (int n = 0; n < harmonics; ++n) {
            const float an = descriptor.at<float>(0, n * 4 + 0);
            const float bn = descriptor.at<float>(0, n * 4 + 1);
            const float cn = descriptor.at<float>(0, n * 4 + 2);
            const float dn = descriptor.at<float>(0, n * 4 + 3);
            descriptor.at<float>(0, n * 4 + 0) = an * cosTheta + bn * sinTheta;
            descriptor.at<float>(0, n * 4 + 1) = -an * sinTheta + bn * cosTheta;
            descriptor.at<float>(0, n * 4 + 2) = cn * cosTheta + dn * sinTheta;
            descriptor.at<float>(0, n * 4 + 3) = -cn * sinTheta + dn * cosTheta;
        }
    }

    if (normalizeScale) {
        const double base = std::sqrt(
            static_cast<double>(descriptor.at<float>(0, 0)) * descriptor.at<float>(0, 0) +
            static_cast<double>(descriptor.at<float>(0, 1)) * descriptor.at<float>(0, 1) +
            static_cast<double>(descriptor.at<float>(0, 2)) * descriptor.at<float>(0, 2) +
            static_cast<double>(descriptor.at<float>(0, 3)) * descriptor.at<float>(0, 3));
        if (!(base > 1e-9)) {
            descriptor.release();
            return descriptor;
        }
        descriptor /= base;
    }

    const double normValue = cv::norm(descriptor);
    if (!(normValue > 1e-9)) {
        descriptor.release();
        return descriptor;
    }
    descriptor /= normValue;
    return descriptor;
}

// 将描述子矩阵中的行号映射回原始 contour 下标。
std::vector<std::vector<cv::DMatch>> remapContourMatches(
    const std::vector<std::vector<cv::DMatch>>& rawMatches,
    const std::vector<int>& srcIndices,
    const std::vector<int>& dstIndices) {

    std::vector<std::vector<cv::DMatch>> remapped;
    remapped.reserve(rawMatches.size());
    for (const auto& neighbours : rawMatches) {
        std::vector<cv::DMatch> mapped;
        mapped.reserve(neighbours.size());
        for (const auto& m : neighbours) {
            if (m.queryIdx < 0 || m.trainIdx < 0 ||
                m.queryIdx >= static_cast<int>(srcIndices.size()) ||
                m.trainIdx >= static_cast<int>(dstIndices.size())) {
                continue;
            }
            mapped.emplace_back(srcIndices[static_cast<size_t>(m.queryIdx)],
                                dstIndices[static_cast<size_t>(m.trainIdx)],
                                m.distance);
        }
        if (!mapped.empty()) {
            remapped.push_back(std::move(mapped));
        }
    }
    return remapped;
}

// 统一封装 BF 匹配入口，供 HU / FD / Shape Context 共用。
std::vector<std::vector<cv::DMatch>> matchContourDescriptors(const cv::Mat& srcDesc,
                                                             const cv::Mat& dstDesc,
                                                             const std::string& mode,
                                                             int knnK,
                                                             float radius) {
    std::vector<std::vector<cv::DMatch>> raw;
    const std::string effectiveMode = mode.empty() ? "KNN" : mode;
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, false);

    if (effectiveMode == "MATCH") {
        std::vector<cv::DMatch> tmp;
        matcher->match(srcDesc, dstDesc, tmp);
        raw.reserve(tmp.size());
        for (auto& x : tmp) {
            raw.push_back({x});
        }
    } else if (effectiveMode == "RADIUS") {
        matcher->radiusMatch(srcDesc, dstDesc, raw, std::max(1.0f, radius));
    } else {
        matcher->knnMatch(srcDesc, dstDesc, raw, std::max(1, knnK));
    }
    return raw;
}

std::vector<cv::DMatch> filterUniqueByDistance(const std::vector<cv::DMatch>& matches,
                                               size_t srcCount,
                                               size_t dstCount) {
    std::vector<cv::DMatch> sorted = matches;
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const cv::DMatch& a, const cv::DMatch& b) {
                         return a.distance < b.distance;
                     });

    std::vector<cv::DMatch> unique;
    unique.reserve(sorted.size());
    std::vector<unsigned char> usedQ(srcCount, 0);
    std::vector<unsigned char> usedT(dstCount, 0);
    for (const auto& m : sorted) {
        const size_t q = static_cast<size_t>(m.queryIdx);
        const size_t t = static_cast<size_t>(m.trainIdx);
        if (q >= usedQ.size() || t >= usedT.size() || usedQ[q] || usedT[t]) {
            continue;
        }
        usedQ[q] = 1;
        usedT[t] = 1;
        unique.push_back(m);
    }
    return unique;
}

std::vector<cv::DMatch> filterContourTranslation(
    const std::vector<cv::DMatch>& candidates,
    const std::vector<std::vector<cv::Point>>& srcContours,
    const std::vector<std::vector<cv::Point>>& dstContours,
    double shiftThreshold) {

    if (candidates.empty()) {
        return {};
    }

    int bestCount = 0;
    double bestCost = 0.0;
    cv::Point2d bestShift(0.0, 0.0);
    for (const auto& seed : candidates) {
        const cv::Point2d c1 = contourCentroid(srcContours[static_cast<size_t>(seed.queryIdx)]);
        const cv::Point2d c2 = contourCentroid(dstContours[static_cast<size_t>(seed.trainIdx)]);
        const cv::Point2d shift = c2 - c1;

        int count = 0;
        double cost = 0.0;
        for (const auto& m : candidates) {
            const cv::Point2d s1 = contourCentroid(srcContours[static_cast<size_t>(m.queryIdx)]);
            const cv::Point2d s2 = contourCentroid(dstContours[static_cast<size_t>(m.trainIdx)]);
            const double err = cv::norm((s2 - s1) - shift);
            if (err <= shiftThreshold) {
                ++count;
                cost += err;
            }
        }
        if (count > bestCount || (count == bestCount && cost < bestCost)) {
            bestCount = count;
            bestCost = cost;
            bestShift = shift;
        }
    }

    std::vector<cv::DMatch> filtered;
    for (const auto& m : candidates) {
        const cv::Point2d s1 = contourCentroid(srcContours[static_cast<size_t>(m.queryIdx)]);
        const cv::Point2d s2 = contourCentroid(dstContours[static_cast<size_t>(m.trainIdx)]);
        if (cv::norm((s2 - s1) - bestShift) <= shiftThreshold) {
            filtered.push_back(m);
        }
    }
    return filtered;
}

// 更适合旋转 + 平移场景的预筛选：
// 先用描述子候选的轮廓质心构造点对，再用刚体 RANSAC 保留几何一致的内点。
std::vector<cv::DMatch> filterContourRigid(const std::vector<cv::DMatch>& candidates,
                                           const std::vector<std::vector<cv::Point>>& srcContours,
                                           const std::vector<std::vector<cv::Point>>& dstContours,
                                           double reprojThreshold,
                                           int maxIterations,
                                           int minInliers) {
    if (candidates.size() < 2) {
        return {};
    }

    std::vector<cv::Point2f> srcPts;
    std::vector<cv::Point2f> dstPts;
    srcPts.reserve(candidates.size());
    dstPts.reserve(candidates.size());
    for (const auto& m : candidates) {
        const cv::Point2d c1 = contourCentroid(srcContours[static_cast<size_t>(m.queryIdx)]);
        const cv::Point2d c2 = contourCentroid(dstContours[static_cast<size_t>(m.trainIdx)]);
        srcPts.emplace_back(static_cast<float>(c1.x), static_cast<float>(c1.y));
        dstPts.emplace_back(static_cast<float>(c2.x), static_cast<float>(c2.y));
    }

    std::vector<unsigned char> mask;
    cv::Mat rigid = cv::estimateAffinePartial2D(srcPts,
                                                dstPts,
                                                mask,
                                                cv::RANSAC,
                                                reprojThreshold,
                                                static_cast<size_t>(std::max(1, maxIterations)),
                                                0.99,
                                                10);
    if (rigid.empty()) {
        return {};
    }

    std::vector<cv::DMatch> filtered;
    filtered.reserve(candidates.size());
    for (size_t i = 0; i < candidates.size() && i < mask.size(); ++i) {
        if (mask[i]) {
            filtered.push_back(candidates[i]);
        }
    }
    if (static_cast<int>(filtered.size()) < minInliers) {
        return {};
    }
    return filtered;
}

// 先做面积比预筛，再根据几何模型做轮廓匹配预筛。
// RIGID 模式保留 KNN/RADIUS 的多候选，让刚体一致性选择正确候选，避免过早 top-1 去重。
std::vector<cv::DMatch> filterContourGeometric(
    const std::vector<cv::DMatch>& raw,
    const std::vector<std::vector<cv::Point>>& srcContours,
    const std::vector<std::vector<cv::Point>>& dstContours,
    double areaRatioMin,
    const std::string& geometricModel,
    double shiftThreshold,
    double rigidReprojThreshold,
    int rigidRansacIterations,
    int rigidMinInliers) {

    std::vector<cv::DMatch> candidates;
    candidates.reserve(raw.size());
    for (const auto& m : raw) {
        if (m.queryIdx < 0 || m.trainIdx < 0 ||
            m.queryIdx >= static_cast<int>(srcContours.size()) ||
            m.trainIdx >= static_cast<int>(dstContours.size())) {
            continue;
        }
        const double a1 = contourAreaVal(srcContours[static_cast<size_t>(m.queryIdx)]);
        const double a2 = contourAreaVal(dstContours[static_cast<size_t>(m.trainIdx)]);
        if (a1 <= 0.0 || a2 <= 0.0) {
            continue;
        }
        const double ratio = std::min(a1, a2) / std::max(a1, a2);
        if (ratio < areaRatioMin) {
            continue;
        }
        candidates.push_back(m);
    }
    if (candidates.empty()) {
        return {};
    }

    if (string_utils::toUpperAscii(geometricModel) == "TRANSLATION") {
        const std::vector<cv::DMatch> uniqueCandidates =
            filterUniqueByDistance(candidates, srcContours.size(), dstContours.size());
        if (uniqueCandidates.empty()) {
            return {};
        }
        return filterUniqueByDistance(
            filterContourTranslation(uniqueCandidates, srcContours, dstContours, shiftThreshold),
            srcContours.size(),
            dstContours.size());
    }

    const std::vector<cv::DMatch> rigidFiltered = filterContourRigid(candidates,
                                                                     srcContours,
                                                                     dstContours,
                                                                     rigidReprojThreshold,
                                                                     rigidRansacIterations,
                                                                     rigidMinInliers);
    if (!rigidFiltered.empty()) {
        return filterUniqueByDistance(rigidFiltered, srcContours.size(), dstContours.size());
    }

    const std::vector<cv::DMatch> uniqueCandidates =
        filterUniqueByDistance(candidates, srcContours.size(), dstContours.size());
    if (uniqueCandidates.empty()) {
        return {};
    }

    // 刚体预筛失败时回退到平移一致性策略，避免直接清空所有候选。
    return filterUniqueByDistance(
        filterContourTranslation(uniqueCandidates, srcContours, dstContours, shiftThreshold),
        srcContours.size(),
        dstContours.size());
}
} // namespace

bool computeHuMoments(const std::vector<std::vector<cv::Point>>& contours,
                      cv::Mat& descriptors,
                      std::string& message,
                      std::vector<int>* contourIndices) {
    descriptors.release();
    if (contourIndices) {
        contourIndices->clear();
    }
    if (contours.empty()) {
        message = "contour set is empty";
        return false;
    }

    std::vector<cv::Mat> rows;
    rows.reserve(contours.size());
    int validCount = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (!isValidContourForDescriptor(contours[i])) {
            continue;
        }
        cv::Moments m = cv::moments(contours[i]);
        std::vector<double> hu(7);
        cv::HuMoments(m, hu);

        cv::Mat row = cv::Mat::zeros(1, 7, CV_32F);
        for (int j = 0; j < 7; ++j) {
            const double v = std::abs(hu[j]);
            row.at<float>(0, j) = static_cast<float>(v > 1e-10 ? std::log10(v) : -10.0);
        }
        rows.push_back(row);
        if (contourIndices) {
            contourIndices->push_back(static_cast<int>(i));
        }
        ++validCount;
    }

    if (rows.empty()) {
        message = "no valid contours for HuMoments";
        return false;
    }
    cv::vconcat(rows, descriptors);
    IR_LOG_INFO("HuMoments computed ", validCount, " / ", contours.size(), " descriptors");
    return true;
}

bool computeFourierDescriptor(const std::vector<std::vector<cv::Point>>& contours,
                              cv::Mat& descriptors,
                              int samplePoints,
                              int coefficientCount,
                              std::string& message,
                              std::vector<int>* contourIndices) {
    descriptors.release();
    if (contourIndices) {
        contourIndices->clear();
    }
    if (contours.empty()) {
        message = "contour set is empty";
        return false;
    }
    if (samplePoints < 16 || coefficientCount <= 0) {
        message = "invalid Fourier Descriptor parameters";
        return false;
    }

    std::vector<cv::Mat> rows;
    rows.reserve(contours.size());
    int validCount = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (!isValidContourForDescriptor(contours[i])) {
            continue;
        }
        const std::vector<cv::Point2f> sampled = resampleContour(contours[i], samplePoints);
        if (sampled.size() < static_cast<size_t>(samplePoints)) {
            continue;
        }
        cv::Mat row = computeFourierDescriptorForSampledContour(sampled, coefficientCount);
        if (row.empty()) {
            continue;
        }
        rows.push_back(row);
        if (contourIndices) {
            contourIndices->push_back(static_cast<int>(i));
        }
        ++validCount;
    }

    if (rows.empty()) {
        message = "no valid contours for Fourier Descriptor";
        return false;
    }
    cv::vconcat(rows, descriptors);
    IR_LOG_INFO("FourierDescriptor computed ", validCount, " / ", contours.size(),
                " descriptors, dim=", descriptors.cols);
    return true;
}

bool computeShapeContext(const std::vector<std::vector<cv::Point>>& contours,
                         cv::Mat& descriptors,
                         int samplePoints,
                         int radialBins,
                         int angularBins,
                         float innerRadius,
                         float outerRadius,
                         std::string& message,
                         std::vector<int>* contourIndices) {
    descriptors.release();
    if (contourIndices) {
        contourIndices->clear();
    }
    if (contours.empty()) {
        message = "contour set is empty";
        return false;
    }
    if (samplePoints < 8 || radialBins < 2 || angularBins < 4 || innerRadius <= 0.0f ||
        outerRadius <= innerRadius) {
        message = "invalid Shape Context parameters";
        return false;
    }

    std::vector<cv::Mat> rows;
    rows.reserve(contours.size());
    int validCount = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (!isValidContourForDescriptor(contours[i])) {
            continue;
        }
        const std::vector<cv::Point2f> sampled = resampleContour(contours[i], samplePoints);
        if (sampled.size() < static_cast<size_t>(samplePoints)) {
            continue;
        }
        cv::Mat row = computeShapeContextForSampledContour(sampled,
                                                           radialBins,
                                                           angularBins,
                                                           innerRadius,
                                                           outerRadius);
        if (row.empty()) {
            continue;
        }
        rows.push_back(row);
        if (contourIndices) {
            contourIndices->push_back(static_cast<int>(i));
        }
        ++validCount;
    }

    if (rows.empty()) {
        message = "no valid contours for Shape Context";
        return false;
    }
    cv::vconcat(rows, descriptors);
    IR_LOG_INFO("ShapeContext computed ", validCount, " / ", contours.size(),
                " descriptors, dim=", descriptors.cols);
    return true;
}

bool computeEllipticFourierDescriptor(const std::vector<std::vector<cv::Point>>& contours,
                                      cv::Mat& descriptors,
                                      int harmonics,
                                      bool normalizeRotation,
                                      bool normalizeScale,
                                      std::string& message,
                                      std::vector<int>* contourIndices) {
    descriptors.release();
    if (contourIndices) {
        contourIndices->clear();
    }
    if (contours.empty()) {
        message = "contour set is empty";
        return false;
    }
    if (harmonics <= 0) {
        message = "invalid EFD harmonic count";
        return false;
    }

    std::vector<cv::Mat> rows;
    rows.reserve(contours.size());
    int validCount = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        if (!isValidContourForDescriptor(contours[i])) {
            continue;
        }
        cv::Mat row = computeEfdForContour(
            contours[i], harmonics, normalizeRotation, normalizeScale);
        if (row.empty()) {
            continue;
        }
        rows.push_back(row);
        if (contourIndices) {
            contourIndices->push_back(static_cast<int>(i));
        }
        ++validCount;
    }

    if (rows.empty()) {
        message = "no valid contours for EFD";
        return false;
    }
    cv::vconcat(rows, descriptors);
    IR_LOG_INFO("EllipticFourierDescriptor computed ", validCount, " / ", contours.size(),
                " descriptors, dim=", descriptors.cols);
    return true;
}

ContourDescriptorAssociator::ContourDescriptorAssociator(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    _descriptor = yaml_utils::getString(params, "descriptor", "HU");
    _matcher = yaml_utils::getString(params, "matcher", "BF");
    _matchMode = yaml_utils::getString(params, "match_mode", "KNN");
    _knnK = yaml_utils::getInt(params, "knn_k", 2);
    _matchRadius = yaml_utils::getFloat(params, "match_radius", 50.0f);
    _minMatches = yaml_utils::getInt(params, "min_matches", 2);
    _geometricFilter = yaml_utils::getBool(params, "geometric_filter", true);
    _areaRatioMin = yaml_utils::getDouble(params, "area_ratio_min", 0.30);
    _shiftConsistencyThreshold =
        yaml_utils::getDouble(params, "shift_consistency_threshold", 30.0);
    _geometricModel = yaml_utils::getString(params, "geometric_model", "RIGID");
    _rigidRansacThreshold = yaml_utils::getDouble(params, "rigid_ransac_threshold", 8.0);
    _rigidRansacIterations = yaml_utils::getInt(params, "rigid_ransac_iterations", 200);
    _rigidMinInliers = yaml_utils::getInt(params, "rigid_min_inliers", 2);
    _fourierSamplePoints = yaml_utils::getInt(params, "fourier_sample_points", 128);
    _fourierCoefficients = yaml_utils::getInt(params, "fourier_coefficients", 16);
    _efdHarmonics = yaml_utils::getInt(params, "efd_harmonics", 12);
    _efdNormalizeRotation = yaml_utils::getBool(params, "efd_normalize_rotation", true);
    _efdNormalizeScale = yaml_utils::getBool(params, "efd_normalize_scale", true);
    _shapeContextSamplePoints = yaml_utils::getInt(params, "shape_context_sample_points", 64);
    _shapeContextRadialBins = yaml_utils::getInt(params, "shape_context_radial_bins", 5);
    _shapeContextAngularBins = yaml_utils::getInt(params, "shape_context_angular_bins", 12);
    _shapeContextInnerRadius =
        yaml_utils::getFloat(params, "shape_context_inner_radius", 0.125f);
    _shapeContextOuterRadius =
        yaml_utils::getFloat(params, "shape_context_outer_radius", 2.0f);

    IR_LOG_INFO("ContourDescriptorAssociator: descriptor=", _descriptor,
                ", matcher=", _matcher,
                ", mode=", _matchMode,
                ", knnK=", _knnK,
                ", minMatches=", _minMatches,
                ", geometricFilter=", _geometricFilter,
                ", geometricModel=", _geometricModel,
                ", areaRatioMin=", _areaRatioMin,
                ", shiftThr=", _shiftConsistencyThreshold,
                ", rigidThr=", _rigidRansacThreshold);
    if (string_utils::toUpperAscii(_descriptor) == "FD") {
        IR_LOG_INFO("ContourDescriptorAssociator FourierDescriptor: samplePoints=",
                    _fourierSamplePoints,
                    ", coefficients=", _fourierCoefficients);
    }
    if (string_utils::toUpperAscii(_descriptor) == "EFD") {
        IR_LOG_INFO("ContourDescriptorAssociator EFD: harmonics=",
                    _efdHarmonics,
                    ", normalizeRotation=", _efdNormalizeRotation,
                    ", normalizeScale=", _efdNormalizeScale);
    }
    if (string_utils::toUpperAscii(_descriptor) == "SHAPE_CONTEXT") {
        IR_LOG_INFO("ContourDescriptorAssociator ShapeContext: samplePoints=",
                    _shapeContextSamplePoints,
                    ", radialBins=", _shapeContextRadialBins,
                    ", angularBins=", _shapeContextAngularBins,
                    ", innerRadius=", _shapeContextInnerRadius,
                    ", outerRadius=", _shapeContextOuterRadius);
    }
}

bool ContourDescriptorAssociator::associate(RegistrationContext& ctx) {
    auto& md = ctx.structure_match_data;
    md.clear();
    md.method = name();

    if (ctx.structure_data.type != StructureType::CONTOUR) {
        md.message = "ContourDescriptorAssociator requires CONTOUR structure data";
        return false;
    }

    const auto& srcContours = ctx.structure_data.first.contours;
    const auto& dstContours = ctx.structure_data.second.contours;
    IR_LOG_INFO("ContourDescriptorAssociator input: srcContours=", srcContours.size(),
                ", dstContours=", dstContours.size());

    const std::string descriptor = string_utils::toUpperAscii(_descriptor);
    const std::string mode = string_utils::toUpperAscii(_matchMode);

    if (descriptor == "HU" || descriptor == "SHAPE_CONTEXT" || descriptor == "FD" ||
        descriptor == "EFD") {
        cv::Mat srcDesc;
        cv::Mat dstDesc;
        std::vector<int> srcIndices;
        std::vector<int> dstIndices;

        bool ok = false;
        if (descriptor == "HU") {
            ok = computeHuMoments(srcContours, srcDesc, md.message, &srcIndices) &&
                 computeHuMoments(dstContours, dstDesc, md.message, &dstIndices);
        } else if (descriptor == "FD") {
            ok = computeFourierDescriptor(srcContours,
                                          srcDesc,
                                          _fourierSamplePoints,
                                          _fourierCoefficients,
                                          md.message,
                                          &srcIndices) &&
                 computeFourierDescriptor(dstContours,
                                          dstDesc,
                                          _fourierSamplePoints,
                                          _fourierCoefficients,
                                          md.message,
                                          &dstIndices);
        } else if (descriptor == "EFD") {
            ok = computeEllipticFourierDescriptor(srcContours,
                                                  srcDesc,
                                                  _efdHarmonics,
                                                  _efdNormalizeRotation,
                                                  _efdNormalizeScale,
                                                  md.message,
                                                  &srcIndices) &&
                 computeEllipticFourierDescriptor(dstContours,
                                                  dstDesc,
                                                  _efdHarmonics,
                                                  _efdNormalizeRotation,
                                                  _efdNormalizeScale,
                                                  md.message,
                                                  &dstIndices);
        } else {
            ok = computeShapeContext(srcContours,
                                     srcDesc,
                                     _shapeContextSamplePoints,
                                     _shapeContextRadialBins,
                                     _shapeContextAngularBins,
                                     _shapeContextInnerRadius,
                                     _shapeContextOuterRadius,
                                     md.message,
                                     &srcIndices) &&
                 computeShapeContext(dstContours,
                                     dstDesc,
                                     _shapeContextSamplePoints,
                                     _shapeContextRadialBins,
                                     _shapeContextAngularBins,
                                     _shapeContextInnerRadius,
                                     _shapeContextOuterRadius,
                                     md.message,
                                     &dstIndices);
        }
        if (!ok) {
            IR_LOG_WARN("ContourDescriptorAssociator ", descriptor, " failed: ", md.message);
            return false;
        }
        md.raw_matches_knn = remapContourMatches(
            matchContourDescriptors(srcDesc, dstDesc, mode, _knnK, _matchRadius),
            srcIndices,
            dstIndices);
    } else if (descriptor == "MATCH_SHAPES") {
        const int keep = mode == "MATCH" ? 1 : std::max(1, _knnK);
        const float radius = mode == "RADIUS" ? std::max(0.0f, _matchRadius) : -1.0f;
        md.raw_matches_knn.reserve(srcContours.size());
        for (size_t qi = 0; qi < srcContours.size(); ++qi) {
            if (!isValidContourForDescriptor(srcContours[qi])) {
                continue;
            }
            std::vector<cv::DMatch> candidates;
            for (size_t ti = 0; ti < dstContours.size(); ++ti) {
                if (!isValidContourForDescriptor(dstContours[ti])) {
                    continue;
                }
                const double d = cv::matchShapes(srcContours[qi], dstContours[ti],
                                                 cv::CONTOURS_MATCH_I1, 0);
                if (radius > 0.0f && d > radius) {
                    continue;
                }
                candidates.emplace_back(static_cast<int>(qi),
                                        static_cast<int>(ti),
                                        static_cast<float>(d));
            }
            if (candidates.empty()) {
                continue;
            }
            std::sort(candidates.begin(), candidates.end(),
                      [](const cv::DMatch& a, const cv::DMatch& b) {
                          return a.distance < b.distance;
                      });
            const size_t n = std::min(static_cast<size_t>(keep), candidates.size());
            md.raw_matches_knn.emplace_back(candidates.begin(), candidates.begin() + n);
        }
    } else {
        md.message = "unsupported descriptor: " + _descriptor;
        return false;
    }

    IR_LOG_INFO("ContourDescriptorAssociator KNN: raw groups=", md.raw_matches_knn.size());
    if (md.raw_matches_knn.empty()) {
        md.message = "matching produced no matches";
        return false;
    }

    std::vector<cv::DMatch> rawFlat;
    for (const auto& nb : md.raw_matches_knn) {
        for (const auto& m : nb) {
            rawFlat.push_back(m);
        }
    }

    std::vector<cv::DMatch> selected;
    if (_geometricFilter) {
        // 轮廓预筛选优先做面积比和一对一去重，再按 RIGID/TRANSLATION 模型保留几何一致的候选。
        selected = filterContourGeometric(rawFlat,
                                          srcContours,
                                          dstContours,
                                          _areaRatioMin,
                                          _geometricModel,
                                          _shiftConsistencyThreshold,
                                          _rigidRansacThreshold,
                                          _rigidRansacIterations,
                                          _rigidMinInliers);
    } else {
        selected = filterUniqueByDistance(rawFlat, srcContours.size(), dstContours.size());
    }

    md.filtered_matches = selected;
    md.line_matches = selected;
    md.inlier_line_matches = selected;
    md.valid = static_cast<int>(selected.size()) >= _minMatches;
    md.score = srcContours.empty()
                   ? 0.0
                   : static_cast<double>(selected.size()) /
                         static_cast<double>(srcContours.size());

    if (!md.valid) {
        md.message = "not enough contour matches: " + std::to_string(selected.size());
        IR_LOG_WARN("ContourDescriptorAssociator rejected: ", md.message);
        return false;
    }

    cv::Point2d sumShift(0.0, 0.0);
    int shiftCount = 0;
    for (const auto& m : selected) {
        const cv::Point2d c1 = contourCentroid(srcContours[static_cast<size_t>(m.queryIdx)]);
        const cv::Point2d c2 = contourCentroid(dstContours[static_cast<size_t>(m.trainIdx)]);
        sumShift += (c2 - c1);
        ++shiftCount;
    }
    if (shiftCount > 0) {
        md.translation = {sumShift.x / shiftCount, sumShift.y / shiftCount};
        md.affine = (cv::Mat_<double>(2, 3) << 1.0,
                     0.0,
                     md.translation.x,
                     0.0,
                     1.0,
                     md.translation.y);
    }

    IR_LOG_INFO("ContourDescriptorAssociator produced ", descriptor, " matches: ",
                selected.size(), " / ", rawFlat.size(), " raw, translation=",
                md.translation);
    return true;
}

} // namespace ir

