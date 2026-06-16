#include "utils/visualization/structure/draw_structure_matches.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/imgproc.hpp>

#ifdef IR_HAS_OPENCV_LINE_DESCRIPTOR
#include <opencv2/line_descriptor.hpp>
#endif

#include "utils/logger.h"

namespace ir {

namespace {

// 将任意输入图像转为 BGR，便于在统一画布上叠加彩色结构连线。
cv::Mat toBgr(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }
    if (image.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if (image.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }
    return image.clone();
}

// 从结构响应图中收集非零点，作为结构匹配连线可视化的候选点。
std::vector<cv::Point> collectResponsePoints(const cv::Mat& response) {
    std::vector<cv::Point> points;
    if (response.empty()) {
        return points;
    }

    cv::Mat gray;
    if (response.channels() == 1) {
        gray = response;
    } else {
        cv::cvtColor(response, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat binary;
    cv::threshold(gray, binary, 0.0, 255.0, cv::THRESH_BINARY);
    if (binary.depth() != CV_8U) {
        binary.convertTo(binary, CV_8U);
    }
    cv::findNonZero(binary, points);
    return points;
}

// 在 target 响应图的局部窗口内寻找距离投影点最近的结构点。
bool nearestResponsePoint(const cv::Mat& response,
                          const cv::Point& center,
                          int radius,
                          cv::Point& nearest) {
    if (response.empty()) {
        return false;
    }

    const int x0 = std::max(0, center.x - radius);
    const int y0 = std::max(0, center.y - radius);
    const int x1 = std::min(response.cols - 1, center.x + radius);
    const int y1 = std::min(response.rows - 1, center.y + radius);
    if (x0 > x1 || y0 > y1) {
        return false;
    }

    // 局部邻域线性扫描足够直观，也避免为可视化阶段引入额外索引结构。
    double bestDist2 = std::numeric_limits<double>::infinity();
    bool found = false;
    for (int y = y0; y <= y1; ++y) {
        const uchar* row = response.ptr<uchar>(y);
        for (int x = x0; x <= x1; ++x) {
            if (row[x] == 0) {
                continue;
            }
            const double dx = static_cast<double>(x - center.x);
            const double dy = static_cast<double>(y - center.y);
            const double d2 = dx * dx + dy * dy;
            if (d2 < bestDist2) {
                bestDist2 = d2;
                nearest = cv::Point(x, y);
                found = true;
            }
        }
    }
    return found;
}

cv::Point lineMidpoint(const cv::Vec4i& line) {
    return cv::Point((line[0] + line[2]) / 2, (line[1] + line[3]) / 2);
}

cv::Point2f contourCentroidPoint(const std::vector<cv::Point>& contour) {
    const cv::Moments m = cv::moments(contour);
    if (std::abs(m.m00) < 1e-9) {
        return {-1.0f, -1.0f};
    }
    return {static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00)};
}

cv::Point2d applyAffinePoint(const cv::Mat& A, const cv::Point2d& p) {
    return {A.at<double>(0, 0) * p.x + A.at<double>(0, 1) * p.y + A.at<double>(0, 2),
            A.at<double>(1, 0) * p.x + A.at<double>(1, 1) * p.y + A.at<double>(1, 2)};
}

cv::Point2f linePoint1(const cv::Vec4i& line) {
    return {static_cast<float>(line[0]), static_cast<float>(line[1])};
}

cv::Point2f linePoint2(const cv::Vec4i& line) {
    return {static_cast<float>(line[2]), static_cast<float>(line[3])};
}

#ifdef IR_HAS_OPENCV_LINE_DESCRIPTOR
cv::line_descriptor::KeyLine toDrawableKeyLine(const cv::Vec4i& line, int classId) {
    const cv::Point2f p1 = linePoint1(line);
    const cv::Point2f p2 = linePoint2(line);
    const cv::Point2f delta = p2 - p1;

    cv::line_descriptor::KeyLine keyLine;
    keyLine.startPointX = p1.x;
    keyLine.startPointY = p1.y;
    keyLine.endPointX = p2.x;
    keyLine.endPointY = p2.y;
    keyLine.sPointInOctaveX = p1.x;
    keyLine.sPointInOctaveY = p1.y;
    keyLine.ePointInOctaveX = p2.x;
    keyLine.ePointInOctaveY = p2.y;
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

std::vector<cv::line_descriptor::KeyLine> toDrawableKeyLines(
    const std::vector<cv::Vec4i>& lines) {
    std::vector<cv::line_descriptor::KeyLine> out;
    out.reserve(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) {
        out.push_back(toDrawableKeyLine(lines[i], static_cast<int>(i)));
    }
    return out;
}
#endif

} // namespace

cv::Mat renderLineSegmentMatches(const RegistrationContext& ctx,
                                 const std::vector<cv::DMatch>& matches,
                                 int maxMatches) {
    if (matches.empty() || ctx.images.first.empty() || ctx.images.second.empty() ||
        ctx.structure_data.first.lines.empty() || ctx.structure_data.second.lines.empty()) {
        return {};
    }

    cv::Mat src = toBgr(ctx.images.first);
    cv::Mat dst = toBgr(ctx.images.second);
    if (src.empty() || dst.empty()) {
        return {};
    }

    std::vector<cv::DMatch> draw = matches;
    if (maxMatches > 0 && static_cast<int>(draw.size()) > maxMatches) {
        std::partial_sort(draw.begin(),
                          draw.begin() + maxMatches,
                          draw.end(),
                          [](const cv::DMatch& a, const cv::DMatch& b) {
                              return a.distance < b.distance;
                          });
        draw.resize(maxMatches);
    }

#ifdef IR_HAS_OPENCV_LINE_DESCRIPTOR
    const std::vector<cv::line_descriptor::KeyLine> srcKeyLines =
        toDrawableKeyLines(ctx.structure_data.first.lines);
    const std::vector<cv::line_descriptor::KeyLine> dstKeyLines =
        toDrawableKeyLines(ctx.structure_data.second.lines);

    try {
        cv::Mat lineMatches;
        const std::vector<char> matchesMask(draw.size(), 1);
        cv::line_descriptor::drawLineMatches(src,
                                             srcKeyLines,
                                             dst,
                                             dstKeyLines,
                                             draw,
                                             lineMatches,
                                             cv::Scalar(255, 180, 0),
                                             cv::Scalar(0, 220, 255),
                                             matchesMask);
        if (!lineMatches.empty()) {
            return lineMatches;
        }
    } catch (const cv::Exception& e) {
        IR_LOG_WARN("drawLineMatches failed; falling back to manual line rendering: ", e.what());
    }
#endif

    const int canvasRows = std::max(src.rows, dst.rows);
    const int canvasCols = src.cols + dst.cols;
    cv::Mat canvas(canvasRows, canvasCols, src.type(), cv::Scalar::all(0));
    src.copyTo(canvas(cv::Rect(0, 0, src.cols, src.rows)));
    dst.copyTo(canvas(cv::Rect(src.cols, 0, dst.cols, dst.rows)));

    int drawn = 0;
    for (const auto& m : draw) {
        if (m.queryIdx < 0 || m.trainIdx < 0 ||
            m.queryIdx >= static_cast<int>(ctx.structure_data.first.lines.size()) ||
            m.trainIdx >= static_cast<int>(ctx.structure_data.second.lines.size())) {
            continue;
        }

        const cv::Vec4i srcLine = ctx.structure_data.first.lines[static_cast<size_t>(m.queryIdx)];
        const cv::Vec4i dstLine = ctx.structure_data.second.lines[static_cast<size_t>(m.trainIdx)];
        const cv::Point dstMidRaw = lineMidpoint(dstLine);

        const cv::Point srcP1(srcLine[0], srcLine[1]);
        const cv::Point srcP2(srcLine[2], srcLine[3]);
        const cv::Point dstP1(dstLine[0] + src.cols, dstLine[1]);
        const cv::Point dstP2(dstLine[2] + src.cols, dstLine[3]);
        const cv::Point srcMid = lineMidpoint(srcLine);
        const cv::Point dstMid(dstMidRaw.x + src.cols, dstMidRaw.y);

        const cv::Scalar srcColor(0, 180, 255);
        const cv::Scalar dstColor(0, 255, 0);
        const cv::Scalar matchColor(255, 180, 0);
        cv::line(canvas, srcP1, srcP2, srcColor, 2, cv::LINE_AA);
        cv::line(canvas, dstP1, dstP2, dstColor, 2, cv::LINE_AA);
        cv::line(canvas, srcMid, dstMid, matchColor, 1, cv::LINE_AA);
        cv::circle(canvas, srcMid, 2, srcColor, cv::FILLED, cv::LINE_AA);
        cv::circle(canvas, dstMid, 2, dstColor, cv::FILLED, cv::LINE_AA);
        ++drawn;
    }

    return drawn > 0 ? canvas : cv::Mat{};
}

cv::Mat renderContourMatches(const RegistrationContext& ctx,
                             const std::vector<cv::DMatch>& matches,
                             int maxMatches) {
    if (matches.empty() || ctx.images.first.empty() || ctx.images.second.empty() ||
        ctx.structure_data.first.contours.empty() || ctx.structure_data.second.contours.empty()) {
        return {};
    }

    cv::Mat src = toBgr(ctx.images.first);
    cv::Mat dst = toBgr(ctx.images.second);
    if (src.empty() || dst.empty()) {
        return {};
    }

    std::vector<cv::DMatch> draw = matches;
    if (maxMatches > 0 && static_cast<int>(draw.size()) > maxMatches) {
        std::partial_sort(draw.begin(),
                          draw.begin() + maxMatches,
                          draw.end(),
                          [](const cv::DMatch& a, const cv::DMatch& b) {
                              return a.distance < b.distance;
                          });
        draw.resize(maxMatches);
    }

    const int canvasRows = std::max(src.rows, dst.rows);
    const int canvasCols = src.cols + dst.cols;
    cv::Mat canvas(canvasRows, canvasCols, src.type(), cv::Scalar::all(0));
    src.copyTo(canvas(cv::Rect(0, 0, src.cols, src.rows)));
    dst.copyTo(canvas(cv::Rect(src.cols, 0, dst.cols, dst.rows)));

    int drawn = 0;
    for (const auto& m : draw) {
        if (m.queryIdx < 0 || m.trainIdx < 0 ||
            m.queryIdx >= static_cast<int>(ctx.structure_data.first.contours.size()) ||
            m.trainIdx >= static_cast<int>(ctx.structure_data.second.contours.size())) {
            continue;
        }

        const auto& srcContour =
            ctx.structure_data.first.contours[static_cast<size_t>(m.queryIdx)];
        const auto& dstContour =
            ctx.structure_data.second.contours[static_cast<size_t>(m.trainIdx)];
        if (srcContour.empty() || dstContour.empty()) {
            continue;
        }

        const cv::Point2f srcCentroid = contourCentroidPoint(srcContour);
        const cv::Point2f dstCentroid = contourCentroidPoint(dstContour);
        if (srcCentroid.x < 0.0f || srcCentroid.y < 0.0f ||
            dstCentroid.x < 0.0f || dstCentroid.y < 0.0f) {
            continue;
        }

        std::vector<std::vector<cv::Point>> srcDrawContours{srcContour};
        std::vector<cv::Point> shiftedDstContour;
        shiftedDstContour.reserve(dstContour.size());
        for (const cv::Point& p : dstContour) {
            shiftedDstContour.emplace_back(p.x + src.cols, p.y);
        }
        std::vector<std::vector<cv::Point>> dstDrawContours{shiftedDstContour};

        const cv::Scalar srcColor(0, 180, 255);
        const cv::Scalar dstColor(0, 255, 0);
        const cv::Scalar matchColor(255, 180, 0);
        cv::drawContours(canvas, srcDrawContours, -1, srcColor, 2, cv::LINE_AA);
        cv::drawContours(canvas, dstDrawContours, -1, dstColor, 2, cv::LINE_AA);

        const cv::Point srcCenter(cvRound(srcCentroid.x), cvRound(srcCentroid.y));
        const cv::Point dstCenter(cvRound(dstCentroid.x) + src.cols, cvRound(dstCentroid.y));
        cv::line(canvas, srcCenter, dstCenter, matchColor, 1, cv::LINE_AA);
        cv::circle(canvas, srcCenter, 3, srcColor, cv::FILLED, cv::LINE_AA);
        cv::circle(canvas, dstCenter, 3, dstColor, cv::FILLED, cv::LINE_AA);
        ++drawn;
    }

    return drawn > 0 ? canvas : cv::Mat{};
}

cv::Mat renderStructureMatches(const RegistrationContext& ctx, int maxMatches) {
    if (!ctx.structure_match_data.valid || ctx.images.first.empty() || ctx.images.second.empty() ||
        ctx.structure_data.first.response.empty() || ctx.structure_data.second.response.empty()) {
        return {};
    }

    cv::Mat src = toBgr(ctx.images.first);
    cv::Mat dst = toBgr(ctx.images.second);
    if (src.empty() || dst.empty()) {
        return {};
    }

    cv::Mat dstResponseGray;
    if (ctx.structure_data.second.response.channels() == 1) {
        dstResponseGray = ctx.structure_data.second.response;
    } else {
        cv::cvtColor(ctx.structure_data.second.response, dstResponseGray, cv::COLOR_BGR2GRAY);
    }
    cv::Mat dstResponse;
    cv::threshold(dstResponseGray, dstResponse, 0.0, 255.0, cv::THRESH_BINARY);
    if (dstResponse.depth() != CV_8U) {
        dstResponse.convertTo(dstResponse, CV_8U);
    }

    const std::vector<cv::Point> srcPoints =
        collectResponsePoints(ctx.structure_data.first.response);
    if (srcPoints.empty() || cv::countNonZero(dstResponse) == 0) {
        return {};
    }

    // 左右拼接原图，保持和 keypoint drawMatches 类似的视觉布局。
    const int canvasRows = std::max(src.rows, dst.rows);
    const int canvasCols = src.cols + dst.cols;
    cv::Mat canvas(canvasRows, canvasCols, src.type(), cv::Scalar::all(0));
    src.copyTo(canvas(cv::Rect(0, 0, src.cols, src.rows)));
    dst.copyTo(canvas(cv::Rect(src.cols, 0, dst.cols, dst.rows)));

    const int limit = maxMatches > 0 ? maxMatches : 100;
    const int candidateCount = std::max(limit * 20, limit);
    const int stride =
        std::max(1, static_cast<int>(srcPoints.size()) / std::max(1, candidateCount));
    const int searchRadius = 10;
    const cv::Point2d shift = ctx.structure_match_data.translation;
    const cv::Mat& affine = ctx.structure_match_data.affine;
    const bool hasAffine = !affine.empty() && affine.rows == 2 && affine.cols == 3 &&
                           affine.type() == CV_64F;

    // 均匀抽样 source 响应点，避免边缘点过密导致连线图不可读。
    int drawn = 0;
    for (size_t i = 0; i < srcPoints.size() && drawn < limit; i += static_cast<size_t>(stride)) {
        const cv::Point& p = srcPoints[i];
        const cv::Point2d projectedFloat =
            hasAffine ? applyAffinePoint(affine, cv::Point2d(p)) : cv::Point2d(p) + shift;
        const cv::Point projected(cvRound(projectedFloat.x), cvRound(projectedFloat.y));
        if (projected.x < 0 || projected.y < 0 || projected.x >= dst.cols ||
            projected.y >= dst.rows) {
            continue;
        }

        cv::Point q;
        if (!nearestResponsePoint(dstResponse, projected, searchRadius, q)) {
            continue;
        }

        const cv::Point pCanvas = p;
        const cv::Point qCanvas(q.x + src.cols, q.y);
        const cv::Scalar color(0, 255, 255);
        cv::line(canvas, pCanvas, qCanvas, color, 1, cv::LINE_AA);
        cv::circle(canvas, pCanvas, 2, cv::Scalar(0, 180, 255), cv::FILLED, cv::LINE_AA);
        cv::circle(canvas, qCanvas, 2, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
        ++drawn;
    }

    return drawn > 0 ? canvas : cv::Mat{};
}

} // namespace ir

