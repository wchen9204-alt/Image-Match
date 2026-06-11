#include "structure/line_extractor.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

#ifdef IR_HAS_OPENCV_LINE_DESCRIPTOR
#include <opencv2/line_descriptor.hpp>
#endif

#include "utils/logger.h"
#include "utils/image_utils.h"
#include "utils/string_utils.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

LineDetectorType lineDetectorTypeFromString(const std::string& raw) {
    const std::string key = string_utils::normalizedKey(raw);
    if (key == "HOUGHLINES" || key == "HOUGH" || key == "STANDARDHOUGH") {
        return LineDetectorType::HOUGH_LINES;
    }
    if (key == "HOUGHLINESP" || key == "HOUGHP" || key == "PPHT" ||
        key == "PROBABILISTICHOUGH") {
        return LineDetectorType::HOUGH_LINES_P;
    }
    if (key == "LSD" || key == "LINESEGMENTDETECTOR") {
        return LineDetectorType::LSD;
    }
    if (key == "FLD" || key == "FASTLINEDETECTOR") {
        return LineDetectorType::FLD;
    }
    return LineDetectorType::HOUGH_LINES_P;
}

const char* toString(LineDetectorType t) {
    switch (t) {
    case LineDetectorType::HOUGH_LINES:
        return "HOUGH_LINES";
    case LineDetectorType::HOUGH_LINES_P:
        return "HOUGH_LINES_P";
    case LineDetectorType::LSD:
        return "LSD";
    case LineDetectorType::FLD:
        return "FLD";
    default:
        return "HOUGH_LINES_P";
    }
}

double lineLength(const cv::Vec4i& line) {
    const cv::Point2d p1(static_cast<double>(line[0]), static_cast<double>(line[1]));
    const cv::Point2d p2(static_cast<double>(line[2]), static_cast<double>(line[3]));
    return cv::norm(p2 - p1);
}

void limitLines(std::vector<cv::Vec4i>& lines, int maxLines) {
    if (maxLines <= 0 || static_cast<int>(lines.size()) <= maxLines) {
        return;
    }

    // 优先保留较长线段，避免响应图被大量短碎线占满。
    std::stable_sort(lines.begin(), lines.end(), [](const cv::Vec4i& a, const cv::Vec4i& b) {
        return lineLength(a) > lineLength(b);
    });
    lines.resize(static_cast<size_t>(maxLines));
}

double lineAngle(const cv::Vec4i& line) {
    double angle = std::atan2(static_cast<double>(line[3] - line[1]),
                              static_cast<double>(line[2] - line[0]));
    if (angle < 0.0) {
        angle += CV_PI;
    }
    if (angle >= CV_PI) {
        angle -= CV_PI;
    }
    return angle;
}

double angleDistance(double a, double b) {
    const double d = std::abs(a - b);
    return std::min(d, CV_PI - d);
}

double pointSegmentDistance(const cv::Point2d& p, const cv::Vec4i& line) {
    std::vector<cv::Point2f> segment{
        cv::Point2f(static_cast<float>(line[0]), static_cast<float>(line[1])),
        cv::Point2f(static_cast<float>(line[2]), static_cast<float>(line[3]))};
    return std::abs(cv::pointPolygonTest(segment, cv::Point2f(p), true));
}

bool nearDuplicateLine(const cv::Vec4i& candidate,
                       const cv::Vec4i& kept,
                       double angleThreshold,
                       double distanceThreshold) {
    if (angleDistance(lineAngle(candidate), lineAngle(kept)) > angleThreshold) {
        return false;
    }

    const cv::Point2d c1(static_cast<double>(candidate[0]), static_cast<double>(candidate[1]));
    const cv::Point2d c2(static_cast<double>(candidate[2]), static_cast<double>(candidate[3]));
    const cv::Point2d k1(static_cast<double>(kept[0]), static_cast<double>(kept[1]));
    const cv::Point2d k2(static_cast<double>(kept[2]), static_cast<double>(kept[3]));
    const double candidateToKept =
        0.5 * (pointSegmentDistance(c1, kept) + pointSegmentDistance(c2, kept));
    const double keptToCandidate =
        0.5 * (pointSegmentDistance(k1, candidate) + pointSegmentDistance(k2, candidate));
    return std::min(candidateToKept, keptToCandidate) <= distanceThreshold;
}

void deduplicateLines(std::vector<cv::Vec4i>& lines,
                      double duplicateAngleDeg,
                      double duplicateDistance) {
    if (lines.empty()) {
        return;
    }

    std::stable_sort(lines.begin(), lines.end(), [](const cv::Vec4i& a, const cv::Vec4i& b) {
        return lineLength(a) > lineLength(b);
    });

    const double angleThreshold = std::max(0.0, duplicateAngleDeg) * CV_PI / 180.0;
    const double distanceThreshold = std::max(0.0, duplicateDistance);
    std::vector<cv::Vec4i> unique;
    unique.reserve(lines.size());
    for (const auto& line : lines) {
        bool duplicate = false;
        for (const auto& kept : unique) {
            if (nearDuplicateLine(line, kept, angleThreshold, distanceThreshold)) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            unique.push_back(line);
        }
    }
    lines.swap(unique);
}

void postprocessLines(std::vector<cv::Vec4i>& lines,
                      int maxLines,
                      bool deduplicate,
                      double duplicateAngleDeg,
                      double duplicateDistance) {
    if (deduplicate) {
        deduplicateLines(lines, duplicateAngleDeg, duplicateDistance);
    }
    limitLines(lines, maxLines);
}

void renderLineResponse(const cv::Size& size,
                        const std::vector<cv::Vec4i>& lines,
                        int lineThickness,
                        cv::Mat& response) {
    response = cv::Mat::zeros(size, CV_8U);
    for (const auto& line : lines) {
        cv::line(response,
                 cv::Point(line[0], line[1]),
                 cv::Point(line[2], line[3]),
                 cv::Scalar(255),
                 std::max(1, lineThickness),
                 cv::LINE_AA);
    }
}

bool extractHoughLinesP(const cv::Mat& gray,
                        double cannyThreshold1,
                        double cannyThreshold2,
                        int apertureSize,
                        double rho,
                        double thetaDegrees,
                        int threshold,
                        double minLineLength,
                        double maxLineGap,
                        int maxLines,
                        bool deduplicate,
                        double duplicateAngleDeg,
                        double duplicateDistance,
                        std::vector<cv::Vec4i>& lines) {
    cv::Mat edges;
    cv::Canny(gray, edges, cannyThreshold1, cannyThreshold2, apertureSize);
    cv::HoughLinesP(edges,
                    lines,
                    rho,
                    thetaDegrees * CV_PI / 180.0,
                    threshold,
                    minLineLength,
                    maxLineGap);
    postprocessLines(lines, maxLines, deduplicate, duplicateAngleDeg, duplicateDistance);
    return !lines.empty();
}

bool clipInfiniteHoughLine(float rho, float theta, const cv::Size& size, cv::Vec4i& segment) {
    // 1. 找到直线上最靠近原点的点（垂足）
    const double a = std::cos(theta);
    const double b = std::sin(theta);
    const double x0 = a * rho;
    const double y0 = b * rho;

    // 2. 从这个点，沿着直线方向前后各延伸 2000 单位
    cv::Point p1(cvRound(x0 + 2000.0 * (-b)), cvRound(y0 + 2000.0 * a));
    cv::Point p2(cvRound(x0 - 2000.0 * (-b)), cvRound(y0 - 2000.0 * a));
    if (!cv::clipLine(size, p1, p2)) {
        return false;
    }

    // 3. 把超长线段，剪切到图像范围内
    segment = cv::Vec4i(p1.x, p1.y, p2.x, p2.y);
    return true;
}

bool extractHoughLines(const cv::Mat& gray,
                       double cannyThreshold1,
                       double cannyThreshold2,
                       int apertureSize,
                       double rho,
                       double thetaDegrees,
                       int threshold,
                       double minLineLength,
                       int maxLines,
                       bool deduplicate,
                       double duplicateAngleDeg,
                       double duplicateDistance,
                       std::vector<cv::Vec4i>& lines) {
    cv::Mat edges;
    cv::Canny(gray, edges, cannyThreshold1, cannyThreshold2, apertureSize);

    std::vector<cv::Vec2f> rawLines;
    cv::HoughLines(edges, rawLines, rho, thetaDegrees * CV_PI / 180.0, threshold);

    lines.clear();
    lines.reserve(rawLines.size());
    for (const auto& raw : rawLines) {
        cv::Vec4i segment;
        if (!clipInfiniteHoughLine(raw[0], raw[1], gray.size(), segment)) {
            continue;
        }
        if (lineLength(segment) < minLineLength) {
            continue;
        }
        lines.push_back(segment);
    }
    postprocessLines(lines, maxLines, deduplicate, duplicateAngleDeg, duplicateDistance);
    return !lines.empty();
}

bool extractLsdLines(const cv::Mat& gray,
                     int refine,
                     double scale,
                     double sigmaScale,
                     double quant,
                     double angTh,
                     double logEps,
                     double densityTh,
                     int nBins,
                     double minLineLength,
                     int maxLines,
                     bool deduplicate,
                     double duplicateAngleDeg,
                     double duplicateDistance,
                     int detectorScale,
                     int detectorNumOctaves,
                     std::vector<cv::Vec4i>& lines) {
#ifdef IR_HAS_OPENCV_LINE_DESCRIPTOR
    // 主路径：使用 line_descriptor::LSDDetector，和 LBD 保持同一套 KeyLine 生态，
    // 避免后续再走一遍独立的线段检测流程。
    cv::Ptr<cv::line_descriptor::LSDDetector> lbdDetector =
        cv::line_descriptor::LSDDetector::createLSDDetector();
    if (lbdDetector) {
        std::vector<cv::line_descriptor::KeyLine> keyLines;
        lbdDetector->detect(gray,
                            keyLines,
                            std::max(1, detectorScale),
                            std::max(1, detectorNumOctaves));

        lines.clear();
        lines.reserve(keyLines.size());
        for (const auto& keyLine : keyLines) {
            cv::Vec4i segment(cvRound(keyLine.startPointX),
                              cvRound(keyLine.startPointY),
                              cvRound(keyLine.endPointX),
                              cvRound(keyLine.endPointY));
            if (lineLength(segment) < minLineLength) {
                continue;
            }
            lines.push_back(segment);
        }
        postprocessLines(lines, maxLines, deduplicate, duplicateAngleDeg, duplicateDistance);
        return !lines.empty();
    }

    // 兼容回退：如果专用检测器在运行时创建失败，则退回到经典 LSD 行为。
    IR_LOG_WARN("LineExtractor: failed to create line_descriptor LSDDetector; falling back to "
                "cv::LineSegmentDetector.");
#else
    // 构建期回退：如果没有 line_descriptor 模块，只能使用经典 OpenCV LSD。
    IR_LOG_WARN("LineExtractor: OpenCV line_descriptor is not available; LSD uses "
                "cv::LineSegmentDetector fallback.");
#endif

    cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector(
        refine, scale, sigmaScale, quant, angTh, logEps, densityTh, nBins);
    if (!detector) {
        return false;
    }

    std::vector<cv::Vec4f> detected;
    detector->detect(gray, detected);

    lines.clear();
    lines.reserve(detected.size());
    for (const auto& line : detected) {
        cv::Vec4i segment(cvRound(line[0]), cvRound(line[1]), cvRound(line[2]), cvRound(line[3]));
        if (lineLength(segment) < minLineLength) {
            continue;
        }
        lines.push_back(segment);
    }
    postprocessLines(lines, maxLines, deduplicate, duplicateAngleDeg, duplicateDistance);
    return !lines.empty();
}

bool extractFldLines(const cv::Mat& gray,
                     int lengthThreshold,
                     double distanceThreshold,
                     double cannyThreshold1,
                     double cannyThreshold2,
                     int cannyApertureSize,
                     bool doMerge,
                     double minLineLength,
                     int maxLines,
                     bool deduplicate,
                     double duplicateAngleDeg,
                     double duplicateDistance,
                     std::vector<cv::Vec4i>& lines) {
    cv::Ptr<cv::ximgproc::FastLineDetector> detector =
        cv::ximgproc::createFastLineDetector(lengthThreshold,
                                             distanceThreshold,
                                             cannyThreshold1,
                                             cannyThreshold2,
                                             cannyApertureSize,
                                             doMerge);
    if (!detector) {
        return false;
    }

    std::vector<cv::Vec4f> detected;
    detector->detect(gray, detected);

    lines.clear();
    lines.reserve(detected.size());
    for (const auto& line : detected) {
        cv::Vec4i segment(cvRound(line[0]), cvRound(line[1]), cvRound(line[2]), cvRound(line[3]));
        if (lineLength(segment) < minLineLength) {
            continue;
        }
        lines.push_back(segment);
    }
    postprocessLines(lines, maxLines, deduplicate, duplicateAngleDeg, duplicateDistance);
    return !lines.empty();
}

bool extractLinesForImage(const cv::Mat& gray,
                          LineDetectorType method,
                          cv::Mat& response,
                          std::vector<cv::Vec4i>& lines,
                          double cannyThreshold1,
                          double cannyThreshold2,
                          int apertureSize,
                          double rho,
                          double thetaDegrees,
                          int threshold,
                          int maxLines,
                          double minLineLength,
                          double maxLineGap,
                          int lineThickness,
                          bool deduplicate,
                          double duplicateAngleDeg,
                          double duplicateDistance,
                          int lsdRefine,
                          double lsdScale,
                          double lsdSigmaScale,
                          double lsdQuant,
                          double lsdAngTh,
                          double lsdLogEps,
                          double lsdDensityTh,
                          int lsdNBins,
                          int lsdDetectorScale,
                          int lsdDetectorNumOctaves,
                          int fldLengthThreshold,
                          double fldDistanceThreshold,
                          double fldCannyThreshold1,
                          double fldCannyThreshold2,
                          int fldCannyApertureSize,
                          bool fldDoMerge) {
    lines.clear();

    bool ok = false;
    switch (method) {
    case LineDetectorType::HOUGH_LINES:
        ok = extractHoughLines(gray,
                               cannyThreshold1,
                               cannyThreshold2,
                               apertureSize,
                               rho,
                               thetaDegrees,
                               threshold,
                               minLineLength,
                               maxLines,
                               deduplicate,
                               duplicateAngleDeg,
                               duplicateDistance,
                               lines);
        break;
    case LineDetectorType::HOUGH_LINES_P:
        ok = extractHoughLinesP(gray,
                                cannyThreshold1,
                                cannyThreshold2,
                                apertureSize,
                                rho,
                                thetaDegrees,
                                threshold,
                                minLineLength,
                                maxLineGap,
                                maxLines,
                                deduplicate,
                                duplicateAngleDeg,
                                duplicateDistance,
                                lines);
        break;
    case LineDetectorType::LSD:
        ok = extractLsdLines(gray,
                             lsdRefine,
                             lsdScale,
                             lsdSigmaScale,
                             lsdQuant,
                             lsdAngTh,
                             lsdLogEps,
                             lsdDensityTh,
                             lsdNBins,
                             minLineLength,
                             maxLines,
                             deduplicate,
                             duplicateAngleDeg,
                             duplicateDistance,
                             lsdDetectorScale,
                             lsdDetectorNumOctaves,
                             lines);
        break;
    case LineDetectorType::FLD:
        ok = extractFldLines(gray,
                             fldLengthThreshold,
                             fldDistanceThreshold,
                             fldCannyThreshold1,
                             fldCannyThreshold2,
                             fldCannyApertureSize,
                             fldDoMerge,
                             minLineLength,
                             maxLines,
                             deduplicate,
                             duplicateAngleDeg,
                             duplicateDistance,
                             lines);
        break;
    }

    renderLineResponse(gray.size(), lines, lineThickness, response);
    return ok;
}

} // namespace

LineExtractor::LineExtractor(const YAML::Node& cfg) {
    const YAML::Node extractor = cfg["extractor"];
    const YAML::Node params =
        extractor && extractor["params"] ? extractor["params"] : cfg["params"];

    _method =
        lineDetectorTypeFromString(yaml_utils::getString(extractor, "method", "HOUGH_LINES_P"));
    _cannyThreshold1 = yaml_utils::getDouble(params, "cannyThreshold1", 50.0);
    _cannyThreshold2 = yaml_utils::getDouble(params, "cannyThreshold2", 150.0);
    _apertureSize =
        image_utils::normalizedCannyAperture(yaml_utils::getInt(params, "apertureSize", 3));
    _rho = yaml_utils::getDouble(params, "rho", 1.0);
    _thetaDegrees = yaml_utils::getDouble(params, "thetaDegrees", 1.0);
    _threshold = yaml_utils::getInt(params, "threshold", 50);
    _maxLines = yaml_utils::getInt(params, "maxLines", 300);
    _minLineLength = yaml_utils::getDouble(params, "minLineLength", 30.0);
    _maxLineGap = yaml_utils::getDouble(params, "maxLineGap", 10.0);
    _lineThickness = yaml_utils::getInt(params, "lineThickness", 2);
    _deduplicateLines = yaml_utils::getBool(params, "deduplicateLines", true);
    _duplicateAngleDeg = yaml_utils::getDouble(params, "duplicateAngleDeg", 3.0);
    _duplicateDistance = yaml_utils::getDouble(params, "duplicateDistance", 8.0);

    const YAML::Node lsd = params && params["lsd"] ? params["lsd"] : YAML::Node();
    _lsdRefine = yaml_utils::getInt(lsd, "refine", 1);
    _lsdScale = yaml_utils::getDouble(lsd, "scale", 0.8);
    _lsdSigmaScale = yaml_utils::getDouble(lsd, "sigmaScale", 0.6);
    _lsdQuant = yaml_utils::getDouble(lsd, "quant", 2.0);
    _lsdAngTh = yaml_utils::getDouble(lsd, "angTh", 22.5);
    _lsdLogEps = yaml_utils::getDouble(lsd, "logEps", 0.0);
    _lsdDensityTh = yaml_utils::getDouble(lsd, "densityTh", 0.7);
    _lsdNBins = yaml_utils::getInt(lsd, "nBins", 1024);
    _lsdDetectorScale = yaml_utils::getInt(lsd, "detectorScale", 2);
    _lsdDetectorNumOctaves = yaml_utils::getInt(lsd, "detectorNumOctaves", 2);

    const YAML::Node fld = params && params["fld"] ? params["fld"] : YAML::Node();
    _fldLengthThreshold = yaml_utils::getInt(fld, "lengthThreshold", 10);
    _fldDistanceThreshold = yaml_utils::getDouble(fld, "distanceThreshold", 1.414213562);
    _fldCannyThreshold1 = yaml_utils::getDouble(fld, "cannyThreshold1", 50.0);
    _fldCannyThreshold2 = yaml_utils::getDouble(fld, "cannyThreshold2", 50.0);
    _fldCannyApertureSize =
        image_utils::normalizedCannyAperture(yaml_utils::getInt(fld, "cannyApertureSize", 3));
    _fldDoMerge = yaml_utils::getBool(fld, "doMerge", false);

    IR_LOG_INFO("LineExtractor: method=",
                toString(_method),
                ", threshold=",
                _threshold,
                ", minLineLength=",
                _minLineLength,
                ", maxLineGap=",
                _maxLineGap,
                ", deduplicateLines=",
                _deduplicateLines,
                ", duplicateAngleDeg=",
                _duplicateAngleDeg,
                ", duplicateDistance=",
                _duplicateDistance);
}

std::string LineExtractor::outputLabel() const {
    return std::string("LINE_") + toString(_method);
}

bool LineExtractor::extract(RegistrationContext& ctx) {
    auto& sd = ctx.structure_data;
    const auto& images = ctx.images;

    if (images.first_gray.empty() || images.second_gray.empty()) {
        IR_LOG_ERROR("LineExtractor: input grayscale images are empty.");
        return false;
    }

    sd.clear();
    sd.type = StructureType::LINE;
    const bool ok1 = extractLinesForImage(images.first_gray,
                                          _method,
                                          sd.first.response,
                                          sd.first.lines,
                                          _cannyThreshold1,
                                          _cannyThreshold2,
                                          _apertureSize,
                                          _rho,
                                          _thetaDegrees,
                                          _threshold,
                                          _maxLines,
                                          _minLineLength,
                                          _maxLineGap,
                                          _lineThickness,
                                          _deduplicateLines,
                                          _duplicateAngleDeg,
                                          _duplicateDistance,
                                          _lsdRefine,
                                          _lsdScale,
                                          _lsdSigmaScale,
                                          _lsdQuant,
                                          _lsdAngTh,
                                          _lsdLogEps,
                                          _lsdDensityTh,
                                          _lsdNBins,
                                          _lsdDetectorScale,
                                          _lsdDetectorNumOctaves,
                                          _fldLengthThreshold,
                                          _fldDistanceThreshold,
                                          _fldCannyThreshold1,
                                          _fldCannyThreshold2,
                                          _fldCannyApertureSize,
                                          _fldDoMerge);
    const bool ok2 = extractLinesForImage(images.second_gray,
                                          _method,
                                          sd.second.response,
                                          sd.second.lines,
                                          _cannyThreshold1,
                                          _cannyThreshold2,
                                          _apertureSize,
                                          _rho,
                                          _thetaDegrees,
                                          _threshold,
                                          _maxLines,
                                          _minLineLength,
                                          _maxLineGap,
                                          _lineThickness,
                                          _deduplicateLines,
                                          _duplicateAngleDeg,
                                          _duplicateDistance,
                                          _lsdRefine,
                                          _lsdScale,
                                          _lsdSigmaScale,
                                          _lsdQuant,
                                          _lsdAngTh,
                                          _lsdLogEps,
                                          _lsdDensityTh,
                                          _lsdNBins,
                                          _lsdDetectorScale,
                                          _lsdDetectorNumOctaves,
                                          _fldLengthThreshold,
                                          _fldDistanceThreshold,
                                          _fldCannyThreshold1,
                                          _fldCannyThreshold2,
                                          _fldCannyApertureSize,
                                          _fldDoMerge);

    IR_LOG_INFO("LineExtractor extracted lines with ",
                toString(_method),
                ": ",
                sd.first.lines.size(),
                " / ",
                sd.second.lines.size());
    return ok1 && ok2;
}

} // namespace ir
