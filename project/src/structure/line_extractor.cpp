#include "structure/line_extractor.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

std::string methodKey(std::string s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(std::toupper(c)));
        }
    }
    return out;
}

LineDetectorType lineDetectorTypeFromString(const std::string& raw) {
    const std::string key = methodKey(raw);
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

int normalizeAperture(int value) {
    if (value != 3 && value != 5 && value != 7) {
        return 3;
    }
    return value;
}

double lineLength(const cv::Vec4i& line) {
    const double dx = static_cast<double>(line[2] - line[0]);
    const double dy = static_cast<double>(line[3] - line[1]);
    return std::sqrt(dx * dx + dy * dy);
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
    limitLines(lines, maxLines);
    return !lines.empty();
}

bool clipInfiniteHoughLine(float rho, float theta, const cv::Size& size, cv::Vec4i& segment) {
    const double a = std::cos(theta);
    const double b = std::sin(theta);
    const double x0 = a * rho;
    const double y0 = b * rho;

    cv::Point p1(cvRound(x0 + 2000.0 * (-b)), cvRound(y0 + 2000.0 * a));
    cv::Point p2(cvRound(x0 - 2000.0 * (-b)), cvRound(y0 - 2000.0 * a));
    if (!cv::clipLine(size, p1, p2)) {
        return false;
    }

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
    limitLines(lines, maxLines);
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
                     std::vector<cv::Vec4i>& lines) {
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
    limitLines(lines, maxLines);
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
    limitLines(lines, maxLines);
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
                          int lsdRefine,
                          double lsdScale,
                          double lsdSigmaScale,
                          double lsdQuant,
                          double lsdAngTh,
                          double lsdLogEps,
                          double lsdDensityTh,
                          int lsdNBins,
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
    _apertureSize = normalizeAperture(yaml_utils::getInt(params, "apertureSize", 3));
    _rho = yaml_utils::getDouble(params, "rho", 1.0);
    _thetaDegrees = yaml_utils::getDouble(params, "thetaDegrees", 1.0);
    _threshold = yaml_utils::getInt(params, "threshold", 50);
    _maxLines = yaml_utils::getInt(params, "maxLines", 300);
    _minLineLength = yaml_utils::getDouble(params, "minLineLength", 30.0);
    _maxLineGap = yaml_utils::getDouble(params, "maxLineGap", 10.0);
    _lineThickness = yaml_utils::getInt(params, "lineThickness", 2);

    const YAML::Node lsd = params && params["lsd"] ? params["lsd"] : YAML::Node();
    _lsdRefine = yaml_utils::getInt(lsd, "refine", 1);
    _lsdScale = yaml_utils::getDouble(lsd, "scale", 0.8);
    _lsdSigmaScale = yaml_utils::getDouble(lsd, "sigmaScale", 0.6);
    _lsdQuant = yaml_utils::getDouble(lsd, "quant", 2.0);
    _lsdAngTh = yaml_utils::getDouble(lsd, "angTh", 22.5);
    _lsdLogEps = yaml_utils::getDouble(lsd, "logEps", 0.0);
    _lsdDensityTh = yaml_utils::getDouble(lsd, "densityTh", 0.7);
    _lsdNBins = yaml_utils::getInt(lsd, "nBins", 1024);

    const YAML::Node fld = params && params["fld"] ? params["fld"] : YAML::Node();
    _fldLengthThreshold = yaml_utils::getInt(fld, "lengthThreshold", 10);
    _fldDistanceThreshold = yaml_utils::getDouble(fld, "distanceThreshold", 1.414213562);
    _fldCannyThreshold1 = yaml_utils::getDouble(fld, "cannyThreshold1", 50.0);
    _fldCannyThreshold2 = yaml_utils::getDouble(fld, "cannyThreshold2", 50.0);
    _fldCannyApertureSize =
        normalizeAperture(yaml_utils::getInt(fld, "cannyApertureSize", 3));
    _fldDoMerge = yaml_utils::getBool(fld, "doMerge", false);

    IR_LOG_INFO("LineExtractor: method=",
                toString(_method),
                ", threshold=",
                _threshold,
                ", minLineLength=",
                _minLineLength,
                ", maxLineGap=",
                _maxLineGap);
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
                                          _lsdRefine,
                                          _lsdScale,
                                          _lsdSigmaScale,
                                          _lsdQuant,
                                          _lsdAngTh,
                                          _lsdLogEps,
                                          _lsdDensityTh,
                                          _lsdNBins,
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
                                          _lsdRefine,
                                          _lsdScale,
                                          _lsdSigmaScale,
                                          _lsdQuant,
                                          _lsdAngTh,
                                          _lsdLogEps,
                                          _lsdDensityTh,
                                          _lsdNBins,
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
