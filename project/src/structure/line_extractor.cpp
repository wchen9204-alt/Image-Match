#include "structure/line_extractor.h"

#include <cmath>

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

int normalizeAperture(int value) {
    if (value != 3 && value != 5 && value != 7) {
        return 3;
    }
    return value;
}

bool extractLinesForImage(const cv::Mat& gray,
                          cv::Mat& mask,
                          std::vector<cv::Vec4i>& lines,
                          double cannyThreshold1,
                          double cannyThreshold2,
                          int apertureSize,
                          double rho,
                          double thetaDegrees,
                          int threshold,
                          double minLineLength,
                          double maxLineGap,
                          int lineThickness) {
    cv::Mat edges;
    cv::Canny(gray, edges, cannyThreshold1, cannyThreshold2, apertureSize);
    cv::HoughLinesP(edges,
                    lines,
                    rho,
                    thetaDegrees * CV_PI / 180.0,
                    threshold,
                    minLineLength,
                    maxLineGap);

    mask = cv::Mat::zeros(gray.size(), CV_8U);
    for (const auto& l : lines) {
        cv::line(mask, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255), lineThickness);
    }
    return !lines.empty();
}

} // namespace

LineExtractor::LineExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    _cannyThreshold1 = yaml_utils::getDouble(params, "cannyThreshold1", 50.0);
    _cannyThreshold2 = yaml_utils::getDouble(params, "cannyThreshold2", 150.0);
    _apertureSize = normalizeAperture(yaml_utils::getInt(params, "apertureSize", 3));
    _rho = yaml_utils::getDouble(params, "rho", 1.0);
    _thetaDegrees = yaml_utils::getDouble(params, "thetaDegrees", 1.0);
    _threshold = yaml_utils::getInt(params, "threshold", 50);
    _minLineLength = yaml_utils::getDouble(params, "minLineLength", 30.0);
    _maxLineGap = yaml_utils::getDouble(params, "maxLineGap", 10.0);
    _lineThickness = yaml_utils::getInt(params, "lineThickness", 2);

    IR_LOG_INFO("LineExtractor: threshold=",
                _threshold,
                ", minLineLength=",
                _minLineLength,
                ", maxLineGap=",
                _maxLineGap);
}

bool LineExtractor::extract(RegistrationContext& ctx) {
    auto& sd = ctx.structure_data;
    const auto& fd = ctx.feature_data;

    if (fd.first.gray.empty() || fd.second.gray.empty()) {
        IR_LOG_ERROR("LineExtractor: input grayscale images are empty.");
        return false;
    }

    sd.clear();
    sd.type = StructureType::LINE;
    const bool ok1 = extractLinesForImage(fd.first.gray,
                                          sd.first.mask,
                                          sd.first.lines,
                                          _cannyThreshold1,
                                          _cannyThreshold2,
                                          _apertureSize,
                                          _rho,
                                          _thetaDegrees,
                                          _threshold,
                                          _minLineLength,
                                          _maxLineGap,
                                          _lineThickness);
    const bool ok2 = extractLinesForImage(fd.second.gray,
                                          sd.second.mask,
                                          sd.second.lines,
                                          _cannyThreshold1,
                                          _cannyThreshold2,
                                          _apertureSize,
                                          _rho,
                                          _thetaDegrees,
                                          _threshold,
                                          _minLineLength,
                                          _maxLineGap,
                                          _lineThickness);

    IR_LOG_INFO("LineExtractor extracted lines: ",
                sd.first.lines.size(),
                " / ",
                sd.second.lines.size());
    return ok1 && ok2;
}

} // namespace ir
