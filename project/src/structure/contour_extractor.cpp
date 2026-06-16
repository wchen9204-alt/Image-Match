#include "structure/contour_extractor.h"

#include <algorithm>
#include <string>

#include <opencv2/imgproc.hpp>

#include "utils/image_utils.h"
#include "utils/logger.h"
#include "utils/string_utils.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

int contourRetrievalModeFromString(const std::string& raw) {
    const std::string mode = string_utils::toUpperAscii(raw);
    if (mode == "LIST") return cv::RETR_LIST;
    if (mode == "CCOMP") return cv::RETR_CCOMP;
    if (mode == "TREE") return cv::RETR_TREE;
    return cv::RETR_EXTERNAL;
}

int contourApproxModeFromString(const std::string& raw) {
    const std::string mode = string_utils::toUpperAscii(raw);
    if (mode == "NONE" || mode == "CHAIN_APPROX_NONE") return cv::CHAIN_APPROX_NONE;
    return cv::CHAIN_APPROX_SIMPLE;
}

bool extractContoursForImage(const cv::Mat& gray,
                             cv::Mat& response,
                             std::vector<std::vector<cv::Point>>& contours,
                             double cannyThreshold1,
                             double cannyThreshold2,
                             int apertureSize,
                             int retrievalMode,
                             int approxMode,
                             double minArea,
                             double minPerimeter,
                             int minPoints,
                             int maxContours,
                             int contourThickness) {
    cv::Mat edges;
    cv::Canny(gray, edges, cannyThreshold1, cannyThreshold2, apertureSize);

    std::vector<std::vector<cv::Point>> rawContours;
    cv::findContours(edges, rawContours, retrievalMode, approxMode);

    std::sort(rawContours.begin(), rawContours.end(), [](const auto& a, const auto& b) {
        return cv::contourArea(a) > cv::contourArea(b);
    });

    contours.clear();
    contours.reserve(rawContours.size());
    for (const auto& c : rawContours) {
        if (static_cast<int>(c.size()) < std::max(3, minPoints)) {
            continue;
        }
        if (cv::contourArea(c) < minArea) continue;
        if (cv::arcLength(c, true) < minPerimeter) continue;
        contours.push_back(c);
        if (maxContours > 0 && static_cast<int>(contours.size()) >= maxContours) {
            break;
        }
    }

    response = cv::Mat::zeros(gray.size(), CV_8U);
    cv::drawContours(response, contours, -1, cv::Scalar(255), contourThickness);
    return !contours.empty();
}

} // namespace

ContourExtractor::ContourExtractor(const YAML::Node& cfg) {
    const YAML::Node extractor = cfg["extractor"];
    const YAML::Node params = extractor && extractor["params"] ? extractor["params"] : cfg["params"];
    _cannyThreshold1 = yaml_utils::getDouble(params, "cannyThreshold1", 50.0);
    _cannyThreshold2 = yaml_utils::getDouble(params, "cannyThreshold2", 150.0);
    _apertureSize =
        image_utils::normalizedCannyAperture(yaml_utils::getInt(params, "apertureSize", 3));
    _retrievalMode = yaml_utils::getString(params, "retrievalMode", "EXTERNAL");
    _chainApprox = yaml_utils::getString(params, "chainApprox", "SIMPLE");
    _minArea = yaml_utils::getDouble(params, "minArea", 20.0);
    _minPerimeter = yaml_utils::getDouble(params, "minPerimeter", 0.0);
    _minPoints = yaml_utils::getInt(params, "minPoints", 3);
    _maxContours = yaml_utils::getInt(params, "maxContours", 1000);
    _contourThickness = yaml_utils::getInt(params, "contourThickness", 1);

    IR_LOG_INFO("ContourExtractor: minArea=",
                _minArea,
                ", minPerimeter=",
                _minPerimeter,
                ", minPoints=",
                _minPoints,
                ", maxContours=",
                _maxContours,
                ", retrievalMode=",
                _retrievalMode,
                ", chainApprox=",
                _chainApprox,
                ", contourThickness=",
                _contourThickness);
}

bool ContourExtractor::extract(RegistrationContext& ctx) {
    auto& sd = ctx.structure_data;
    const auto& images = ctx.images;

    if (images.first_gray.empty() || images.second_gray.empty()) {
        IR_LOG_ERROR("ContourExtractor: input grayscale images are empty.");
        return false;
    }

    sd.clear();
    sd.type = StructureType::CONTOUR;
    const int retrievalMode = contourRetrievalModeFromString(_retrievalMode);
    const int approxMode = contourApproxModeFromString(_chainApprox);
    const bool ok1 = extractContoursForImage(images.first_gray,
                                             sd.first.response,
                                             sd.first.contours,
                                             _cannyThreshold1,
                                             _cannyThreshold2,
                                             _apertureSize,
                                             retrievalMode,
                                             approxMode,
                                             _minArea,
                                             _minPerimeter,
                                             _minPoints,
                                             _maxContours,
                                             _contourThickness);
    const bool ok2 = extractContoursForImage(images.second_gray,
                                             sd.second.response,
                                             sd.second.contours,
                                             _cannyThreshold1,
                                             _cannyThreshold2,
                                             _apertureSize,
                                             retrievalMode,
                                             approxMode,
                                             _minArea,
                                             _minPerimeter,
                                             _minPoints,
                                             _maxContours,
                                             _contourThickness);

    IR_LOG_INFO("ContourExtractor extracted contours: ",
                sd.first.contours.size(),
                " / ",
                sd.second.contours.size());
    return ok1 && ok2;
}

} // namespace ir

