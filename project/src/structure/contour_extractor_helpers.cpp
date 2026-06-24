#include "structure/contour_extractor_helpers.h"

#include <algorithm>

#include <opencv2/imgproc.hpp>

#include "structure/contour_feature.h"
#include "utils/string_utils.h"

namespace ir::contour_extractor_helpers {

int contourRetrievalModeFromString(const std::string& raw) {
    const std::string mode = string_utils::toUpperAscii(raw);
    if (mode == "LIST") {
        return cv::RETR_LIST;
    }
    if (mode == "CCOMP") {
        return cv::RETR_CCOMP;
    }
    if (mode == "TREE") {
        return cv::RETR_TREE;
    }
    return cv::RETR_EXTERNAL;
}

int contourApproxModeFromString(const std::string& raw) {
    const std::string mode = string_utils::toUpperAscii(raw);
    if (mode == "NONE" || mode == "CHAIN_APPROX_NONE") {
        return cv::CHAIN_APPROX_NONE;
    }
    return cv::CHAIN_APPROX_SIMPLE;
}

int normalizedBlurKernel(int kernel) {
    if (kernel <= 1) {
        return 0;
    }
    return (kernel % 2 == 0) ? (kernel + 1) : kernel;
}

std::pair<double, double> estimateAutoCannyThresholds(const cv::Mat& gray) {
    cv::Mat flat = gray.reshape(1, 1).clone();
    cv::sort(flat, flat, cv::SORT_ASCENDING);
    const int medianIndex = flat.cols / 2;
    const double median = static_cast<double>(flat.at<unsigned char>(0, medianIndex));
    const double sigma = 0.33;
    const double low = std::clamp((1.0 - sigma) * median, 0.0, 255.0);
    const double high = std::clamp((1.0 + sigma) * median, 0.0, 255.0);
    return {low, std::max(low + 1.0, high)};
}

bool extractContoursForImage(const cv::Mat& gray,
                             cv::Mat& response,
                             std::vector<std::vector<cv::Point>>& contours,
                             int blurKernel,
                             bool autoCanny,
                             double cannyThreshold1,
                             double cannyThreshold2,
                             int apertureSize,
                             int retrievalMode,
                             int approxMode,
                             double minArea,
                             double minPerimeter,
                             int minPoints,
                             int minBboxWidth,
                             int minBboxHeight,
                             double minExtent,
                             double maxAspectRatio,
                             int maxContours,
                             int contourThickness) {
    cv::Mat working = gray;
    cv::Mat blurred;
    if (blurKernel > 1) {
        cv::GaussianBlur(gray, blurred, cv::Size(blurKernel, blurKernel), 0.0);
        working = blurred;
    }

    double lowThreshold = cannyThreshold1;
    double highThreshold = cannyThreshold2;
    if (autoCanny) {
        const auto thresholds = estimateAutoCannyThresholds(working);
        lowThreshold = thresholds.first;
        highThreshold = thresholds.second;
    }

    cv::Mat edges;
    cv::Canny(working, edges, lowThreshold, highThreshold, apertureSize);

    std::vector<std::vector<cv::Point>> rawContours;
    cv::findContours(edges, rawContours, retrievalMode, approxMode);

    // 先按面积从大到小排序，截断时优先保留主结构。
    std::sort(rawContours.begin(), rawContours.end(), [](const auto& lhs, const auto& rhs) {
        return buildContourFeature(lhs).area > buildContourFeature(rhs).area;
    });

    contours.clear();
    contours.reserve(rawContours.size());
    for (const auto& contour : rawContours) {
        const ContourFeature feature = buildContourFeature(contour);

        if (static_cast<int>(contour.size()) < std::max(3, minPoints)) {
            continue;
        }
        if (!feature.valid) {
            continue;
        }
        if (feature.area < minArea) {
            continue;
        }
        if (feature.perimeter < minPerimeter) {
            continue;
        }
        if (feature.bbox.width < minBboxWidth || feature.bbox.height < minBboxHeight) {
            continue;
        }
        if (feature.extent < minExtent) {
            continue;
        }
        if (maxAspectRatio > 0.0 && feature.aspectRatio > maxAspectRatio) {
            continue;
        }

        contours.push_back(contour);
        if (maxContours > 0 && static_cast<int>(contours.size()) >= maxContours) {
            break;
        }
    }

    response = cv::Mat::zeros(gray.size(), CV_8U);
    cv::drawContours(response, contours, -1, cv::Scalar(255), contourThickness);
    return !contours.empty();
}

} // namespace ir::contour_extractor_helpers
