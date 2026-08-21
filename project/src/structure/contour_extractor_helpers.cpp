#include "structure/contour_extractor_helpers.h"

#include <algorithm>
#include <cmath>

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

namespace {

cv::Mat normalizeEdgeResponse(const cv::Mat& response) {
    double minValue = 0.0;
    double maxValue = 0.0;
    cv::minMaxLoc(response, &minValue, &maxValue);
    if (!std::isfinite(minValue) || !std::isfinite(maxValue) || maxValue <= minValue) {
        return cv::Mat::zeros(response.size(), CV_8U);
    }

    cv::Mat normalized;
    response.convertTo(normalized,
                       CV_8U,
                       255.0 / (maxValue - minValue),
                       -minValue * 255.0 / (maxValue - minValue));
    return normalized;
}

void makeWorkingImage(const cv::Mat& gray,
                      int blurKernel,
                      double gaussianSigma,
                      cv::Mat& working) {
    if (gaussianSigma > 0.0) {
        cv::GaussianBlur(gray, working, cv::Size(), gaussianSigma);
    } else if (blurKernel > 1) {
        cv::GaussianBlur(gray, working, cv::Size(blurKernel, blurKernel), 0.0);
    } else {
        working = gray;
    }
}

bool buildEdgeResponse(const cv::Mat& gray,
                       cv::Mat& edgeResponse,
                       cv::Mat& binaryEdges,
                       const std::string& edgeOperator,
                       int blurKernel,
                       double gaussianSigma,
                       bool autoCanny,
                       double cannyThreshold1,
                       double cannyThreshold2,
                       int apertureSize,
                       double edgeBinaryThreshold,
                       double logSigma,
                       double logZeroCrossingThreshold) {
    const std::string method = string_utils::toUpperAscii(edgeOperator);

    if (method == "CANNY") {
        cv::Mat working;
        makeWorkingImage(gray, blurKernel, gaussianSigma, working);
        double lowThreshold = cannyThreshold1;
        double highThreshold = cannyThreshold2;
        if (autoCanny) {
            const auto thresholds = estimateAutoCannyThresholds(working);
            lowThreshold = thresholds.first;
            highThreshold = thresholds.second;
        }
        cv::Canny(working, edgeResponse, lowThreshold, highThreshold, apertureSize);
        binaryEdges = edgeResponse.clone();
        return true;
    }

    if (method == "SCHARR") {
        cv::Mat working;
        makeWorkingImage(gray, blurKernel, gaussianSigma, working);
        cv::Mat gradX;
        cv::Mat gradY;
        cv::Mat magnitude;
        cv::Scharr(working, gradX, CV_32F, 1, 0);
        cv::Scharr(working, gradY, CV_32F, 0, 1);
        cv::magnitude(gradX, gradY, magnitude);
        edgeResponse = normalizeEdgeResponse(magnitude);
    } else if (method == "LOG_RESPONSE" || method == "LOG_ZERO_CROSSING") {
        cv::Mat logWorking;
        cv::Mat laplacian;
        cv::GaussianBlur(gray, logWorking, cv::Size(), std::max(0.01, logSigma));
        cv::Laplacian(logWorking, laplacian, CV_32F);
        if (method == "LOG_ZERO_CROSSING") {
            edgeResponse = cv::Mat::zeros(laplacian.size(), CV_8U);
            const float threshold = static_cast<float>(std::max(0.0, logZeroCrossingThreshold));
            for (int y = 1; y < laplacian.rows - 1; ++y) {
                for (int x = 1; x < laplacian.cols - 1; ++x) {
                    const float center = laplacian.at<float>(y, x);
                    bool crossing = false;
                    for (int dy = -1; dy <= 1 && !crossing; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx == 0 && dy == 0) {
                                continue;
                            }
                            const float neighbor = laplacian.at<float>(y + dy, x + dx);
                            if (((center >= 0.0f && neighbor < 0.0f) ||
                                 (center < 0.0f && neighbor >= 0.0f)) &&
                                std::abs(center - neighbor) >= threshold) {
                                crossing = true;
                                break;
                            }
                        }
                    }
                    if (crossing) {
                        edgeResponse.at<unsigned char>(y, x) = 255;
                    }
                }
            }
            binaryEdges = edgeResponse.clone();
            return true;
        }

        cv::Mat absoluteLaplacian;
        cv::absdiff(laplacian, cv::Scalar::all(0), absoluteLaplacian);
        edgeResponse = normalizeEdgeResponse(absoluteLaplacian);
    } else if (method == "MORPH_GRADIENT") {
        cv::Mat working;
        makeWorkingImage(gray, blurKernel, gaussianSigma, working);
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(working, edgeResponse, cv::MORPH_GRADIENT, kernel);
    } else {
        return false;
    }

    const double threshold = edgeBinaryThreshold > 0.0 ? edgeBinaryThreshold : 0.0;
    const int thresholdType = edgeBinaryThreshold > 0.0 ? cv::THRESH_BINARY
                                                         : cv::THRESH_BINARY | cv::THRESH_OTSU;
    cv::threshold(edgeResponse, binaryEdges, threshold, 255.0, thresholdType);
    return true;
}

} // namespace

bool extractContoursForImage(const cv::Mat& gray,
                             cv::Mat& edgeResponse,
                             cv::Mat& response,
                             std::vector<std::vector<cv::Point>>& contours,
                             bool& responseIsPrimary,
                             const std::string& edgeOperator,
                             bool useFindContours,
                             bool filterContours,
                             int blurKernel,
                             double gaussianSigma,
                             bool autoCanny,
                             double cannyThreshold1,
                             double cannyThreshold2,
                             int apertureSize,
                             double edgeBinaryThreshold,
                             double logSigma,
                             double logZeroCrossingThreshold,
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
    responseIsPrimary = false;
    cv::Mat edges;
    if (!buildEdgeResponse(gray,
                           edgeResponse,
                           edges,
                           edgeOperator,
                           blurKernel,
                           gaussianSigma,
                           autoCanny,
                           cannyThreshold1,
                           cannyThreshold2,
                           apertureSize,
                           edgeBinaryThreshold,
                           logSigma,
                           logZeroCrossingThreshold)) {
        response.release();
        contours.clear();
        return false;
    }

    // 边缘结构模式：二值算子输出直接供响应图关联器和重叠验证使用。
    // 不调用 findContours，避免把开放边缘误解释为有面积的区域轮廓。
    if (!useFindContours) {
        response = edges;
        contours.clear();
        responseIsPrimary = true;
        return cv::countNonZero(response) > 0;
    }

    std::vector<std::vector<cv::Point>> rawContours;
    cv::findContours(edges, rawContours, retrievalMode, approxMode);

    contours.clear();
    if (filterContours) {
        // 先按面积从大到小排序，截断时优先保留主结构。
        std::sort(rawContours.begin(), rawContours.end(), [](const auto& lhs, const auto& rhs) {
            return buildContourFeature(lhs).area > buildContourFeature(rhs).area;
        });

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
    } else {
        // 关闭过滤时保留 findContours 的全部原始结果；不再排序或截断。
        contours = std::move(rawContours);
    }

    response = cv::Mat::zeros(gray.size(), CV_8U);
    cv::drawContours(response, contours, -1, cv::Scalar(255), contourThickness);
    return !contours.empty();
}

} // namespace ir::contour_extractor_helpers
