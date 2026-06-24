#include "structure/contour_feature.h"

#include <algorithm>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace ir {

ContourFeature buildContourFeature(const std::vector<cv::Point>& contour, int index) {
    ContourFeature feature;
    feature.index = index;

    if (contour.size() < 3) {
        return feature;
    }

    feature.area = cv::contourArea(contour);
    feature.perimeter = cv::arcLength(contour, true);
    feature.bbox = cv::boundingRect(contour);
    feature.bboxArea = static_cast<double>(feature.bbox.width) * feature.bbox.height;

    if (feature.area <= 0.0 || feature.perimeter <= 1e-6 || feature.bboxArea <= 1e-9) {
        return feature;
    }

    const cv::Moments m = cv::moments(contour);
    if (m.m00 > 1e-9) {
        feature.centroid = {m.m10 / m.m00, m.m01 / m.m00};
    }

    feature.extent = feature.area / feature.bboxArea;
    const double longSide =
        static_cast<double>(std::max(feature.bbox.width, feature.bbox.height));
    const double shortSide =
        static_cast<double>(std::max(1, std::min(feature.bbox.width, feature.bbox.height)));
    feature.aspectRatio = longSide / shortSide;
    feature.valid = true;
    return feature;
}

std::vector<ContourFeature> buildContourFeatures(
    const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<ContourFeature> features;
    features.reserve(contours.size());
    for (size_t i = 0; i < contours.size(); ++i) {
        features.push_back(buildContourFeature(contours[i], static_cast<int>(i)));
    }
    return features;
}

} // namespace ir
