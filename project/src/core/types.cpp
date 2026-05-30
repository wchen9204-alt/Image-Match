#include "core/types.h"

#include <algorithm>
#include <cctype>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace ir {

namespace {

std::string toUpper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

} // namespace

std::string toString(FeatureType t) {
    switch (t) {
    case FeatureType::SIFT:
        return "SIFT";
    case FeatureType::SURF:
        return "SURF";
    case FeatureType::ORB:
        return "ORB";
    case FeatureType::BRISK:
        return "BRISK";
    case FeatureType::KAZE:
        return "KAZE";
    case FeatureType::AKAZE:
        return "AKAZE";
    case FeatureType::UNKNOWN:
    default:
        return "UNKNOWN";
    }
}

FeatureType featureTypeFromString(const std::string& s) {
    const std::string u = toUpper(s);
    if (u == "SIFT") {
        return FeatureType::SIFT;
    }
    if (u == "SURF") {
        return FeatureType::SURF;
    }
    if (u == "ORB") {
        return FeatureType::ORB;
    }
    if (u == "BRISK") {
        return FeatureType::BRISK;
    }
    if (u == "KAZE") {
        return FeatureType::KAZE;
    }
    if (u == "AKAZE") {
        return FeatureType::AKAZE;
    }
    return FeatureType::UNKNOWN;
}

std::string toString(NormType t) {
    switch (t) {
    case NormType::L1:
        return "L1";
    case NormType::L2:
        return "L2";
    case NormType::HAMMING:
        return "HAMMING";
    case NormType::HAMMING2:
        return "HAMMING2";
    case NormType::UNKNOWN:
    default:
        return "UNKNOWN";
    }
}

NormType normTypeFromString(const std::string& s) {
    const std::string u = toUpper(s);
    if (u == "L1") {
        return NormType::L1;
    }
    if (u == "L2") {
        return NormType::L2;
    }
    if (u == "HAMMING") {
        return NormType::HAMMING;
    }
    if (u == "HAMMING2") {
        return NormType::HAMMING2;
    }
    if (u == "AUTO") {
        return NormType::UNKNOWN;
    }
    return NormType::UNKNOWN;
}

int toCvNorm(NormType t) {
    switch (t) {
    case NormType::L1:
        return cv::NORM_L1;
    case NormType::L2:
        return cv::NORM_L2;
    case NormType::HAMMING:
        return cv::NORM_HAMMING;
    case NormType::HAMMING2:
        return cv::NORM_HAMMING2;
    case NormType::UNKNOWN:
    default:
        return cv::NORM_L2;
    }
}

std::string toString(MatchMethod t) {
    switch (t) {
    case MatchMethod::MATCH:
        return "MATCH";
    case MatchMethod::KNN:
        return "KNN";
    case MatchMethod::RADIUS:
        return "RADIUS";
    case MatchMethod::UNKNOWN:
    default:
        return "UNKNOWN";
    }
}

MatchMethod matchMethodFromString(const std::string& s) {
    const std::string u = toUpper(s);
    if (u == "MATCH" || u == "TOP1" || u == "NN") {
        return MatchMethod::MATCH;
    }
    if (u == "KNN" || u == "KNNMATCH") {
        return MatchMethod::KNN;
    }
    if (u == "RADIUS" || u == "RADIUSMATCH") {
        return MatchMethod::RADIUS;
    }
    return MatchMethod::UNKNOWN;
}

std::string toString(GeometryType t) {
    switch (t) {
    case GeometryType::HOMOGRAPHY:
        return "HOMOGRAPHY";
    case GeometryType::AFFINE:
        return "AFFINE";
    case GeometryType::RIGID:
        return "RIGID";
    case GeometryType::SIMILARITY:
        return "SIMILARITY";
    case GeometryType::UNKNOWN:
    default:
        return "UNKNOWN";
    }
}

GeometryType geometryTypeFromString(const std::string& s) {
    const std::string u = toUpper(s);
    if (u == "HOMOGRAPHY") {
        return GeometryType::HOMOGRAPHY;
    }
    if (u == "AFFINE") {
        return GeometryType::AFFINE;
    }
    if (u == "RIGID" || u == "EUCLIDEAN") {
        return GeometryType::RIGID;
    }
    if (u == "SIMILARITY") {
        return GeometryType::SIMILARITY;
    }
    return GeometryType::UNKNOWN;
}

int robustMethodFromString(const std::string& s) {
    const std::string u = toUpper(s);
    if (u == "RANSAC") {
        return cv::RANSAC;
    }
    if (u == "LMEDS") {
        return cv::LMEDS;
    }
    if (u == "RHO") {
        return cv::RHO;
    }
    if (u == "7POINT") {
        return cv::FM_7POINT;
    }
    if (u == "8POINT") {
        return cv::FM_8POINT;
    }
    if (u == "NONE" || u == "0") {
        return 0;
    }
    return -1;
}

} // namespace ir
