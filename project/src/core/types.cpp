#include "core/types.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "utils/string_utils.h"

namespace ir {

std::string toString(KeypointType t) {
    switch (t) {
    case KeypointType::SIFT:
        return "SIFT";
    case KeypointType::SURF:
        return "SURF";
    case KeypointType::ORB:
        return "ORB";
    case KeypointType::BRISK:
        return "BRISK";
    case KeypointType::KAZE:
        return "KAZE";
    case KeypointType::AKAZE:
        return "AKAZE";
    case KeypointType::UNKNOWN:
    default:
        return "UNKNOWN";
    }
}

KeypointType keypointTypeFromString(const std::string& s) {
    const std::string u = string_utils::toUpperAscii(s);
    if (u == "SIFT") {
        return KeypointType::SIFT;
    }
    if (u == "SURF") {
        return KeypointType::SURF;
    }
    if (u == "ORB") {
        return KeypointType::ORB;
    }
    if (u == "BRISK") {
        return KeypointType::BRISK;
    }
    if (u == "KAZE") {
        return KeypointType::KAZE;
    }
    if (u == "AKAZE") {
        return KeypointType::AKAZE;
    }
    return KeypointType::UNKNOWN;
}

std::string toString(StructureType t) {
    switch (t) {
    case StructureType::EDGE:
        return "EDGE";
    case StructureType::LINE:
        return "LINE";
    case StructureType::CONTOUR:
        return "CONTOUR";
    case StructureType::UNKNOWN:
    default:
        return "UNKNOWN";
    }
}

StructureType structureTypeFromString(const std::string& s) {
    const std::string u = string_utils::toUpperAscii(s);
    if (u == "EDGE" || u == "CANNY") {
        return StructureType::EDGE;
    }
    if (u == "LINE" || u == "HOUGH_LINE" || u == "HOUGH") {
        return StructureType::LINE;
    }
    if (u == "CONTOUR" || u == "CONTOURS") {
        return StructureType::CONTOUR;
    }
    return StructureType::UNKNOWN;
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
    const std::string u = string_utils::toUpperAscii(s);
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
    const std::string u = string_utils::toUpperAscii(s);
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
    const std::string u = string_utils::toUpperAscii(s);
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

std::string toString(TransformType t) {
    switch (t) {
    case TransformType::PERSPECTIVE:
        return "PERSPECTIVE";
    case TransformType::AFFINE:
        return "AFFINE";
    case TransformType::UNKNOWN:
    default:
        return "UNKNOWN";
    }
}

TransformType toTransformType(GeometryType g) {
    switch (g) {
    case GeometryType::HOMOGRAPHY:
        return TransformType::PERSPECTIVE;
    case GeometryType::AFFINE:
    case GeometryType::RIGID:
    case GeometryType::SIMILARITY:
        return TransformType::AFFINE;
    default:
        return TransformType::UNKNOWN;
    }
}

int robustMethodFromString(const std::string& s) {
    const std::string u = string_utils::toUpperAscii(s);
    if (u == "RANSAC") {
        return cv::RANSAC;
    }
    if (u == "LMEDS") {
        return cv::LMEDS;
    }
    if (u == "RHO") {
        return cv::RHO;
    }
    if (u == "USAC_MAGSAC" || u == "MAGSAC" || u == "MAGSAC++") {
        return cv::USAC_MAGSAC;
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
