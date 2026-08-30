#include "evaluator/quality/edge_structure_diagnostic.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/edge_drawing.hpp>

namespace ir::edge_structure_diagnostic {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEpsilon = 1e-9;
// OpenCV 浣跨�?BGR锛歴ource 缁胯壊銆乼arget 绾㈣壊銆佸弻鏂逛弗鏍奸噸鍙犵殑瀹為檯鏀拺涓洪粍鑹层€?
const cv::Scalar kSourceLineColor(0, 255, 0);
const cv::Scalar kTargetLineColor(0, 0, 255);
const cv::Scalar kMatchedOverlapColor(0, 255, 255);

double degrees(double radians) {
    return radians * 180.0 / kPi;
}

double radians(double degreesValue) {
    return degreesValue * kPi / 180.0;
}

double clamp01(double value) {
    return std::clamp(value, 0.0, 1.0);
}

double length(const cv::Point2d& value) {
    return std::hypot(value.x, value.y);
}

cv::Point2d normalized(const cv::Point2d& value) {
    const double norm = length(value);
    return norm > kEpsilon ? value * (1.0 / norm) : cv::Point2d(1.0, 0.0);
}

double dot(const cv::Point2d& first, const cv::Point2d& second) {
    return first.x * second.x + first.y * second.y;
}

double cross(const cv::Point2d& first, const cv::Point2d& second) {
    return first.x * second.y - first.y * second.x;
}

double angle180(double angle) {
    angle = std::fmod(angle, 180.0);
    if (angle < 0.0) {
        angle += 180.0;
    }
    return angle;
}

double angleDifference180(double first, double second) {
    const double difference = std::abs(angle180(first) - angle180(second));
    return std::min(difference, 180.0 - difference);
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return -1.0;
    }
    std::sort(values.begin(), values.end());
    const double position = clamp01(fraction) * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    if (lower == upper) {
        return values[lower];
    }
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

double ratioOrInvalid(double numerator, double denominator) {
    return denominator > kEpsilon ? numerator / denominator : -1.0;
}

bool toGray8(const cv::Mat& image, cv::Mat& gray) {
    gray.release();
    if (image.empty()) {
        return false;
    }
    if (image.channels() == 1) {
        gray = image;
    } else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        return false;
    }
    if (gray.depth() == CV_8U) {
        return true;
    }
    cv::Mat normalizedGray;
    cv::normalize(gray, normalizedGray, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);
    gray = normalizedGray;
    return true;
}

bool buildForegroundMasks(const cv::Mat& gray,
                          int threshold,
                          cv::Mat& binaryMask,
                          cv::Mat& visibilityMask) {
    binaryMask.release();
    visibilityMask.release();
    if (gray.empty()) {
        return false;
    }
    cv::compare(gray, std::clamp(threshold, 0, 255), binaryMask, cv::CMP_GT);
    if (cv::countNonZero(binaryMask) == 0) {
        return false;
    }

    // 鐢ㄥ嚫鍖呰〃绀哄彲瑙佸墠鏅寖鍥达紝鍐呴儴绌鸿厰涓嶄細鎶婂彲姣旇緝瑙嗛噹鍒囩�?
    std::vector<cv::Point> points;
    cv::findNonZero(binaryMask, points);
    visibilityMask = binaryMask.clone();
    if (points.size() >= 3) {
        std::vector<cv::Point> hull;
        cv::convexHull(points, hull);
        cv::Mat hullMask = cv::Mat::zeros(gray.size(), CV_8U);
        cv::fillConvexPoly(hullMask, hull, cv::Scalar(255));
        visibilityMask = hullMask;
    }
    return cv::countNonZero(visibilityMask) > 0;
}

struct ForegroundMetrics {
    double elongation = -1.0;
    double axisOccupancy = -1.0;
    double centerlineDeviationRatio = -1.0;
    double longSide = -1.0;
    cv::RotatedRect box;
};

ForegroundMetrics measureForeground(const cv::Mat& rawMask) {
    ForegroundMetrics metrics;
    const int totalForeground = rawMask.empty() ? 0 : cv::countNonZero(rawMask);
    if (totalForeground <= 0) {
        return metrics;
    }

    if (totalForeground < 3) {
        return metrics;
    }

    std::vector<cv::Point> points;
    cv::findNonZero(rawMask, points);
    if (points.size() < 3) {
        return metrics;
    }
    metrics.box = cv::minAreaRect(points);
    const double width = std::max(0.0, static_cast<double>(metrics.box.size.width));
    const double height = std::max(0.0, static_cast<double>(metrics.box.size.height));
    const double shortSide = std::min(width, height);
    metrics.longSide = std::max(width, height);
    metrics.elongation = shortSide > kEpsilon ? metrics.longSide / shortSide : -1.0;

    // PCA 鍜屼腑蹇冪嚎鎻忚堪鍓嶆櫙鐨勫杞粨锛岄伩鍏嶅唴閮ㄦí绾裤€佸瓟娲炲拰寮哄急绾圭悊浣夸腑蹇冪嚎鍋忕Щ�?
    std::vector<cv::Point> hull;
    cv::convexHull(points, hull);
    cv::Mat hullMask = cv::Mat::zeros(rawMask.size(), CV_8U);
    cv::fillConvexPoly(hullMask, hull, cv::Scalar(255));
    std::vector<cv::Point> hullPoints;
    cv::findNonZero(hullMask, hullPoints);
    const cv::Moments moments = cv::moments(hullMask, true);
    if (moments.m00 <= kEpsilon || metrics.longSide <= kEpsilon) {
        return metrics;
    }
    const cv::Point2d centroid(moments.m10 / moments.m00,
                               moments.m01 / moments.m00);
    const double covarianceX = moments.mu20 / moments.m00;
    const double covarianceY = moments.mu02 / moments.m00;
    const double covarianceXY = moments.mu11 / moments.m00;
    const double angle = 0.5 * std::atan2(2.0 * covarianceXY,
                                          covarianceX - covarianceY);
    const cv::Point2d majorAxis(std::cos(angle), std::sin(angle));
    const cv::Point2d normalAxis(-majorAxis.y, majorAxis.x);

    double minimumPosition = std::numeric_limits<double>::infinity();
    double maximumPosition = -std::numeric_limits<double>::infinity();
    for (const cv::Point& point : points) {
        const double position = dot(cv::Point2d(point.x, point.y), majorAxis);
        minimumPosition = std::min(minimumPosition, position);
        maximumPosition = std::max(maximumPosition, position);
    }
    if (!std::isfinite(minimumPosition) || !std::isfinite(maximumPosition)) {
        return metrics;
    }

    // 涓ゅ儚绱犲鐨勪富杞村尯闂村彲瀹瑰繊鏍呮牸鍖栭€犳垚鐨勪簹鍍忕礌绌洪殭锛屽悓鏃朵粛鑳芥毚闇叉槑鏄炬柇瑁傘€?
    constexpr double kAxisBinWidth = 2.0;
    const int binCount = std::max(
        1, static_cast<int>(std::floor((maximumPosition - minimumPosition) /
                                      kAxisBinWidth)) + 1);
    std::vector<unsigned char> occupied(static_cast<size_t>(binCount), 0);
    for (const cv::Point& point : points) {
        const cv::Point2d value(point.x, point.y);
        const double position = dot(value, majorAxis);
        const int bin = std::clamp(
            static_cast<int>(std::floor((position - minimumPosition) / kAxisBinWidth)),
            0,
            binCount - 1);
        occupied[static_cast<size_t>(bin)] = 1;
    }

    std::vector<double> normalSums(static_cast<size_t>(binCount), 0.0);
    std::vector<int> binSamples(static_cast<size_t>(binCount), 0);
    for (const cv::Point& point : hullPoints) {
        const cv::Point2d value(point.x, point.y);
        const double position = dot(value, majorAxis);
        const int bin = std::clamp(
            static_cast<int>(std::floor((position - minimumPosition) / kAxisBinWidth)),
            0,
            binCount - 1);
        normalSums[static_cast<size_t>(bin)] += dot(value, normalAxis);
        ++binSamples[static_cast<size_t>(bin)];
    }

    int occupiedBins = 0;
    std::vector<double> centerlineDeviations;
    centerlineDeviations.reserve(static_cast<size_t>(binCount));
    const double referenceNormalPosition = dot(centroid, normalAxis);
    for (int bin = 0; bin < binCount; ++bin) {
        occupiedBins += occupied[static_cast<size_t>(bin)] != 0 ? 1 : 0;
        const int samples = binSamples[static_cast<size_t>(bin)];
        if (samples <= 0) {
            continue;
        }
        const double centerNormal = normalSums[static_cast<size_t>(bin)] /
                                    static_cast<double>(samples);
        centerlineDeviations.push_back(
            std::abs(centerNormal - referenceNormalPosition));
    }
    metrics.axisOccupancy = ratioOrInvalid(occupiedBins, binCount);
    const double centerlineP90 = percentile(centerlineDeviations, 0.90);
    metrics.centerlineDeviationRatio =
        centerlineP90 >= 0.0 ? centerlineP90 / metrics.longSide : -1.0;
    return metrics;
}

struct CommonCanvas {
    cv::Size size;
    cv::Mat sourceTransform;
    cv::Point2d offset;
};

// 杈撳�?source 宸茬敱璋冪敤�?warp �?target 鍧愭爣绯汇€傝繖閲屼粎涓哄弻鏂瑰鍔犵浉鍚岃竟璺濓紝
// 鍚庣画鍐嶇敤鍚屼竴涓弬鑰冩棆杞缓绔嬫按�?绔栫洿鍧愭爣锛岃€屼笉鍗曠嫭淇浠讳竴鍥惧儚銆?
bool buildOriginalCoordinateCanvas(const cv::Size& sourceSize,
                                   const cv::Size& targetSize,
                                   const Options& options,
                                   CommonCanvas& canvas,
                                   std::string& error) {
    const int padding = 2;
    const int width = std::max(sourceSize.width, targetSize.width) + padding * 2;
    const int height = std::max(sourceSize.height, targetSize.height) + padding * 2;
    if (width <= 0 || height <= 0 || width > options.max_canvas_side_pixels ||
        height > options.max_canvas_side_pixels ||
        static_cast<long long>(width) * static_cast<long long>(height) >
            static_cast<long long>(options.max_canvas_pixels)) {
        error = "translation reference canvas exceeds configured limit";
        return false;
    }
    canvas.size = cv::Size(width, height);
    canvas.offset = cv::Point2d(static_cast<double>(padding), static_cast<double>(padding));
    canvas.sourceTransform = cv::Mat::eye(3, 3, CV_64F);
    canvas.sourceTransform.at<double>(0, 2) = canvas.offset.x;
    canvas.sourceTransform.at<double>(1, 2) = canvas.offset.y;
    return true;
}

bool warpImage(const cv::Mat& source,
               const cv::Size& canvasSize,
               const cv::Mat& transform,
               int interpolation,
               cv::Mat& warped) {
    if (source.empty() || transform.empty()) {
        return false;
    }
    if (transform.rows == 2 && transform.cols == 3) {
        cv::warpAffine(source,
                       warped,
                       transform,
                       canvasSize,
                       interpolation,
                       cv::BORDER_CONSTANT,
                       cv::Scalar::all(0));
        return true;
    }
    cv::warpPerspective(source,
                        warped,
                        transform,
                        canvasSize,
                        interpolation,
                        cv::BORDER_CONSTANT,
                        cv::Scalar::all(0));
    return true;
}

void pasteTarget(const cv::Mat& target, const CommonCanvas& canvas, cv::Mat& output) {
    output = cv::Mat::zeros(canvas.size, target.type());
    const int x = static_cast<int>(std::llround(canvas.offset.x));
    const int y = static_cast<int>(std::llround(canvas.offset.y));
    const cv::Rect destination(x, y, target.cols, target.rows);
    if (destination.x >= 0 && destination.y >= 0 &&
        destination.x + destination.width <= output.cols &&
        destination.y + destination.height <= output.rows) {
        target.copyTo(output(destination));
    }
}

struct ReferenceRotationCanvas {
    cv::Size size;
    cv::Mat transform;
};

// 鍦ㄥ叡鍚岀敾甯冧腑鐢ㄥ悓涓€涓棆杞妸鍙傝€冧富鏂瑰悜鏀惧埌姘村钩杞达紝骞朵负鏃嬭浆鍚庤竟鐣岄鐣欏畨鍏ㄥ尯鍩熴�?
bool buildReferenceRotationCanvas(const cv::Size& inputSize,
                                  double referenceAngle,
                                  const Options& options,
                                  ReferenceRotationCanvas& canvas,
                                  std::string& error) {
    if (inputSize.width <= 0 || inputSize.height <= 0) {
        error = "reference rotation canvas has invalid input size";
        return false;
    }
    // 绾挎柟鍚戜互 180 搴︿负鍛ㄦ湡銆傚彇鏈€灏忕瓑浠锋棆杞紝閬垮�?179 搴﹁涓嶅繀瑕佸湴缈昏浆鏁村紶鍥俱€?
    const double rotationDegrees = referenceAngle > 90.0 ? referenceAngle - 180.0
                                                          : referenceAngle;
    cv::Mat transform = cv::getRotationMatrix2D(cv::Point2f(0.0F, 0.0F),
                                                rotationDegrees,
                                                1.0);
    const std::vector<cv::Point2d> corners = {
        {0.0, 0.0},
        {static_cast<double>(inputSize.width - 1), 0.0},
        {static_cast<double>(inputSize.width - 1), static_cast<double>(inputSize.height - 1)},
        {0.0, static_cast<double>(inputSize.height - 1)}};
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();
    for (const auto& corner : corners) {
        const double x = transform.at<double>(0, 0) * corner.x +
                         transform.at<double>(0, 1) * corner.y +
                         transform.at<double>(0, 2);
        const double y = transform.at<double>(1, 0) * corner.x +
                         transform.at<double>(1, 1) * corner.y +
                         transform.at<double>(1, 2);
        minX = std::min(minX, x);
        maxX = std::max(maxX, x);
        minY = std::min(minY, y);
        maxY = std::max(maxY, y);
    }
    constexpr int kPadding = 2;
    transform.at<double>(0, 2) += static_cast<double>(kPadding) - minX;
    transform.at<double>(1, 2) += static_cast<double>(kPadding) - minY;
    const int width = static_cast<int>(std::ceil(maxX - minX + 1.0)) + kPadding * 2;
    const int height = static_cast<int>(std::ceil(maxY - minY + 1.0)) + kPadding * 2;
    if (width <= 0 || height <= 0 || width > options.max_canvas_side_pixels ||
        height > options.max_canvas_side_pixels ||
        static_cast<long long>(width) * static_cast<long long>(height) >
            static_cast<long long>(options.max_canvas_pixels)) {
        error = "reference rotation canvas exceeds configured limit";
        return false;
    }
    canvas.size = cv::Size(width, height);
    canvas.transform = transform;
    return true;
}

cv::Point2d transformPoint(const cv::Mat& transform, const cv::Point2d& point) {
    return {transform.at<double>(0, 0) * point.x + transform.at<double>(0, 1) * point.y +
                transform.at<double>(0, 2),
            transform.at<double>(1, 0) * point.x + transform.at<double>(1, 1) * point.y +
                transform.at<double>(1, 2)};
}

// 鍒濆妫€娴嬩繚鎸佸湪鏃嬭浆鍓嶅畬鎴愶紱杩欓噷浠呭皢鍘熷杈撳嚭鍚屾鏄犲皠鍒板叡鍚屽弬鑰冨潗鏍囥€?
std::vector<cv::Vec4f> transformLineSegments(const std::vector<cv::Vec4f>& segments,
                                              const cv::Mat& transform) {
    std::vector<cv::Vec4f> output;
    output.reserve(segments.size());
    for (const auto& segment : segments) {
        const cv::Point2d first = transformPoint(transform, {segment[0], segment[1]});
        const cv::Point2d second = transformPoint(transform, {segment[2], segment[3]});
        output.emplace_back(static_cast<float>(first.x),
                            static_cast<float>(first.y),
                            static_cast<float>(second.x),
                            static_cast<float>(second.y));
    }
    return output;
}

void normalizeResponse(const cv::Mat& input, cv::Mat& response) {
    response.release();
    if (input.empty()) {
        return;
    }
    cv::Mat floatInput;
    input.convertTo(floatInput, CV_32F);
    double minValue = 0.0;
    double maxValue = 0.0;
    cv::minMaxLoc(floatInput, &minValue, &maxValue);
    if (maxValue <= minValue + kEpsilon) {
        response = cv::Mat::zeros(input.size(), CV_32F);
        return;
    }
    response = (floatInput - static_cast<float>(minValue)) /
               static_cast<float>(maxValue - minValue);
}

void computeGradient(const cv::Mat& gray,
                     cv::Mat& gradX,
                     cv::Mat& gradY,
                     cv::Mat& magnitude) {
    cv::Scharr(gray, gradX, CV_32F, 1, 0);
    cv::Scharr(gray, gradY, CV_32F, 0, 1);
    cv::Mat rawMagnitude;
    cv::magnitude(gradX, gradY, rawMagnitude);
    normalizeResponse(rawMagnitude, magnitude);
}

struct Fragment {
    cv::Point2d first;
    cv::Point2d second;
    double length = 0.0;
    double angle = 0.0;
};

// Keep the detector's raw fragment directions separate from the later
// least-squares fitted direction.  A 180-degree line orientation is averaged
// on the doubled-angle circle so opposite endpoint order has no effect.
double rawFragmentAngle(const std::vector<Fragment>& fragments) {
    double cosine = 0.0;
    double sine = 0.0;
    double weight = 0.0;
    for (const auto& fragment : fragments) {
        const double radiansValue = radians(fragment.angle) * 2.0;
        const double fragmentWeight = std::max(1.0, fragment.length);
        cosine += fragmentWeight * std::cos(radiansValue);
        sine += fragmentWeight * std::sin(radiansValue);
        weight += fragmentWeight;
    }
    if (weight <= kEpsilon || (std::abs(cosine) <= kEpsilon && std::abs(sine) <= kEpsilon)) {
        return -1.0;
    }
    return angle180(degrees(0.5 * std::atan2(sine, cosine)));
}

struct Interval {
    double begin = 0.0;
    double end = 0.0;
};

struct LineGroup {
    std::vector<Fragment> fragments;
    cv::Point2d origin;
    cv::Point2d direction{1.0, 0.0};
    cv::Point2d center;
    double angle = 0.0;
    double rawAngle = -1.0;
    double span = 0.0;
    double actualEdgeLength = 0.0;
    double maxGap = 0.0;
    double continuity = 0.0;
    double gapRatio = 0.0;
    double fragmentSpread = 0.0;
    double fitResidual = 0.0;
    double prominence = 0.0;
    bool valid = false;
    bool mainCandidate = false;
};

struct GroupBuildResult {
    // 妫€娴嬪櫒鐩存帴杩斿洖鐨勭墖娈碉紝鐢ㄤ簬鍖哄垎鈥滄娴嬩笉鍒扳€濆拰鈥滃悗缁杩囨护鈥濄€?
    std::vector<cv::Vec4f> initialSegments;
    std::vector<LineGroup> groups;
    int fragmentCount = 0;
};

double pointLineDistance(const cv::Point2d& point,
                         const cv::Point2d& origin,
                         const cv::Point2d& direction) {
    return std::abs(cross(point - origin, direction));
}

struct FragmentCluster {
    Fragment anchor;
    std::vector<Fragment> fragments;
};

// 鍏卞悓鍙傝€冩柟鍚戜笅鐨勬寮忕嚎缁勫€欓€夈€傚潎鍊奸殢鐗囨鍔犲叆鑰屾洿鏂帮紝閬垮厤棣栫墖娈靛浐瀹氫负閿氱偣�?
struct ReferenceFragmentCluster {
    int axis = -1;
    std::vector<Fragment> fragments;
    double normalPosition = 0.0;
    double angleOffset = 0.0;
    double totalLength = 0.0;
};

bool fragmentFitsCluster(const Fragment& fragment,
                         const FragmentCluster& cluster,
                         const Options& options,
                         double& score) {
    const double angleError = angleDifference180(fragment.angle, cluster.anchor.angle);
    if (angleError > options.group_max_angle_difference_degrees) {
        return false;
    }
    const cv::Point2d direction = normalized(cluster.anchor.second - cluster.anchor.first);
    const double normalDistance = 0.5 *
                                  (pointLineDistance(fragment.first,
                                                     cluster.anchor.first,
                                                     direction) +
                                   pointLineDistance(fragment.second,
                                                     cluster.anchor.first,
                                                     direction));
    if (normalDistance > options.group_max_normal_distance_pixels) {
        return false;
    }
    const double begin = std::min(dot(fragment.first, direction), dot(fragment.second, direction));
    const double end = std::max(dot(fragment.first, direction), dot(fragment.second, direction));
    for (const auto& existing : cluster.fragments) {
        const double existingBegin =
            std::min(dot(existing.first, direction), dot(existing.second, direction));
        const double existingEnd =
            std::max(dot(existing.first, direction), dot(existing.second, direction));
        const double overlap = std::min(end, existingEnd) - std::max(begin, existingBegin);
        if (overlap > kEpsilon) {
            return false;
        }
    }
    score = angleError / std::max(kEpsilon, options.group_max_angle_difference_degrees) +
            normalDistance / std::max(kEpsilon, options.group_max_normal_distance_pixels);
    return true;
}

std::vector<Interval> mergeIntervals(std::vector<Interval> intervals, double& total, double& maxGap) {
    total = 0.0;
    maxGap = 0.0;
    if (intervals.empty()) {
        return intervals;
    }
    for (auto& interval : intervals) {
        if (interval.begin > interval.end) {
            std::swap(interval.begin, interval.end);
        }
    }
    std::sort(intervals.begin(), intervals.end(), [](const Interval& first, const Interval& second) {
        return first.begin < second.begin;
    });
    std::vector<Interval> merged;
    merged.push_back(intervals.front());
    for (size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i].begin <= merged.back().end) {
            merged.back().end = std::max(merged.back().end, intervals[i].end);
        } else {
            maxGap = std::max(maxGap, intervals[i].begin - merged.back().end);
            merged.push_back(intervals[i]);
        }
    }
    for (const auto& interval : merged) {
        total += std::max(0.0, interval.end - interval.begin);
    }
    return merged;
}

double intervalOverlap(const std::vector<Interval>& first,
                        const std::vector<Interval>& second) {
    double overlap = 0.0;
    size_t firstIndex = 0;
    size_t secondIndex = 0;
    while (firstIndex < first.size() && secondIndex < second.size()) {
        const double begin = std::max(first[firstIndex].begin, second[secondIndex].begin);
        const double end = std::min(first[firstIndex].end, second[secondIndex].end);
        if (end > begin) {
            overlap += end - begin;
        }
        if (first[firstIndex].end < second[secondIndex].end) {
            ++firstIndex;
        } else {
            ++secondIndex;
        }
    }
    return overlap;
}

// EDLines 鐩存帴浠庡叡鍚屽彲瑙佸尯鍩熷唴鐨勭伆搴﹀浘鎻愬彇鍒濆浜岀淮鐗囨�?
std::vector<cv::Vec4f> detectInitialLineSegments(const cv::Mat& lineInput,
                                                  const Options& options) {
    std::vector<cv::Vec4f> lines;
    cv::ximgproc::EdgeDrawing::Params params;
    params.MinLineLength = std::max(1, cvRound(options.min_fragment_length_pixels));
    cv::Ptr<cv::ximgproc::EdgeDrawing> detector = cv::ximgproc::createEdgeDrawing();
    if (detector) {
        detector->setParams(params);
        detector->detectEdges(lineInput);
        detector->detectLines(lines);
    }
    return lines;
}

// 鍒犻櫎閫€鍖?杩囩煭妫€娴嬬粨鏋滐紝骞朵繚鐣欑墖娈靛湪褰撳墠鍏卞悓鐢诲竷涓殑鐪熷疄瑙掑害�?
std::vector<Fragment> makeFragments(const std::vector<cv::Vec4f>& rawLines,
                                    const Options& options) {
    std::vector<Fragment> fragments;
    fragments.reserve(rawLines.size());
    for (const auto& line : rawLines) {
        const cv::Point2d first(line[0], line[1]);
        const cv::Point2d second(line[2], line[3]);
        const double segmentLength = length(second - first);
        if (segmentLength < options.min_fragment_length_pixels) {
            continue;
        }
        fragments.push_back({first,
                             second,
                             segmentLength,
                             angle180(degrees(std::atan2(second.y - first.y,
                                                         second.x - first.x)))});
    }
    return fragments;
}

double signedAngleOffset180(double angle, double referenceAngle) {
    double offset = angle180(angle) - angle180(referenceAngle);
    if (offset > 90.0) {
        offset -= 180.0;
    } else if (offset < -90.0) {
        offset += 180.0;
    }
    return offset;
}

// 灏嗘娴嬬墖娈垫嫙鍚堟垚閫昏緫绾跨粍锛涚湡瀹炴柇瑁備繚鐣欏�?fragments 涓�?
LineGroup finalizeLineGroup(const std::vector<Fragment>& component, const Options& options) {
    std::vector<cv::Point2f> points;
    points.reserve(component.size() * 2);
    for (const auto& fragment : component) {
        points.emplace_back(static_cast<float>(fragment.first.x),
                            static_cast<float>(fragment.first.y));
        points.emplace_back(static_cast<float>(fragment.second.x),
                            static_cast<float>(fragment.second.y));
    }
    cv::Vec4f fitted;
    cv::fitLine(points, fitted, cv::DIST_L2, 0.0, 0.01, 0.01);
    const cv::Point2d origin(fitted[2], fitted[3]);
    cv::Point2d direction = normalized(cv::Point2d(fitted[0], fitted[1]));
    if (direction.x < 0.0 || (std::abs(direction.x) < kEpsilon && direction.y < 0.0)) {
        direction *= -1.0;
    }

    LineGroup group;
    group.fragments = component;
    group.origin = origin;
    group.direction = direction;
    group.angle = angle180(degrees(std::atan2(direction.y, direction.x)));
    group.rawAngle = rawFragmentAngle(component);

    std::vector<Interval> intervals;
    std::vector<double> fragmentAngleErrors;
    std::vector<double> residuals;
    double lower = std::numeric_limits<double>::max();
    double upper = std::numeric_limits<double>::lowest();
    for (const auto& fragment : component) {
        const double firstProjection = dot(fragment.first - origin, direction);
        const double secondProjection = dot(fragment.second - origin, direction);
        intervals.push_back({std::min(firstProjection, secondProjection),
                             std::max(firstProjection, secondProjection)});
        lower = std::min(lower, intervals.back().begin);
        upper = std::max(upper, intervals.back().end);
        fragmentAngleErrors.push_back(angleDifference180(fragment.angle, group.angle));
        residuals.push_back(pointLineDistance(fragment.first, origin, direction));
        residuals.push_back(pointLineDistance(fragment.second, origin, direction));
    }

    double actual = 0.0;
    double maxGap = 0.0;
    mergeIntervals(std::move(intervals), actual, maxGap);
    group.span = std::max(0.0, upper - lower);
    group.actualEdgeLength = actual;
    group.maxGap = maxGap;
    group.continuity = ratioOrInvalid(actual, group.span);
    group.gapRatio = ratioOrInvalid(maxGap, group.span);
    group.fragmentSpread = percentile(fragmentAngleErrors, 0.90);
    group.fitResidual = percentile(residuals, 0.90);
    group.center = origin + direction * ((lower + upper) * 0.5);
    group.valid = group.actualEdgeLength >= options.min_line_group_actual_length_pixels &&
                  group.continuity >= options.min_line_group_continuity_ratio &&
                  group.gapRatio <= options.max_line_group_gap_ratio &&
                  group.fragmentSpread <= options.max_fragment_direction_spread_degrees &&
                  group.fitResidual <= options.max_line_fit_residual_pixels;
    return group;
}

GroupBuildResult buildLineGroups(const cv::Mat& lineInput, const Options& options) {
    GroupBuildResult output;
    if (lineInput.empty()) {
        return output;
    }
    const auto rawLines = detectInitialLineSegments(lineInput, options);
    output.initialSegments = rawLines;
    std::vector<Fragment> fragments;
    fragments.reserve(rawLines.size());
    for (const auto& line : rawLines) {
        const cv::Point2d first(line[0], line[1]);
        const cv::Point2d second(line[2], line[3]);
        const double segmentLength = length(second - first);
        if (segmentLength < options.min_fragment_length_pixels) {
            continue;
        }
        fragments.push_back({first,
                             second,
                             segmentLength,
                             angle180(degrees(std::atan2(second.y - first.y,
                                                         second.x - first.x)))});
    }
    output.fragmentCount = static_cast<int>(fragments.size());
    if (fragments.empty()) {
        return output;
    }

    std::sort(fragments.begin(), fragments.end(), [](const Fragment& first, const Fragment& second) {
        return first.length > second.length;
    });
    std::vector<FragmentCluster> clusters;
    for (const auto& fragment : fragments) {
        int bestCluster = -1;
        double bestScore = std::numeric_limits<double>::max();
        for (int index = 0; index < static_cast<int>(clusters.size()); ++index) {
            double score = 0.0;
            if (fragmentFitsCluster(fragment, clusters[index], options, score) &&
                score < bestScore) {
                bestScore = score;
                bestCluster = index;
            }
        }
        if (bestCluster < 0) {
            FragmentCluster cluster;
            cluster.anchor = fragment;
            cluster.fragments.push_back(fragment);
            clusters.push_back(std::move(cluster));
            continue;
        }
        auto& cluster = clusters[bestCluster];
        cluster.fragments.push_back(fragment);
    }

    for (const auto& cluster : clusters) {
        if (!cluster.fragments.empty()) {
            output.groups.push_back(finalizeLineGroup(cluster.fragments, options));
        }
    }
    return output;
}

// 鍦ㄤ富鏂瑰悜闂ㄦ帶閫氳繃鍚庯紝鎸夊弻鏂瑰叡鐢ㄧ殑鍙傝€冩柟鍚戦噸鏂扮粍缁囨寮忕嚎缁勩€?
// 鐗囨濮嬬粓淇濈暀鍦ㄥ師鍏卞悓鐢诲竷涓紝鍙湁鍒嗙被鍜屽垎缁勪娇鐢ㄥ弬鑰冧富杞达紝涓嶅涓ゅ紶鍥剧嫭绔嬫棆姝ｃ€?
GroupBuildResult buildReferenceLineGroups(const std::vector<cv::Vec4f>& rawLines,
                                           double referenceAngle,
                                           const Options& options) {
    GroupBuildResult output;
    output.initialSegments = rawLines;
    std::vector<Fragment> fragments = makeFragments(rawLines, options);
    output.fragmentCount = static_cast<int>(fragments.size());
    if (fragments.empty()) {
        return output;
    }

    const cv::Point2d horizontal(std::cos(radians(referenceAngle)),
                                 std::sin(radians(referenceAngle)));
    const cv::Point2d vertical(-horizontal.y, horizontal.x);
    const auto axisDirection = [&](int axis) {
        return axis == 0 ? horizontal : vertical;
    };
    const auto fragmentAxis = [&](const Fragment& fragment) {
        const double horizontalError = angleDifference180(fragment.angle, referenceAngle);
        const double verticalError = angleDifference180(fragment.angle, referenceAngle + 90.0);
        if (horizontalError <= options.group_max_angle_difference_degrees &&
            horizontalError <= verticalError) {
            return 0;
        }
        if (verticalError <= options.group_max_angle_difference_degrees) {
            return 1;
        }
        return -1;
    };

    struct ClassifiedFragment {
        Fragment fragment;
        int axis = -1;
    };
    std::vector<ClassifiedFragment> classified;
    classified.reserve(fragments.size());
    for (const auto& fragment : fragments) {
        const int axis = fragmentAxis(fragment);
        if (axis >= 0) {
            classified.push_back({fragment, axis});
        }
    }

    // 鍒嗙粍涓嶄緷璧栫墖娈甸暱搴︽垨妫€娴嬪櫒杈撳嚭椤哄簭銆傜墖娈靛姞鍏ョ嚎缁勫墠蹇呴』鍜岀粍鍐呮墍�?
    // 宸叉湁鐗囨鍏煎锛屼笉鑳介€氳繃鍏卞悓鐗囨鐨勪紶閫掑叧绯婚棿鎺ュ悎骞躲€傜嚎缁勬€婚暱搴︺€佹柇�?
    // 鍜岃鐩栫巼�?finalizeLineGroup() 涓啀鐙珛缁熻�?
    const auto normalAt = [](const Fragment& fragment,
                             const cv::Point2d& tangent,
                             const cv::Point2d& normal,
                             double tangentPosition) {
        const double firstTangent = dot(fragment.first, tangent);
        const double secondTangent = dot(fragment.second, tangent);
        const double denominator = secondTangent - firstTangent;
        if (std::abs(denominator) <= kEpsilon) {
            return 0.5 * (dot(fragment.first, normal) + dot(fragment.second, normal));
        }
        const double ratio = (tangentPosition - firstTangent) / denominator;
        return dot(fragment.first, normal) +
               ratio * (dot(fragment.second, normal) - dot(fragment.first, normal));
    };
    const auto tangentBegin = [&](const ClassifiedFragment& item) {
        const cv::Point2d tangent = axisDirection(item.axis);
        return std::min(dot(item.fragment.first, tangent), dot(item.fragment.second, tangent));
    };
    std::sort(classified.begin(), classified.end(), [&](const ClassifiedFragment& first,
                                                         const ClassifiedFragment& second) {
        if (first.axis != second.axis) {
            return first.axis < second.axis;
        }
        const double firstBegin = tangentBegin(first);
        const double secondBegin = tangentBegin(second);
        if (std::abs(firstBegin - secondBegin) > kEpsilon) {
            return firstBegin < secondBegin;
        }
        const cv::Point2d tangent = axisDirection(first.axis);
        const cv::Point2d normal(-tangent.y, tangent.x);
        return dot(first.fragment.first + first.fragment.second, normal) <
               dot(second.fragment.first + second.fragment.second, normal);
    });

    struct FragmentGroup {
        int axis = -1;
        std::vector<Fragment> fragments;
    };
    std::vector<FragmentGroup> groups;
    for (const auto& item : classified) {
        const cv::Point2d tangent = axisDirection(item.axis);
        const cv::Point2d normal(-tangent.y, tangent.x);
        const double itemBegin = std::min(dot(item.fragment.first, tangent),
                                          dot(item.fragment.second, tangent));
        const double itemEnd = std::max(dot(item.fragment.first, tangent),
                                        dot(item.fragment.second, tangent));
        int selectedGroup = -1;
        double selectedNormalDistance = std::numeric_limits<double>::max();
        for (int index = 0; index < static_cast<int>(groups.size()); ++index) {
            const auto& group = groups[index];
            if (group.axis != item.axis) {
                continue;
            }
            bool compatible = true;
            double worstNormalDistance = 0.0;
            for (const auto& existing : group.fragments) {
                if (angleDifference180(item.fragment.angle, existing.angle) >
                    options.group_max_angle_difference_degrees) {
                    compatible = false;
                    break;
                }
                const double existingBegin = std::min(dot(existing.first, tangent),
                                                      dot(existing.second, tangent));
                const double existingEnd = std::max(dot(existing.first, tangent),
                                                    dot(existing.second, tangent));
                // 鍚屼竴绾跨粍鍐呯殑鐪熷疄鐗囨涓嶈兘鍏锋湁鍒囧悜閲嶅彔銆傛娴嬪櫒鍦ㄥ叡浜鐐?
                // 澶勫彲鑳界暀涓嬩簹鍍忕礌浜ゅ弶锛?px 浠ュ唴瑙嗕负绔偣閲忓寲璇樊鑰岄潪鐪熷疄閲嶅彔�?
                constexpr double kEndpointOverlapTolerancePixels = 1.0;
                const double overlapBegin = std::max(itemBegin, existingBegin);
                const double overlapEnd = std::min(itemEnd, existingEnd);
                if (overlapEnd > overlapBegin + kEndpointOverlapTolerancePixels) {
                    compatible = false;
                    break;
                }
                const double comparisonTangent = 0.5 * (itemEnd < existingBegin
                                                             ? itemEnd + existingBegin
                                                             : existingEnd + itemBegin);
                const double normalDistance = std::abs(
                    normalAt(item.fragment, tangent, normal, comparisonTangent) -
                    normalAt(existing, tangent, normal, comparisonTangent));
                if (normalDistance > options.group_max_normal_distance_pixels) {
                    compatible = false;
                    break;
                }
                worstNormalDistance = std::max(worstNormalDistance, normalDistance);
            }
            if (compatible && worstNormalDistance < selectedNormalDistance) {
                selectedGroup = index;
                selectedNormalDistance = worstNormalDistance;
            }
        }
        if (selectedGroup < 0) {
            groups.push_back({item.axis, {item.fragment}});
        } else {
            groups[selectedGroup].fragments.push_back(item.fragment);
        }
    }

    output.groups.reserve(groups.size());
    for (const auto& group : groups) {
        output.groups.push_back(finalizeLineGroup(group.fragments, options));
    }
    return output;
}

// 棣栨鎸夊師濮嬬墖娈靛垎缁勫悗锛屼粛鍙兘鏈夊悓涓€鏂杈瑰洜灞€閮ㄦ硶鍚戝亸绉昏€屾媶鎴愬涓嚎缁勩€?
// 浠呴噸鍒嗛厤鍒囧悜鍖呯粶浜掍笉閲嶅彔鐨勬嫙鍚堢粍锛岄伩鍏嶅皢鍚屼竴浣嶇疆鐨勭浉閭荤墿鐞嗚竟鐣屽悎骞躲�?
std::vector<LineGroup> refitSeparatedLineGroups(std::vector<LineGroup> groups,
                                                 double referenceAngle,
                                                 bool repairLargeGapGroupsOnly,
                                                 const Options& options) {
    const double maximumNormalDistance =
        std::max(0.0, options.post_fit_group_normal_distance_pixels);
    if (maximumNormalDistance <= 0.0 || groups.size() < 2) {
        return groups;
    }

    const cv::Point2d horizontal(std::cos(radians(referenceAngle)),
                                 std::sin(radians(referenceAngle)));
    const cv::Point2d vertical(-horizontal.y, horizontal.x);
    const auto groupAxis = [&](const LineGroup& group) {
        return angleDifference180(group.angle, referenceAngle) <=
                       angleDifference180(group.angle, referenceAngle + 90.0)
                   ? 0
                   : 1;
    };
    const auto tangentIntervals = [](const LineGroup& group, const cv::Point2d& tangent) {
        std::vector<Interval> intervals;
        intervals.reserve(group.fragments.size());
        for (const auto& fragment : group.fragments) {
            intervals.push_back({std::min(dot(fragment.first, tangent), dot(fragment.second, tangent)),
                                 std::max(dot(fragment.first, tangent), dot(fragment.second, tangent))});
        }
        double ignoredLength = 0.0;
        double ignoredGap = 0.0;
        return mergeIntervals(std::move(intervals), ignoredLength, ignoredGap);
    };
    const auto normalAt = [](const LineGroup& group,
                             const cv::Point2d& tangent,
                             const cv::Point2d& normal,
                             double tangentPosition) {
        const double tangentSlope = dot(group.direction, tangent);
        if (std::abs(tangentSlope) <= kEpsilon) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double originTangent = dot(group.origin, tangent);
        const double originNormal = dot(group.origin, normal);
        return originNormal + dot(group.direction, normal) / tangentSlope *
                                  (tangentPosition - originTangent);
    };

    while (true) {
        int firstBest = -1;
        int secondBest = -1;
        double bestDistance = std::numeric_limits<double>::max();
        for (int first = 0; first < static_cast<int>(groups.size()); ++first) {
            const int axis = groupAxis(groups[first]);
            const cv::Point2d tangent = axis == 0 ? horizontal : vertical;
            const cv::Point2d normal(-tangent.y, tangent.x);
            const auto firstIntervals = tangentIntervals(groups[first], tangent);
            for (int second = first + 1; second < static_cast<int>(groups.size()); ++second) {
                if (groupAxis(groups[second]) != axis ||
                    angleDifference180(groups[first].angle, groups[second].angle) >
                        options.group_max_angle_difference_degrees) {
                    continue;
                }
                const bool firstNeedsRepair = !groups[first].valid &&
                                              groups[first].gapRatio >
                                                  options.max_line_group_gap_ratio;
                const bool secondNeedsRepair = !groups[second].valid &&
                                               groups[second].gapRatio >
                                                   options.max_line_group_gap_ratio;
                if (repairLargeGapGroupsOnly && !firstNeedsRepair && !secondNeedsRepair) {
                    continue;
                }
                const auto secondIntervals = tangentIntervals(groups[second], tangent);
                double closestFirst = 0.0;
                double closestSecond = 0.0;
                double smallestGap = std::numeric_limits<double>::max();
                for (const auto& firstInterval : firstIntervals) {
                    for (const auto& secondInterval : secondIntervals) {
                        if (std::min(firstInterval.end, secondInterval.end) >
                            std::max(firstInterval.begin, secondInterval.begin) + kEpsilon) {
                            smallestGap = -1.0;
                            break;
                        }
                        const double gap = firstInterval.end < secondInterval.begin
                                               ? secondInterval.begin - firstInterval.end
                                               : firstInterval.begin - secondInterval.end;
                        if (gap < smallestGap) {
                            smallestGap = gap;
                            closestFirst = firstInterval.end < secondInterval.begin
                                               ? firstInterval.end
                                               : firstInterval.begin;
                            closestSecond = firstInterval.end < secondInterval.begin
                                                ? secondInterval.begin
                                                : secondInterval.end;
                        }
                    }
                    if (smallestGap < 0.0) {
                        break;
                    }
                }
                if (smallestGap < 0.0 || !std::isfinite(smallestGap)) {
                    continue;
                }
                const double comparisonTangent = 0.5 * (closestFirst + closestSecond);
                const double firstNormal =
                    normalAt(groups[first], tangent, normal, comparisonTangent);
                const double secondNormal =
                    normalAt(groups[second], tangent, normal, comparisonTangent);
                if (!std::isfinite(firstNormal) || !std::isfinite(secondNormal)) {
                    continue;
                }
                const double normalDistance = std::abs(firstNormal - secondNormal);
                if (normalDistance > maximumNormalDistance || normalDistance >= bestDistance) {
                    continue;
                }
                std::vector<Fragment> fragments = groups[first].fragments;
                fragments.insert(fragments.end(),
                                 groups[second].fragments.begin(),
                                 groups[second].fragments.end());
                const LineGroup refitted = finalizeLineGroup(fragments, options);
                const bool refitAccepted =
                    refitted.actualEdgeLength >= options.min_line_group_actual_length_pixels &&
                    refitted.fragmentSpread <= options.max_fragment_direction_spread_degrees &&
                    refitted.fitResidual <= options.max_line_fit_residual_pixels;
                if (!refitAccepted) {
                    continue;
                }
                firstBest = first;
                secondBest = second;
                bestDistance = normalDistance;
            }
        }
        if (firstBest < 0) {
            break;
        }
        std::vector<Fragment> fragments = groups[firstBest].fragments;
        fragments.insert(fragments.end(),
                         groups[secondBest].fragments.begin(),
                         groups[secondBest].fragments.end());
        LineGroup merged = finalizeLineGroup(fragments, options);
        // 浜屾褰掔粍鐨勭洰鐨勫氨鏄繚鐣欑湡瀹炴柇瑁傜墖娈碉紱鏂姣斾緥鍙褰曚负璇婃柇鍊硷�?
        // 涓嶅啀闃绘宸茬粡閫氳繃鍑犱綍閲嶆嫙鍚堢殑绾跨粍杩涘叆鍚庣画璇佹嵁绛涢€夈�?
        merged.valid = merged.actualEdgeLength >= options.min_line_group_actual_length_pixels &&
                      merged.fragmentSpread <= options.max_fragment_direction_spread_degrees &&
                      merged.fitResidual <= options.max_line_fit_residual_pixels;
        groups[firstBest] = std::move(merged);
        groups.erase(groups.begin() + secondBest);
    }
    return groups;
}

double sampleResponse(const cv::Mat& response, const cv::Point2d& point) {
    const int x = cvRound(point.x);
    const int y = cvRound(point.y);
    if (x < 0 || y < 0 || x >= response.cols || y >= response.rows) {
        return 0.0;
    }
    return static_cast<double>(response.at<float>(y, x));
}

void assignProminence(std::vector<LineGroup>& groups, const cv::Mat& gradientMagnitude) {
    double maximum = 0.0;
    cv::minMaxLoc(gradientMagnitude, nullptr, &maximum);
    if (maximum <= kEpsilon) {
        return;
    }
    for (auto& group : groups) {
        double total = 0.0;
        double count = 0.0;
        const cv::Point2d normal(-group.direction.y, group.direction.x);
        (void)normal;
        for (const auto& fragment : group.fragments) {
            const int samples = std::max(1, cvRound(fragment.length));
            for (int index = 0; index <= samples; ++index) {
                const double ratio = static_cast<double>(index) / samples;
                total += sampleResponse(gradientMagnitude,
                                        fragment.first * (1.0 - ratio) + fragment.second * ratio);
                count += 1.0;
            }
        }
        group.prominence = count > 0.0 ? clamp01((total / count) / maximum) : 0.0;
    }
}

struct MainDirection {
    bool reliable = false;
    double angle = -1.0;
    double supportRatio = -1.0;
    double spread = -1.0;
    double margin = -1.0;
    double maxActualLengthRatio = -1.0;
    int candidateCount = 0;
};

struct DirectionCluster {
    std::vector<int> indices;
    double weight = 0.0;
    double center = -1.0;
    double spread = -1.0;
};

double directionCenter(const std::vector<LineGroup>& groups,
                       const std::vector<int>& indices) {
    double sumSin = 0.0;
    double sumCos = 0.0;
    for (const int index : indices) {
        const double angle = radians((groups[index].rawAngle >= 0.0 ?
                                          groups[index].rawAngle : groups[index].angle) * 2.0);
        const double weight = groups[index].span * groups[index].continuity;
        sumSin += weight * std::sin(angle);
        sumCos += weight * std::cos(angle);
    }
    return angle180(degrees(0.5 * std::atan2(sumSin, sumCos)));
}

MainDirection estimateMainDirection(std::vector<LineGroup>& groups,
                                    double longSide,
                                    const Options& options) {
    MainDirection output;
    if (longSide <= kEpsilon) {
        return output;
    }
    std::vector<int> candidates;
    for (int index = 0; index < static_cast<int>(groups.size()); ++index) {
        if (!groups[index].valid) {
            continue;
        }
        const double actualLengthRatio = ratioOrInvalid(groups[index].actualEdgeLength, longSide);
        output.maxActualLengthRatio = std::max(output.maxActualLengthRatio, actualLengthRatio);
        if (actualLengthRatio >= options.min_main_line_actual_length_ratio) {
            groups[index].mainCandidate = true;
            candidates.push_back(index);
        }
    }
    output.candidateCount = static_cast<int>(candidates.size());
    if (candidates.empty()) {
        return output;
    }

    std::vector<int> parent(candidates.size());
    std::iota(parent.begin(), parent.end(), 0);
    const auto findRoot = [&](int value, const auto& self) -> int {
        if (parent[value] == value) {
            return value;
        }
        parent[value] = self(parent[value], self);
        return parent[value];
    };
    const auto unite = [&](int first, int second) {
        const int firstRoot = findRoot(first, findRoot);
        const int secondRoot = findRoot(second, findRoot);
        if (firstRoot != secondRoot) {
            parent[secondRoot] = firstRoot;
        }
    };
    for (int first = 0; first < static_cast<int>(candidates.size()); ++first) {
        for (int second = first + 1; second < static_cast<int>(candidates.size()); ++second) {
            const double firstAngle = groups[candidates[first]].rawAngle >= 0.0
                                          ? groups[candidates[first]].rawAngle
                                          : groups[candidates[first]].angle;
            const double secondAngle = groups[candidates[second]].rawAngle >= 0.0
                                           ? groups[candidates[second]].rawAngle
                                           : groups[candidates[second]].angle;
            if (angleDifference180(firstAngle, secondAngle) <=
                options.direction_cluster_tolerance_degrees) {
                unite(first, second);
            }
        }
    }
    std::vector<std::vector<int>> components(candidates.size());
    for (int index = 0; index < static_cast<int>(candidates.size()); ++index) {
        components[findRoot(index, findRoot)].push_back(candidates[index]);
    }
    std::vector<DirectionCluster> clusters;
    for (const auto& component : components) {
        if (component.empty()) {
            continue;
        }
        DirectionCluster cluster;
        cluster.indices = component;
        cluster.center = directionCenter(groups, component);
        for (const int index : component) {
            cluster.weight += groups[index].span * groups[index].continuity;
        }
        std::vector<double> errors;
        for (const int index : component) {
            errors.push_back(angleDifference180(groups[index].angle, cluster.center));
        }
        cluster.spread = percentile(errors, 0.90);
        clusters.push_back(std::move(cluster));
    }
    std::sort(clusters.begin(), clusters.end(), [](const DirectionCluster& first,
                                                    const DirectionCluster& second) {
        return first.weight > second.weight;
    });
    if (clusters.empty()) {
        return output;
    }
    double allWeight = 0.0;
    for (const auto& cluster : clusters) {
        allWeight += cluster.weight;
    }
    const double secondWeight = clusters.size() > 1 ? clusters[1].weight : 0.0;
    const auto& main = clusters.front();
    output.angle = main.center;
    output.supportRatio = ratioOrInvalid(main.weight, allWeight);
    output.spread = main.spread;
    output.margin = ratioOrInvalid(main.weight - secondWeight, main.weight);
    output.reliable = !main.indices.empty() &&
                      output.supportRatio >= options.min_main_direction_support_ratio &&
                      output.spread <= options.max_main_direction_spread_degrees &&
                      output.margin >= options.min_main_direction_margin;
    return output;
}

enum class Axis { OTHER, HORIZONTAL, VERTICAL };

struct DirectionProfiles {
    std::vector<double> horizontal;
    std::vector<double> vertical;
    double horizontalOrigin = 0.0;
    double verticalOrigin = 0.0;
};

void smoothAndNormalizeProfile(std::vector<double>& profile, double sigma) {
    if (profile.empty()) {
        return;
    }
    cv::Mat row(1, static_cast<int>(profile.size()), CV_64F, profile.data());
    if (sigma > 0.0 && profile.size() > 2) {
        cv::GaussianBlur(row, row, cv::Size(), sigma, 0.0, cv::BORDER_REPLICATE);
    }
    double maximum = 0.0;
    cv::minMaxLoc(row, nullptr, &maximum);
    if (maximum > kEpsilon) {
        row /= maximum;
    }
}

DirectionProfiles buildDirectionProfiles(const cv::Mat& gradX,
                                         const cv::Mat& gradY,
                                         const cv::Mat& support,
                                         double referenceAngle,
                                         const Options& options) {
    DirectionProfiles profiles;
    const cv::Point2d horizontal(std::cos(radians(referenceAngle)),
                                 std::sin(radians(referenceAngle)));
    const cv::Point2d vertical(-horizontal.y, horizontal.x);
    const std::vector<cv::Point2d> corners = {
        {0.0, 0.0},
        {static_cast<double>(support.cols - 1), 0.0},
        {static_cast<double>(support.cols - 1), static_cast<double>(support.rows - 1)},
        {0.0, static_cast<double>(support.rows - 1)}};
    double minHorizontal = std::numeric_limits<double>::max();
    double maxHorizontal = std::numeric_limits<double>::lowest();
    double minVertical = std::numeric_limits<double>::max();
    double maxVertical = std::numeric_limits<double>::lowest();
    for (const auto& corner : corners) {
        minHorizontal = std::min(minHorizontal, dot(corner, horizontal));
        maxHorizontal = std::max(maxHorizontal, dot(corner, horizontal));
        minVertical = std::min(minVertical, dot(corner, vertical));
        maxVertical = std::max(maxVertical, dot(corner, vertical));
    }
    profiles.verticalOrigin = std::floor(minHorizontal);
    profiles.horizontalOrigin = std::floor(minVertical);
    profiles.vertical.assign(
        std::max(1, static_cast<int>(std::ceil(maxHorizontal) - profiles.verticalOrigin + 1.0)),
        0.0);
    profiles.horizontal.assign(
        std::max(1, static_cast<int>(std::ceil(maxVertical) - profiles.horizontalOrigin + 1.0)),
        0.0);
    for (int y = 0; y < support.rows; ++y) {
        const unsigned char* supportRow = support.ptr<unsigned char>(y);
        const float* gradXRow = gradX.ptr<float>(y);
        const float* gradYRow = gradY.ptr<float>(y);
        for (int x = 0; x < support.cols; ++x) {
            if (supportRow[x] == 0) {
                continue;
            }
            const cv::Point2d point(x, y);
            const cv::Point2d gradient(gradXRow[x], gradYRow[x]);
            const int horizontalIndex = cvRound(dot(point, vertical) - profiles.horizontalOrigin);
            const int verticalIndex = cvRound(dot(point, horizontal) - profiles.verticalOrigin);
            if (horizontalIndex >= 0 &&
                horizontalIndex < static_cast<int>(profiles.horizontal.size())) {
                profiles.horizontal[horizontalIndex] += std::abs(dot(gradient, vertical));
            }
            if (verticalIndex >= 0 &&
                verticalIndex < static_cast<int>(profiles.vertical.size())) {
                profiles.vertical[verticalIndex] += std::abs(dot(gradient, horizontal));
            }
        }
    }
    smoothAndNormalizeProfile(profiles.horizontal, options.profile_smoothing_sigma);
    smoothAndNormalizeProfile(profiles.vertical, options.profile_smoothing_sigma);
    return profiles;
}

double profileValue(const std::vector<double>& profile, double origin, double position) {
    if (profile.empty()) {
        return 0.0;
    }
    const int index = cvRound(position - origin);
    if (index < 0 || index >= static_cast<int>(profile.size())) {
        return 0.0;
    }
    return profile[index];
}

void applyProfileProminence(std::vector<LineGroup>& groups,
                            double referenceAngle,
                            const DirectionProfiles& profiles,
                            const Options& options) {
    const cv::Point2d horizontal(std::cos(radians(referenceAngle)),
                                 std::sin(radians(referenceAngle)));
    const cv::Point2d vertical(-horizontal.y, horizontal.x);
    for (auto& group : groups) {
        const double evidenceAngle = group.rawAngle >= 0.0 ? group.rawAngle : group.angle;
        const double horizontalError = angleDifference180(evidenceAngle, referenceAngle);
        const double verticalError = angleDifference180(evidenceAngle, referenceAngle + 90.0);
        if (horizontalError <= options.max_axis_classification_error_degrees &&
            horizontalError <= verticalError) {
            group.prominence = profileValue(profiles.horizontal,
                                            profiles.horizontalOrigin,
                                            dot(group.center, vertical));
        } else if (verticalError <= options.max_axis_classification_error_degrees) {
            group.prominence = profileValue(profiles.vertical,
                                            profiles.verticalOrigin,
                                            dot(group.center, horizontal));
        } else {
            group.prominence = 0.0;
        }
    }
}

struct EvidenceGroup {
    int index = -1;
    Axis axis = Axis::OTHER;
    bool mainDirection = false;
    double normalPosition = 0.0;
    double tangentCenter = 0.0;
    double actualLength = 0.0;
    double span = 0.0;
    double angle = 0.0;
    double prominence = 0.0;
    std::vector<Interval> tangentIntervals;
    // 鍗曟潯鎷熷悎绾垮湪鍏卞悓鍧愭爣绯诲垏鍚戜笂鐨勫畬鏁村寘缁滐紱涓嶅寘鍚湡瀹炴柇瑁傞棿闅欍�?
    Interval fittedTangentInterval;
};

Interval fittedTangentInterval(const LineGroup& group,
                               const cv::Point2d& tangentDirection) {
    if (group.fragments.empty()) {
        return {0.0, 0.0};
    }
    double lower = std::numeric_limits<double>::max();
    double upper = std::numeric_limits<double>::lowest();
    for (const auto& fragment : group.fragments) {
        lower = std::min(lower,
                         std::min(dot(fragment.first - group.origin, group.direction),
                                  dot(fragment.second - group.origin, group.direction)));
        upper = std::max(upper,
                         std::max(dot(fragment.first - group.origin, group.direction),
                                  dot(fragment.second - group.origin, group.direction)));
    }
    const cv::Point2d first = group.origin + group.direction * lower;
    const cv::Point2d second = group.origin + group.direction * upper;
    return {std::min(dot(first, tangentDirection), dot(second, tangentDirection)),
            std::max(dot(first, tangentDirection), dot(second, tangentDirection))};
}

// 涓ゆ潯绾跨粍鑻ュ湪鍒囧悜涓婅鐩栧悓涓€娈甸暱杈癸紝涓旀硶鍚戦棿璺濈鍚堢矖绾垮搴︼紝瑙嗕负鍚屼竴鐗╃悊闀胯竟鐨勫弻渚ц竟鐣屻€?
double envelopeOverlapRatio(const EvidenceGroup& first, const EvidenceGroup& second) {
    const double firstBegin = first.fittedTangentInterval.begin;
    const double firstEnd = first.fittedTangentInterval.end;
    const double secondBegin = second.fittedTangentInterval.begin;
    const double secondEnd = second.fittedTangentInterval.end;
    const double overlap = std::max(0.0, std::min(firstEnd, secondEnd) -
                                             std::max(firstBegin, secondBegin));
    return ratioOrInvalid(overlap, std::min(firstEnd - firstBegin, secondEnd - secondBegin));
}

// 绮楅暱杈圭殑鍐呬晶杈瑰鏄撹妯悜缁撴瀯鎵撴柇銆傚畠浠嶄綔涓哄彲瑙嗗寲鍜岃緟鍔╄瘉鎹繚鐣欙紱
// 鏈€缁堝尮閰嶅湪姘村钩鍙岃竟缁勪腑閫夊渚ц竟锛屽湪绔栫洿鍙岃竟缁勪腑閫夊叡鍚屽弬鑰冪郴鍙充晶杈广�?
void retainOuterLongitudinalEvidence(std::vector<EvidenceGroup>& evidence,
                                     double foregroundNormalCenter,
                                     const Options& options) {
    if ((!options.prefer_outer_longitudinal_edges &&
         !options.prefer_right_vertical_edges) ||
        evidence.size() < 2) {
        return;
    }

    const double minimumSeparation = std::max(
        0.0, options.outer_longitudinal_edge_min_normal_separation_pixels);
    const double maximumSeparation = std::max(
        minimumSeparation, options.outer_longitudinal_edge_max_normal_separation_pixels);
    std::vector<bool> suppressed(evidence.size(), false);
    for (size_t first = 0; first < evidence.size(); ++first) {
        if ((evidence[first].axis == Axis::HORIZONTAL &&
             !options.prefer_outer_longitudinal_edges) ||
            (evidence[first].axis == Axis::VERTICAL &&
             !options.prefer_right_vertical_edges) ||
            evidence[first].axis == Axis::OTHER) {
            continue;
        }
        std::vector<size_t> paired{first};
        for (size_t second = first + 1; second < evidence.size(); ++second) {
            if (evidence[second].axis != evidence[first].axis) {
                continue;
            }
            const double separation = std::abs(evidence[first].normalPosition -
                                               evidence[second].normalPosition);
            if (separation < minimumSeparation || separation > maximumSeparation ||
                envelopeOverlapRatio(evidence[first], evidence[second]) <
                    options.outer_longitudinal_edge_min_span_overlap_ratio) {
                continue;
            }
            paired.push_back(second);
        }
        if (paired.size() < 2) {
            continue;
        }

        size_t selected = paired.front();
        for (const size_t index : paired) {
            const bool preferCandidate = evidence[first].axis == Axis::VERTICAL
                                             ? evidence[index].normalPosition >
                                                   evidence[selected].normalPosition + kEpsilon
                                             : std::abs(evidence[index].normalPosition -
                                                        foregroundNormalCenter) >
                                                   std::abs(evidence[selected].normalPosition -
                                                            foregroundNormalCenter) + kEpsilon;
            const bool tie = evidence[first].axis == Axis::VERTICAL
                                 ? std::abs(evidence[index].normalPosition -
                                            evidence[selected].normalPosition) <= kEpsilon
                                 : std::abs(std::abs(evidence[index].normalPosition -
                                                   foregroundNormalCenter) -
                                            std::abs(evidence[selected].normalPosition -
                                                     foregroundNormalCenter)) <= kEpsilon;
            if (preferCandidate ||
                (tie &&
                 (evidence[index].actualLength > evidence[selected].actualLength + kEpsilon ||
                  (std::abs(evidence[index].actualLength - evidence[selected].actualLength) <=
                       kEpsilon &&
                   evidence[index].prominence > evidence[selected].prominence)))) {
                selected = index;
            }
        }
        for (const size_t index : paired) {
            if (index != selected) {
                suppressed[index] = true;
            }
        }
    }

    std::vector<EvidenceGroup> retained;
    retained.reserve(evidence.size());
    for (size_t index = 0; index < evidence.size(); ++index) {
        if (!suppressed[index]) {
            retained.push_back(std::move(evidence[index]));
        }
    }
    evidence = std::move(retained);
}

std::vector<Interval> projectIntervals(const LineGroup& group,
                                       const cv::Point2d& tangentDirection) {
    std::vector<Interval> intervals;
    for (const auto& fragment : group.fragments) {
        intervals.push_back({std::min(dot(fragment.first, tangentDirection),
                                      dot(fragment.second, tangentDirection)),
                             std::max(dot(fragment.first, tangentDirection),
                                      dot(fragment.second, tangentDirection))});
    }
    double total = 0.0;
    double maxGap = 0.0;
    return mergeIntervals(std::move(intervals), total, maxGap);
}

// 涓绘柟鍚戠‘瀹氬悗锛岃緟鍔╃珫绾跨殑鏂瑰悜鍥哄畾涓哄叾娉曠嚎锛涙í鍚戜綅缃敤鐪熷疄鐗囨绔偣鐨勪腑浣嶆暟绋冲仴浼拌銆?
double constrainedNormalPosition(const LineGroup& group, const cv::Point2d& normalDirection) {
    std::vector<double> positions;
    positions.reserve(group.fragments.size() * 2);
    for (const auto& fragment : group.fragments) {
        positions.push_back(dot(fragment.first, normalDirection));
        positions.push_back(dot(fragment.second, normalDirection));
    }
    return positions.empty() ? dot(group.center, normalDirection) : percentile(positions, 0.50);
}

void markReferenceMainDirectionGroups(std::vector<LineGroup>& groups,
                                      double longSide,
                                      const Options& options) {
    for (auto& group : groups) {
        const double groupAngle = group.rawAngle >= 0.0 ? group.rawAngle : group.angle;
        group.mainCandidate = group.valid &&
                              ratioOrInvalid(group.actualEdgeLength, longSide) >=
                                  options.min_main_line_actual_length_ratio &&
                              angleDifference180(groupAngle, 0.0) <=
                                  options.max_main_direction_spread_degrees;
    }
}

std::vector<EvidenceGroup> buildEvidence(const std::vector<LineGroup>& groups,
                                          double referenceAngle,
                                          double longitudinalNormalCenter,
                                          const Options& options) {
    const cv::Point2d horizontalDirection(std::cos(radians(referenceAngle)),
                                           std::sin(radians(referenceAngle)));
    const cv::Point2d verticalDirection(-horizontalDirection.y, horizontalDirection.x);
    std::vector<EvidenceGroup> candidates;
    for (int index = 0; index < static_cast<int>(groups.size()); ++index) {
        const auto& group = groups[index];
        if (!group.valid || group.prominence < options.min_peak_prominence) {
            continue;
        }
        const double evidenceAngle = group.rawAngle >= 0.0 ? group.rawAngle : group.angle;
        const double horizontalError = angleDifference180(evidenceAngle, referenceAngle);
        const double verticalError = angleDifference180(evidenceAngle, referenceAngle + 90.0);
        EvidenceGroup item;
            item.index = index;
        item.mainDirection = group.mainCandidate;
        item.angle = evidenceAngle;
        item.actualLength = group.actualEdgeLength;
        item.span = group.span;
        item.prominence = group.prominence;
        if (horizontalError <= options.max_axis_classification_error_degrees &&
            horizontalError <= verticalError) {
            item.axis = Axis::HORIZONTAL;
            // 涓绘柟鍚戠殑鍖归厤鍑犱綍鍏ㄩ儴鏉ヨ嚜鎷熷悎绾匡紱鐪熷疄鐗囨鍖洪棿鍙﹀瓨浜?tangentIntervals�?
            item.angle = group.angle;
            item.normalPosition = dot(group.center, verticalDirection);
            item.tangentCenter = dot(group.center, horizontalDirection);
            item.tangentIntervals = projectIntervals(group, horizontalDirection);
            item.fittedTangentInterval = fittedTangentInterval(group, horizontalDirection);
        } else if (verticalError <= options.max_axis_classification_error_degrees) {
            item.axis = Axis::VERTICAL;
            // 绔栫洿杈呭姪绾挎寜涓绘柟鍚戞硶绾垮彈绾︽潫閲嶆嫙鍚堬紝涓嶇户鎵挎娴嬪櫒浜х敓鐨勫井灏忓€炬枩�?
            item.angle = angle180(referenceAngle + 90.0);
            item.normalPosition = constrainedNormalPosition(group, horizontalDirection);
            item.tangentIntervals = projectIntervals(group, verticalDirection);
            if (!item.tangentIntervals.empty()) {
                item.span = item.tangentIntervals.back().end - item.tangentIntervals.front().begin;
            }
            item.fittedTangentInterval = {item.tangentIntervals.front().begin,
                                         item.tangentIntervals.back().end};
        } else {
            continue;
        }
        candidates.push_back(std::move(item));
    }

    // 鍏堝湪鏈浘銆佹湰鏂瑰悜鐨勬湁鏁堝€欓€変腑鎵惧埌鏈€闀垮疄闄呯嚎闀匡紝鍐嶇敤瀹冧綔涓虹浉瀵归暱搴﹀垎姣嶃€?
    // 杩欐牱鍓嶆櫙涓殑澶у潡鍖哄煙涓嶄細鎶婂彲鐢ㄧ嚎缁勮鍒や负杩囩煭�?
    double longestHorizontal = 0.0;
    double longestVertical = 0.0;
    for (const auto& item : candidates) {
        if (item.axis == Axis::HORIZONTAL) {
            longestHorizontal = std::max(longestHorizontal, item.actualLength);
        } else {
            longestVertical = std::max(longestVertical, item.actualLength);
        }
    }

    std::vector<EvidenceGroup> evidence;
    evidence.reserve(candidates.size());
    for (auto& item : candidates) {
        const double longest = item.axis == Axis::HORIZONTAL
                                   ? longestHorizontal
                                   : longestVertical;
        const double minimumRatio = item.axis == Axis::HORIZONTAL
                                        ? options.min_horizontal_actual_length_ratio
                                        : options.min_vertical_actual_length_ratio;
        if (ratioOrInvalid(item.actualLength, longest) < minimumRatio) {
            continue;
        }
        evidence.push_back(std::move(item));
    }

    // 鍘绘帀鍚屼竴鐗╃悊杈圭紭琚娴嬪櫒閲嶅杈撳嚭鐨勮繎閲嶅悎绾跨粍锛涙硶鍚戜綅缃槑鏄句笉鍚岀殑涓ゆ潯杈圭晫淇濈暀�?
    std::vector<EvidenceGroup> deduplicated;
    deduplicated.reserve(evidence.size());
    for (auto& item : evidence) {
        int duplicateIndex = -1;
        for (int index = 0; index < static_cast<int>(deduplicated.size()); ++index) {
            auto& existing = deduplicated[index];
            if (existing.axis != item.axis ||
                std::abs(existing.normalPosition - item.normalPosition) >
                    options.duplicate_line_normal_tolerance_pixels ||
                angleDifference180(existing.angle, item.angle) >
                    options.max_axis_classification_error_degrees) {
                continue;
            }
            const double existingBegin = existing.fittedTangentInterval.begin;
            const double existingEnd = existing.fittedTangentInterval.end;
            const double itemBegin = item.fittedTangentInterval.begin;
            const double itemEnd = item.fittedTangentInterval.end;
            const double envelopeOverlap = std::max(
                0.0, std::min(existingEnd, itemEnd) - std::max(existingBegin, itemBegin));
            const double shorterEnvelope = std::min(existingEnd - existingBegin,
                                                     itemEnd - itemBegin);
            if (ratioOrInvalid(envelopeOverlap, shorterEnvelope) <
                options.duplicate_line_min_span_overlap_ratio) {
                continue;
            }
            duplicateIndex = index;
            break;
        }
        if (duplicateIndex < 0) {
            deduplicated.push_back(std::move(item));
            continue;
        }
        auto& existing = deduplicated[duplicateIndex];
        if (item.actualLength > existing.actualLength ||
            (std::abs(item.actualLength - existing.actualLength) <= kEpsilon &&
             item.prominence > existing.prominence)) {
            existing = std::move(item);
        }
    }
    retainOuterLongitudinalEvidence(deduplicated, longitudinalNormalCenter, options);
    return deduplicated;
}

// 灏嗘渶缁堣瘉鎹瓫閫夌殑缁撴灉鐗╁寲涓哄敮涓€绾跨粍闆嗗悎锛屼緵杩囨护鍥俱€佹嫙鍚堝浘鍜屽尮閰嶅叡鍚屼娇鐢ㄣ�?
void retainEvidenceGroups(std::vector<LineGroup>& groups,
                          const std::vector<EvidenceGroup>& evidence) {
    std::vector<bool> retained(groups.size(), false);
    for (const auto& item : evidence) {
        if (item.index >= 0 && item.index < static_cast<int>(retained.size())) {
            retained[item.index] = true;
        }
    }

    std::vector<LineGroup> filtered;
    filtered.reserve(evidence.size());
    for (size_t index = 0; index < groups.size(); ++index) {
        if (retained[index]) {
            filtered.push_back(std::move(groups[index]));
        }
    }
    groups = std::move(filtered);
}

struct CandidatePair {
    int source = -1;
    int target = -1;
    double score = 0.0;
    double normalDifference = 0.0;
    double fittedOverlap = 0.0;
    double actualOverlap = 0.0;
    double sourceOverlapRatio = 0.0;
    double targetOverlapRatio = 0.0;
    double angleDifference = 0.0;
    bool strict = false;
    bool strong = false;
};

double positionToleranceForDirection(bool mainDirection, const Options& options) {
    // 绔栫洿杈归€氬父杈冪煭涓斿彧淇濈暀鍙充晶杈圭晫锛屽厑璁哥害涓ゅ€嶇殑灞€閮ㄤ綅缃宸紱
    // 鏁翠綋鍋忕Щ浠嶇敱澶氭潯绾跨殑涓€鑷存畫宸崟鐙瘖鏂€?
    return mainDirection ? options.final_position_tolerance_pixels
                         : options.final_position_tolerance_pixels * 2.0;
}

double spanOverlap(const EvidenceGroup& source, const EvidenceGroup& target) {
    const double sourceBegin = source.fittedTangentInterval.begin;
    const double sourceEnd = source.fittedTangentInterval.end;
    const double targetBegin = target.fittedTangentInterval.begin;
    const double targetEnd = target.fittedTangentInterval.end;
    return std::max(0.0, std::min(sourceEnd, targetEnd) - std::max(sourceBegin, targetBegin));
}

double fittedNormalAt(const EvidenceGroup& group, double tangentPosition) {
    const double signedAngle = signedAngleOffset180(group.angle, 0.0);
    return group.normalPosition +
           std::tan(radians(signedAngle)) * (tangentPosition - group.tangentCenter);
}

std::vector<CandidatePair> buildCandidates(const std::vector<EvidenceGroup>& source,
                                           const std::vector<EvidenceGroup>& target,
                                           bool mainDirection,
                                           const Options& options) {
    std::vector<CandidatePair> candidates;
    for (int sourceIndex = 0; sourceIndex < static_cast<int>(source.size()); ++sourceIndex) {
        for (int targetIndex = 0; targetIndex < static_cast<int>(target.size()); ++targetIndex) {
            const auto& sourceGroup = source[sourceIndex];
            const auto& targetGroup = target[targetIndex];
            const double normalDifference =
                std::abs(sourceGroup.normalPosition - targetGroup.normalPosition);
            const double actualOverlap =
                intervalOverlap(sourceGroup.tangentIntervals, targetGroup.tangentIntervals);
            const double fittedOverlap = spanOverlap(sourceGroup, targetGroup);
            const double sourceFittedSpan = sourceGroup.fittedTangentInterval.end -
                                            sourceGroup.fittedTangentInterval.begin;
            const double targetFittedSpan = targetGroup.fittedTangentInterval.end -
                                            targetGroup.fittedTangentInterval.begin;
            const double sourceOverlapRatio = ratioOrInvalid(fittedOverlap, sourceFittedSpan);
            const double targetOverlapRatio = ratioOrInvalid(fittedOverlap, targetFittedSpan);

            // 涓€渚х嚎娈靛彲鑳藉彧鏄彁鍙栦笉瀹屾暣锛涘彧鏈夊弻鍚戦噸鍙犵巼閮戒綆鏃舵墠涓㈠純鍊欓€夈�?
            if (fittedOverlap <= kEpsilon ||
                (sourceOverlapRatio < options.candidate_min_span_overlap_ratio &&
                 targetOverlapRatio < options.candidate_min_span_overlap_ratio)) {
                continue;
            }

            const double angleDifference =
                angleDifference180(sourceGroup.angle, targetGroup.angle);
            // 姘村钩涓绘柟鍚戠殑瑙掑害鏄‖绾︽潫锛涚珫鐩存柟鍚戜粎灏嗚搴︿綔涓鸿緟鍔╀唬浠枫€?
            if (mainDirection &&
                angleDifference > options.max_line_pair_angle_difference_degrees) {
                continue;
            }

            const double positionCost =
                clamp01(normalDifference / std::max(
                    kEpsilon, options.candidate_position_tolerance_pixels));
            const double overlapCost = 1.0 -
                                       clamp01(std::min(sourceOverlapRatio,
                                                        targetOverlapRatio));
            const double angleCost = clamp01(angleDifference / std::max(
                kEpsilon, options.max_line_pair_angle_difference_degrees));
            const double prominenceCost =
                std::abs(sourceGroup.prominence - targetGroup.prominence);
            const double cost = options.match_position_cost_weight * positionCost +
                                options.match_overlap_cost_weight * overlapCost +
                                options.match_angle_cost_weight * angleCost +
                                options.match_prominence_cost_weight * prominenceCost;

            CandidatePair candidate;
            candidate.source = sourceIndex;
            candidate.target = targetIndex;
            const double supportScore = clamp01(
                actualOverlap / std::max(kEpsilon,
                                         std::max(sourceGroup.actualLength,
                                                  targetGroup.actualLength)));
            // 鏈夋晥鍏叡鏀拺闀垮害涓轰富锛屼綅�?瑙掑�?鏄捐憲鎬т綔涓鸿緟鍔╅」銆?
            candidate.score = 0.70 * supportScore + 0.30 * std::max(0.001, 1.0 - cost);
            candidate.normalDifference = normalDifference;
            candidate.fittedOverlap = fittedOverlap;
            candidate.actualOverlap = actualOverlap;
            candidate.sourceOverlapRatio = sourceOverlapRatio;
            candidate.targetOverlapRatio = targetOverlapRatio;
            candidate.angleDifference = angleDifference;
            // 鍙屽悜閲嶅彔鐜囬兘浣庢墠鏄棤鏁堝€欓€夛紱鍗曚晶杈冧綆鏃朵繚鐣欎絾闄嶄綆璇勫垎�?
            candidate.strict = normalDifference <= positionToleranceForDirection(mainDirection, options) &&
                              !(sourceOverlapRatio < options.min_shorter_line_overlap_ratio &&
                                targetOverlapRatio < options.min_shorter_line_overlap_ratio);
            candidate.strong = sourceGroup.actualLength >= options.min_strong_line_actual_length_pixels &&
                               targetGroup.actualLength >= options.min_strong_line_actual_length_pixels &&
                               sourceGroup.prominence >= options.min_strong_peak_prominence &&
                               targetGroup.prominence >= options.min_strong_peak_prominence;
            candidates.push_back(candidate);
        }
    }
    return candidates;
}

struct MatchPath {
    std::vector<CandidatePair> selected;
    std::vector<CandidatePair> allCandidates;
};

MatchPath orderedPartialMatch(const std::vector<EvidenceGroup>& source,
                               const std::vector<EvidenceGroup>& target,
                               bool mainDirection,
                               const Options& options) {
    MatchPath output;
    output.allCandidates = buildCandidates(source, target, mainDirection, options);
    const int sourceCount = static_cast<int>(source.size());
    const int targetCount = static_cast<int>(target.size());
    if (sourceCount == 0 || targetCount == 0) {
        return output;
    }

    std::vector<std::vector<const CandidatePair*>> lookup(
        sourceCount, std::vector<const CandidatePair*>(targetCount, nullptr));
    for (const auto& candidate : output.allCandidates) {
        lookup[candidate.source][candidate.target] = &candidate;
    }
    std::vector<std::vector<double>> dp(
        sourceCount + 1, std::vector<double>(targetCount + 1, 0.0));
    std::vector<std::vector<unsigned char>> choice(
        sourceCount + 1, std::vector<unsigned char>(targetCount + 1, 0));
    for (int sourceIndex = 1; sourceIndex <= sourceCount; ++sourceIndex) {
        for (int targetIndex = 1; targetIndex <= targetCount; ++targetIndex) {
            double best = dp[sourceIndex - 1][targetIndex];
            unsigned char bestChoice = 1; // 璺宠�?source 绾跨�?
            if (dp[sourceIndex][targetIndex - 1] > best + kEpsilon) {
                best = dp[sourceIndex][targetIndex - 1];
                bestChoice = 2; // 璺宠�?target 绾跨�?
            }
            const CandidatePair* candidate = lookup[sourceIndex - 1][targetIndex - 1];
            if (candidate != nullptr &&
                dp[sourceIndex - 1][targetIndex - 1] + candidate->score >= best - kEpsilon) {
                best = dp[sourceIndex - 1][targetIndex - 1] + candidate->score;
                bestChoice = 3;
            }
            dp[sourceIndex][targetIndex] = best;
            choice[sourceIndex][targetIndex] = bestChoice;
        }
    }

    int sourceIndex = sourceCount;
    int targetIndex = targetCount;
    while (sourceIndex > 0 && targetIndex > 0) {
        const unsigned char selected = choice[sourceIndex][targetIndex];
        if (selected == 3) {
            const CandidatePair* candidate = lookup[sourceIndex - 1][targetIndex - 1];
            if (candidate != nullptr) {
                output.selected.push_back(*candidate);
            }
            --sourceIndex;
            --targetIndex;
        } else if (selected == 2) {
            --targetIndex;
        } else {
            --sourceIndex;
        }
    }
    std::reverse(output.selected.begin(), output.selected.end());
    return output;
}

// 绔栫洿绾跨粍褰撳墠鍧愭爣涓嬪彲鑳藉畬鍏ㄩ敊寮€锛屾棤娉曡繘鍏ユ櫘閫氬€欓€夐泦鍚堛€傝繖閲屼笉鏀瑰彉鍧愭爣锛?
// 浠呮楠屾槸鍚﹀瓨鍦ㄤ竴涓叡鍚岀殑娉曞悜骞崇Щ锛岃兘澶熻В閲婂弻鏂瑰缁勭湡瀹炴敮鎾戯紱鑻ュ瓨鍦紝
// 璇存槑鏄郴缁熸€ч敊浣嶏紝搴斿垽�?FAIL锛岃€屼笉鏄妸瀹冮檷绾т负璇佹嵁涓嶈冻銆?
struct OffsetMatchEvidence {
    double offset = 0.0;
    double supportedLength = 0.0;
    double sourceSupportedLength = 0.0;
    double targetSupportedLength = 0.0;
    double residualMax = 0.0;
    int matchedCount = 0;
};

OffsetMatchEvidence matchWithOffset(const std::vector<EvidenceGroup>& source,
                                     const std::vector<EvidenceGroup>& target,
                                     bool mainDirection,
                                     const Options& options,
                                     double offset) {
    OffsetMatchEvidence result;
    result.offset = offset;
    const auto candidates = buildCandidates(source, target, mainDirection, options);
    if (source.empty() || target.empty() || candidates.empty()) {
        return result;
    }

    std::vector<int> sourceOrder(source.size());
    std::vector<int> targetOrder(target.size());
    std::iota(sourceOrder.begin(), sourceOrder.end(), 0);
    std::iota(targetOrder.begin(), targetOrder.end(), 0);
    std::sort(sourceOrder.begin(), sourceOrder.end(), [&](int left, int right) {
        return source[left].normalPosition < source[right].normalPosition;
    });
    std::sort(targetOrder.begin(), targetOrder.end(), [&](int left, int right) {
        return target[left].normalPosition < target[right].normalPosition;
    });

    std::vector<std::vector<const CandidatePair*>> lookup(
        source.size(), std::vector<const CandidatePair*>(target.size(), nullptr));
    for (const auto& candidate : candidates) {
        lookup[candidate.source][candidate.target] = &candidate;
    }

    const int sourceCount = static_cast<int>(sourceOrder.size());
    const int targetCount = static_cast<int>(targetOrder.size());
    std::vector<std::vector<double>> dp(
        sourceCount + 1, std::vector<double>(targetCount + 1, 0.0));
    std::vector<std::vector<unsigned char>> choice(
        sourceCount + 1, std::vector<unsigned char>(targetCount + 1, 0));

    for (int sourcePosition = 1; sourcePosition <= sourceCount; ++sourcePosition) {
        for (int targetPosition = 1; targetPosition <= targetCount; ++targetPosition) {
            double best = dp[sourcePosition - 1][targetPosition];
            unsigned char bestChoice = 1;
            if (dp[sourcePosition][targetPosition - 1] > best + kEpsilon) {
                best = dp[sourcePosition][targetPosition - 1];
                bestChoice = 2;
            }
            const int sourceIndex = sourceOrder[sourcePosition - 1];
            const int targetIndex = targetOrder[targetPosition - 1];
            const CandidatePair* candidate = lookup[sourceIndex][targetIndex];
            if (candidate != nullptr) {
                const double residual = std::abs(
                    (target[targetIndex].normalPosition - source[sourceIndex].normalPosition) -
                    offset);
                if (residual <= positionToleranceForDirection(mainDirection, options) &&
                    dp[sourcePosition - 1][targetPosition - 1] + candidate->actualOverlap >
                        best + kEpsilon) {
                    best = dp[sourcePosition - 1][targetPosition - 1] +
                           candidate->actualOverlap;
                    bestChoice = 3;
                }
            }
            dp[sourcePosition][targetPosition] = best;
            choice[sourcePosition][targetPosition] = bestChoice;
        }
    }

    std::vector<double> residuals;
    int sourcePosition = sourceCount;
    int targetPosition = targetCount;
    while (sourcePosition > 0 && targetPosition > 0) {
        const unsigned char selected = choice[sourcePosition][targetPosition];
        if (selected == 3) {
            const int sourceIndex = sourceOrder[sourcePosition - 1];
            const int targetIndex = targetOrder[targetPosition - 1];
            const CandidatePair* candidate = lookup[sourceIndex][targetIndex];
            if (candidate != nullptr) {
                result.supportedLength += candidate->actualOverlap;
                result.sourceSupportedLength +=
                    std::min(source[sourceIndex].actualLength, candidate->actualOverlap);
                result.targetSupportedLength +=
                    std::min(target[targetIndex].actualLength, candidate->actualOverlap);
                residuals.push_back(std::abs(
                    (target[targetIndex].normalPosition - source[sourceIndex].normalPosition) -
                    offset));
                ++result.matchedCount;
            }
            --sourcePosition;
            --targetPosition;
        } else if (selected == 2) {
            --targetPosition;
        } else {
            --sourcePosition;
        }
    }
    result.residualMax = residuals.empty() ? 0.0 : percentile(residuals, 1.0);
    return result;
}

OffsetMatchEvidence bestNonZeroOffsetMatch(const std::vector<EvidenceGroup>& source,
                                           const std::vector<EvidenceGroup>& target,
                                           bool mainDirection,
                                           const Options& options) {
    OffsetMatchEvidence best;
    const auto candidates = buildCandidates(source, target, mainDirection, options);
    for (const auto& candidate : candidates) {
        const double offset = target[candidate.target].normalPosition -
                              source[candidate.source].normalPosition;
        if (std::abs(offset) <= positionToleranceForDirection(mainDirection, options)) {
            continue;
        }
        const OffsetMatchEvidence evidence =
            matchWithOffset(source, target, mainDirection, options, offset);
        if (evidence.supportedLength > best.supportedLength + kEpsilon ||
            (std::abs(evidence.supportedLength - best.supportedLength) <= kEpsilon &&
             evidence.matchedCount > best.matchedCount)) {
            best = evidence;
        }
    }
    return best;
}

void applyStrictPairValidation(MatchPath& path,
                               const std::vector<EvidenceGroup>& source,
                               const std::vector<EvidenceGroup>& target,
                               bool mainDirection,
                               const Options& options) {
    for (auto& pair : path.selected) {
        const auto& sourceGroup = source[pair.source];
        const auto& targetGroup = target[pair.target];
        pair.normalDifference = std::abs(
            targetGroup.normalPosition - sourceGroup.normalPosition);
        pair.strict = pair.normalDifference <= positionToleranceForDirection(mainDirection, options) &&
                      !(pair.sourceOverlapRatio < options.min_shorter_line_overlap_ratio &&
                        pair.targetOverlapRatio < options.min_shorter_line_overlap_ratio);
        if (mainDirection) {
            pair.strict = pair.strict &&
                          pair.angleDifference <=
                              options.max_line_pair_angle_difference_degrees;
        }
    }
}

DirectionResult evaluateDirection(const std::vector<EvidenceGroup>& source,
                                   const std::vector<EvidenceGroup>& target,
                                   bool mainDirection,
                                   const Options& options) {
    DirectionResult result;
    result.source_eligible_line_groups = static_cast<int>(source.size());
    result.target_eligible_line_groups = static_cast<int>(target.size());
    MatchPath path = orderedPartialMatch(source, target, mainDirection, options);
    applyStrictPairValidation(path, source, target, mainDirection, options);
    result.candidate_pairs = static_cast<int>(path.allCandidates.size());
    const OffsetMatchEvidence absoluteEvidence =
        matchWithOffset(source, target, mainDirection, options, 0.0);
    const OffsetMatchEvidence shiftedEvidence =
        bestNonZeroOffsetMatch(source, target, mainDirection, options);
    const double shiftedSupportMargin =
        std::max(1.0, absoluteEvidence.supportedLength * 0.10);
    const bool hasSystematicOffset =
        shiftedEvidence.matchedCount >= 2 &&
        std::abs(shiftedEvidence.offset) > positionToleranceForDirection(mainDirection, options) &&
        shiftedEvidence.residualMax <= positionToleranceForDirection(mainDirection, options) &&
        shiftedEvidence.supportedLength >
            absoluteEvidence.supportedLength + shiftedSupportMargin;

    double sourceTotalLength = 0.0;
    double targetTotalLength = 0.0;
    for (const auto& group : source) {
        sourceTotalLength += group.actualLength;
    }
    for (const auto& group : target) {
        targetTotalLength += group.actualLength;
    }
    std::vector<bool> sourceAssigned(source.size(), false);
    std::vector<bool> targetAssigned(target.size(), false);
    std::vector<double> angleDifferences;
    double ambiguousLength = 0.0;
    for (const auto& selected : path.selected) {
        sourceAssigned[selected.source] = true;
        targetAssigned[selected.target] = true;
        if (selected.strict) {
            ++result.accepted_matches;
            result.source_matched_actual_length += source[selected.source].actualLength;
            result.target_matched_actual_length += target[selected.target].actualLength;
            angleDifferences.push_back(selected.angleDifference);
        } else if (selected.strong) {
            ++result.strong_conflict_count;
            result.source_strong_conflict_actual_length += source[selected.source].actualLength;
            result.target_strong_conflict_actual_length += target[selected.target].actualLength;
        }
        if (!selected.strict) {
            const double sourceLength = source[selected.source].actualLength;
            const double targetLength = target[selected.target].actualLength;
            const double shorterFittedSpan = std::min(
                source[selected.source].fittedTangentInterval.end -
                    source[selected.source].fittedTangentInterval.begin,
                target[selected.target].fittedTangentInterval.end -
                    target[selected.target].fittedTangentInterval.begin);
            const auto recordRejection = [&](StrictRejectionStats& statistics) {
                ++statistics.count;
                statistics.source_actual_length += sourceLength;
                statistics.target_actual_length += targetLength;
            };
            if (selected.normalDifference > positionToleranceForDirection(mainDirection, options)) {
                recordRejection(result.strict_position_rejections);
            }
            if (selected.sourceOverlapRatio < options.min_shorter_line_overlap_ratio &&
                selected.targetOverlapRatio < options.min_shorter_line_overlap_ratio) {
                recordRejection(result.strict_overlap_rejections);
            }
            if (mainDirection &&
                selected.angleDifference > options.max_line_pair_angle_difference_degrees) {
                recordRejection(result.strict_angle_rejections);
            }
        }
        bool ambiguous = false;
        for (const auto& candidate : path.allCandidates) {
            if (candidate.source == selected.source && candidate.target == selected.target) {
                continue;
            }
            if ((candidate.source == selected.source || candidate.target == selected.target) &&
                std::abs(candidate.score - selected.score) <= options.ambiguity_score_margin) {
                ambiguous = true;
                break;
            }
        }
        if (ambiguous) {
            ++result.ambiguous_match_count;
            ambiguousLength += std::max(source[selected.source].actualLength,
                                        target[selected.target].actualLength);
        }
    }
    for (int index = 0; index < static_cast<int>(source.size()); ++index) {
        if (!sourceAssigned[index]) {
            result.source_unmatched_actual_length += source[index].actualLength;
        }
    }
    for (int index = 0; index < static_cast<int>(target.size()); ++index) {
        if (!targetAssigned[index]) {
            result.target_unmatched_actual_length += target[index].actualLength;
        }
    }
    result.source_match_ratio = ratioOrInvalid(result.accepted_matches, source.size());
    result.target_match_ratio = ratioOrInvalid(result.accepted_matches, target.size());
    result.source_strong_conflict_length_ratio =
        ratioOrInvalid(result.source_strong_conflict_actual_length, sourceTotalLength);
    result.target_strong_conflict_length_ratio =
        ratioOrInvalid(result.target_strong_conflict_actual_length, targetTotalLength);
    result.source_unmatched_length_ratio =
        ratioOrInvalid(result.source_unmatched_actual_length, sourceTotalLength);
    result.target_unmatched_length_ratio =
        ratioOrInvalid(result.target_unmatched_actual_length, targetTotalLength);
    result.ambiguous_actual_length_ratio =
        ratioOrInvalid(ambiguousLength, std::max(sourceTotalLength, targetTotalLength));
    result.matched_angle_difference_mean_degrees =
        percentile(angleDifferences, 0.50);
    result.matched_angle_difference_max_degrees =
        percentile(angleDifferences, 1.0);

    if (source.empty() || target.empty()) {
        result.status = "INSUFFICIENT";
        return result;
    }

    // Non-zero offset is diagnostic evidence only and can never create a PASS match.
    if (hasSystematicOffset) {
        result.status = "FAIL";
        return result;
    }

    // A fully supported non-strict pair is an explicit absolute-position conflict.
    const bool hasExplicitPositionConflict =
        result.accepted_matches == 0 &&
        result.strong_conflict_count > 0 &&
        result.source_strong_conflict_length_ratio > 0.0 &&
        result.target_strong_conflict_length_ratio > 0.0;
    if (hasExplicitPositionConflict) {
        result.status = "FAIL";
        return result;
    }

    // No geometrically comparable pair means insufficient evidence, not a direct failure.
    if (path.allCandidates.empty() || result.accepted_matches == 0) {
        result.status = "INSUFFICIENT";
        return result;
    }

    const bool hasBroadVerticalUnmatchedSupport =
        !mainDirection &&
        options.max_vertical_unmatched_length_ratio > 0.0 &&
        result.source_unmatched_length_ratio >=
            options.max_vertical_unmatched_length_ratio &&
        result.target_unmatched_length_ratio >=
            options.max_vertical_unmatched_length_ratio;
    if (hasBroadVerticalUnmatchedSupport) {
        result.status = "INSUFFICIENT";
        return result;
    }

    result.status = "PASS";
    return result;
}

void drawGroup(cv::Mat& image,
               const LineGroup& group,
               const cv::Scalar& color,
               int thickness) {
    for (const auto& fragment : group.fragments) {
        cv::line(image,
                 cv::Point(cvRound(fragment.first.x), cvRound(fragment.first.y)),
                 cv::Point(cvRound(fragment.second.x), cvRound(fragment.second.y)),
                 color,
                 thickness,
                  cv::LINE_AA);
    }
}

// 缁樺埗绾跨粍鎷熷悎鍚庣殑瀹屾暣鏀拺璺ㄥ害锛涘畠鐢ㄤ簬璇婃柇閫昏緫鍚堝苟缁撴灉锛屼笉淇敼鐪熷疄杈圭紭鍍忕礌�?
void drawFittedGroup(cv::Mat& image,
                     const LineGroup& group,
                     const cv::Scalar& color,
                     int thickness) {
    if (group.fragments.empty()) {
        return;
    }
    double lower = std::numeric_limits<double>::max();
    double upper = std::numeric_limits<double>::lowest();
    for (const auto& fragment : group.fragments) {
        lower = std::min(lower,
                         std::min(dot(fragment.first - group.origin, group.direction),
                                  dot(fragment.second - group.origin, group.direction)));
        upper = std::max(upper,
                         std::max(dot(fragment.first - group.origin, group.direction),
                                  dot(fragment.second - group.origin, group.direction)));
    }
    const cv::Point2d first = group.origin + group.direction * lower;
    const cv::Point2d second = group.origin + group.direction * upper;
    cv::line(image,
             cv::Point(cvRound(first.x), cvRound(first.y)),
             cv::Point(cvRound(second.x), cvRound(second.y)),
             color,
             thickness,
             cv::LINE_AA);
}

void drawFittedEvidenceGroup(cv::Mat& image,
                             const LineGroup& group,
                             const EvidenceGroup& evidence,
                             double referenceAngle,
                             const cv::Scalar& color,
                             int thickness) {
    if (evidence.axis != Axis::VERTICAL || evidence.tangentIntervals.empty()) {
        drawFittedGroup(image, group, color, thickness);
        return;
    }
    const cv::Point2d horizontalDirection(std::cos(radians(referenceAngle)),
                                           std::sin(radians(referenceAngle)));
    const cv::Point2d verticalDirection(-horizontalDirection.y, horizontalDirection.x);
    const double begin = evidence.tangentIntervals.front().begin;
    const double end = evidence.tangentIntervals.back().end;
    const cv::Point2d first = horizontalDirection * evidence.normalPosition +
                              verticalDirection * begin;
    const cv::Point2d second = horizontalDirection * evidence.normalPosition +
                               verticalDirection * end;
    cv::line(image,
             cv::Point(cvRound(first.x), cvRound(first.y)),
             cv::Point(cvRound(second.x), cvRound(second.y)),
             color,
             thickness,
             cv::LINE_AA);
}

cv::Mat renderGroups(const cv::Size& size,
                     const std::vector<LineGroup>& groups,
                     const std::vector<EvidenceGroup>& renderable,
                     const cv::Scalar& color) {
    cv::Mat image = cv::Mat::zeros(size, CV_8UC3);
    for (const auto& item : renderable) {
        drawGroup(image, groups[item.index], color, 1);
    }
    return image;
}

// 缁樺埗绾挎妫€娴嬪櫒鐨勭洿鎺ヨ緭鍑猴紝涓嶅寘鍚渶灏忛暱搴︺€佺嚎缁勫悎骞舵垨缁撴瀯璇佹嵁绛涢€夈�?
cv::Mat renderInitialSegments(const cv::Size& size,
                              const std::vector<cv::Vec4f>& segments,
                              const cv::Scalar& color) {
    cv::Mat image = cv::Mat::zeros(size, CV_8UC3);
    for (const auto& segment : segments) {
        cv::line(image,
                 cv::Point(cvRound(segment[0]), cvRound(segment[1])),
                 cv::Point(cvRound(segment[2]), cvRound(segment[3])),
                 color,
                 1,
                 cv::LINE_AA);
    }
    return image;
}

cv::Mat renderFittedGroups(const cv::Size& size,
                           const std::vector<LineGroup>& groups,
                           const std::vector<EvidenceGroup>& renderable,
                           double referenceAngle,
                           const cv::Scalar& color) {
    cv::Mat image = cv::Mat::zeros(size, CV_8UC3);
    for (const auto& item : renderable) {
        drawFittedEvidenceGroup(image,
                                groups[item.index],
                                item,
                                referenceAngle,
                                color,
                                1);
    }
    return image;
}

void renderOverlay(const cv::Size& size,
                   const std::vector<LineGroup>& sourceGroups,
                   const std::vector<LineGroup>& targetGroups,
                   const std::vector<EvidenceGroup>& sourceEvidence,
                   const std::vector<EvidenceGroup>& targetEvidence,
                   const Options& options,
                   double referenceAngle,
                   cv::Mat& overlay) {
    overlay = cv::Mat::zeros(size, CV_8UC3);
    const cv::Point2d horizontalDirection(std::cos(radians(referenceAngle)),
                                           std::sin(radians(referenceAngle)));
    const cv::Point2d verticalDirection(-horizontalDirection.y, horizontalDirection.x);
    // matched 鏄瘖鏂鍥捐€岄潪浠呮垚鍔熺嚎瀵硅鍥撅細淇濈暀鎵€鏈夊悎鏍兼嫙鍚堢嚎锛?
    // 浣挎湭鍖归厤鍜屽啿绐佺嚎浠嶅彲瑙侊紱榛勮壊琛ㄧず涓荤嚎鐨勭湡瀹為噸鍙犳垨绔栫嚎鐨勫彈绾︽潫鎷熷悎鍖呯粶閲嶅彔�?
    for (const auto& item : sourceEvidence) {
        drawGroup(overlay, sourceGroups[item.index], kSourceLineColor, 1);
    }
    for (const auto& item : targetEvidence) {
        drawGroup(overlay, targetGroups[item.index], kTargetLineColor, 1);
    }
    const auto drawAccepted = [&](const std::vector<EvidenceGroup>& source,
                                  const std::vector<EvidenceGroup>& target,
                                  Axis axis) {
        std::vector<EvidenceGroup> sourceAxis;
        std::vector<EvidenceGroup> targetAxis;
        for (const auto& item : source) {
            if (item.axis == axis) {
                sourceAxis.push_back(item);
            }
        }
        for (const auto& item : target) {
            if (item.axis == axis) {
                targetAxis.push_back(item);
            }
        }
        // Keep rendering on the same ordered path used by evaluateDirection.
        // Without this, the original group traversal order could select a
        // different partial matching path and make matched.png disagree with
        // the accepted-match counts in the CSV.
        const auto normalSort = [](const EvidenceGroup& first,
                                   const EvidenceGroup& second) {
            return first.normalPosition < second.normalPosition;
        };
        std::sort(sourceAxis.begin(), sourceAxis.end(), normalSort);
        std::sort(targetAxis.begin(), targetAxis.end(), normalSort);
        const bool mainDirection = axis == Axis::HORIZONTAL;
        MatchPath path = orderedPartialMatch(sourceAxis,
                                             targetAxis,
                                             mainDirection,
                                             options);
        applyStrictPairValidation(path,
                                  sourceAxis,
                                  targetAxis,
                                  mainDirection,
                                  options);
        for (const auto& pair : path.selected) {
            if (!pair.strict) {
                continue;
            }
            const auto& sourceItem = sourceAxis[pair.source];
            const auto& targetItem = targetAxis[pair.target];

            // 榛勮壊娈佃〃绀烘渶缁堢敤浜庡垽瀹氱殑涓ゆ潯鎷熷悎绾垮寘缁滀氦闆嗭紝涓嶅彈涓棿鐪熷疄鏂褰卞搷銆?
            const cv::Point2d tangent = axis == Axis::HORIZONTAL ? horizontalDirection
                                                                     : verticalDirection;
            const cv::Point2d normal = axis == Axis::HORIZONTAL ? verticalDirection
                                                                   : horizontalDirection;
            if (std::abs(sourceItem.normalPosition - targetItem.normalPosition) > 0.5) {
                continue;
            }
            // Use the intersection of real fragment support; fitted envelopes may span gaps.
            size_t sourceInterval = 0;
            size_t targetInterval = 0;
            while (sourceInterval < sourceItem.tangentIntervals.size() &&
                   targetInterval < targetItem.tangentIntervals.size()) {
                const double begin = std::max(
                    sourceItem.tangentIntervals[sourceInterval].begin,
                    targetItem.tangentIntervals[targetInterval].begin);
                const double end = std::min(
                    sourceItem.tangentIntervals[sourceInterval].end,
                    targetItem.tangentIntervals[targetInterval].end);
                if (end > begin) {
                    const cv::Point2d normalPosition = normal * sourceItem.normalPosition;
                    const cv::Point2d first = tangent * begin + normalPosition;
                    const cv::Point2d second = tangent * end + normalPosition;
                    cv::line(overlay,
                             cv::Point(cvRound(first.x), cvRound(first.y)),
                             cv::Point(cvRound(second.x), cvRound(second.y)),
                             kMatchedOverlapColor,
                             1,
                             cv::LINE_AA);
                }
                if (sourceItem.tangentIntervals[sourceInterval].end <
                    targetItem.tangentIntervals[targetInterval].end) {
                    ++sourceInterval;
                } else {
                    ++targetInterval;
                }
            }
        }
    };
    drawAccepted(sourceEvidence, targetEvidence, Axis::HORIZONTAL);
    drawAccepted(sourceEvidence, targetEvidence, Axis::VERTICAL);
}

} // namespace

bool evaluate(const Options& options,
              const cv::Mat& sourceImage,
              const cv::Mat& targetImage,
              Result& result) {
    result = Result{};
    if (!options.enabled) {
        result.status = "NOT_RUN";
        result.message = "disabled";
        return true;
    }

    // 闃舵�?1锛氭瀯寤虹伆搴﹀浘鍜屽墠鏅紝骞跺厛绛涢€夐暱鏉″舰閫傜敤鎬с�?
    cv::Mat sourceGray;
    cv::Mat targetGray;
    if (!toGray8(sourceImage, sourceGray) || !toGray8(targetImage, targetGray)) {
        result.status = "INSUFFICIENT";
        result.message = "[input/foreground] cannot build grayscale images";
        return false;
    }
    cv::Mat sourceShapeMask;
    cv::Mat targetShapeMask;
    cv::Mat sourceVisibilityMask;
    cv::Mat targetVisibilityMask;
    if (!buildForegroundMasks(sourceGray,
                              options.visibility_threshold,
                              sourceShapeMask,
                              sourceVisibilityMask) ||
        !buildForegroundMasks(targetGray,
                              options.visibility_threshold,
                              targetShapeMask,
                              targetVisibilityMask)) {
        result.status = "INSUFFICIENT";
        result.message = "[input/foreground] cannot build foreground masks";
        return false;
    }
    const ForegroundMetrics sourceMetrics = measureForeground(sourceShapeMask);
    const ForegroundMetrics targetMetrics = measureForeground(targetShapeMask);
    result.source_foreground_elongation_ratio = sourceMetrics.elongation;
    result.target_foreground_elongation_ratio = targetMetrics.elongation;
    result.source_axis_occupancy = sourceMetrics.axisOccupancy;
    result.target_axis_occupancy = targetMetrics.axisOccupancy;
    result.source_centerline_deviation_ratio = sourceMetrics.centerlineDeviationRatio;
    result.target_centerline_deviation_ratio = targetMetrics.centerlineDeviationRatio;
    result.source_foreground_long_side = sourceMetrics.longSide;
    result.target_foreground_long_side = targetMetrics.longSide;
    // 灞€閮ㄨ閲庡彲鍙寘鍚暱鏉＄殑涓€娈碉紝鍥犳�?source/target 浠讳竴渚х粏闀垮嵆鍙繘鍏ョ粨鏋勯獙璇併€?
    if (sourceMetrics.elongation < options.min_foreground_elongation_ratio &&
        targetMetrics.elongation < options.min_foreground_elongation_ratio) {
        result.status = "INSUFFICIENT";
        result.message = "[strip suitability] neither foreground is elongated for strip structure";
        return true;
    }
    if (sourceMetrics.axisOccupancy < options.min_axis_occupancy ||
        targetMetrics.axisOccupancy < options.min_axis_occupancy) {
        result.status = "INSUFFICIENT";
        result.message = "[strip suitability] foreground is not continuous along its principal axis";
        return true;
    }
    if (sourceMetrics.centerlineDeviationRatio > options.max_centerline_deviation_ratio ||
        targetMetrics.centerlineDeviationRatio > options.max_centerline_deviation_ratio ||
        sourceMetrics.centerlineDeviationRatio < 0.0 ||
        targetMetrics.centerlineDeviationRatio < 0.0) {
        result.status = "INSUFFICIENT";
        result.message = "[strip suitability] foreground centerline is not sufficiently straight";
        return true;
    }

    // 闃舵�?2锛氫互鍘熷鍧愭爣寤虹珛鍏卞悓鐢诲竷銆傚弻鏂瑰叡浜弬鑰冩柟鍚戯紝浣嗕笉鍋氱浉瀵规棆姝ｆ垨骞崇Щ琛ュ伩銆?
    CommonCanvas canvas;
    std::string canvasError;
    if (!buildOriginalCoordinateCanvas(sourceGray.size(),
                                       targetGray.size(),
                                       options,
                                       canvas,
                                       canvasError)) {
        result.status = "INSUFFICIENT";
        result.message = "[input/foreground] " + canvasError;
        return false;
    }
    result.common_canvas_width = canvas.size.width;
    result.common_canvas_height = canvas.size.height;
    result.common_canvas_offset_x = canvas.offset.x;
    result.common_canvas_offset_y = canvas.offset.y;

    cv::Mat warpedSourceGray;
    cv::Mat warpedSourceVisibility;
    if (!warpImage(sourceGray,
                   canvas.size,
                   canvas.sourceTransform,
                   cv::INTER_LINEAR,
                   warpedSourceGray) ||
        !warpImage(sourceVisibilityMask,
                   canvas.size,
                   canvas.sourceTransform,
                   cv::INTER_NEAREST,
                   warpedSourceVisibility)) {
        result.status = "INSUFFICIENT";
        result.message = "[input/foreground] cannot place source on original-coordinate canvas";
        return false;
    }
    cv::Mat canvasTargetGray;
    cv::Mat canvasTargetVisibility;
    pasteTarget(targetGray, canvas, canvasTargetGray);
    pasteTarget(targetVisibilityMask, canvas, canvasTargetVisibility);
    cv::Mat commonVisibilityMask;
    cv::bitwise_and(warpedSourceVisibility,
                    canvasTargetVisibility,
                    commonVisibilityMask);
    result.common_visibility_pixels = cv::countNonZero(commonVisibilityMask);
    result.source_visibility_pixels = cv::countNonZero(warpedSourceVisibility);
    result.target_visibility_pixels = cv::countNonZero(canvasTargetVisibility);
    result.visibility_area_ratio = ratioOrInvalid(
        std::min(result.source_visibility_pixels, result.target_visibility_pixels),
        std::max(result.source_visibility_pixels, result.target_visibility_pixels));
    result.visibility_overlap_containment = ratioOrInvalid(
        result.common_visibility_pixels,
        std::min(result.source_visibility_pixels, result.target_visibility_pixels));
    result.source_visibility_ratio = ratioOrInvalid(result.common_visibility_pixels,
                                                    result.source_visibility_pixels);
    result.target_visibility_ratio = ratioOrInvalid(result.common_visibility_pixels,
                                                    result.target_visibility_pixels);
    if (result.common_visibility_pixels <= 0) {
        result.status = "INSUFFICIENT";
        result.message = "[input/foreground] empty common visibility";
        return true;
    }

    // 闃舵�?3锛欵DLines 鐩存帴鍦ㄥ叡鍚屽彲瑙佸尯鍩熷唴鐨勭伆搴﹀浘涓婃娴嬪垵濮嬬墖娈点�?
    cv::Mat sourceLineInput = warpedSourceGray.clone();
    cv::Mat targetLineInput = canvasTargetGray.clone();
    sourceLineInput.setTo(0, warpedSourceVisibility == 0);
    targetLineInput.setTo(0, canvasTargetVisibility == 0);

    // 闃舵�?4锛氬垵姝ュ垎缁勫彧璐熻矗浠庡叏鏂瑰悜鐗囨浼拌鍙屾柟鍙俊涓绘柟鍚戙€?
    const ForegroundMetrics warpedSourceMetrics = measureForeground(warpedSourceVisibility);
    const ForegroundMetrics canvasTargetMetrics = measureForeground(canvasTargetVisibility);
    result.source_foreground_long_side = warpedSourceMetrics.longSide;
    result.target_foreground_long_side = canvasTargetMetrics.longSide;
    cv::Mat sourceGradX;
    cv::Mat sourceGradY;
    cv::Mat sourceGradient;
    cv::Mat targetGradX;
    cv::Mat targetGradY;
    cv::Mat targetGradient;
    computeGradient(warpedSourceGray, sourceGradX, sourceGradY, sourceGradient);
    computeGradient(canvasTargetGray, targetGradX, targetGradY, targetGradient);
    GroupBuildResult sourceBuild = buildLineGroups(sourceLineInput, options);
    GroupBuildResult targetBuild = buildLineGroups(targetLineInput, options);
    result.initial_source_line_segments = renderInitialSegments(canvas.size,
                                                                 sourceBuild.initialSegments,
                                                                 kSourceLineColor);
    result.initial_target_line_segments = renderInitialSegments(canvas.size,
                                                                 targetBuild.initialSegments,
                                                                 kTargetLineColor);
    assignProminence(sourceBuild.groups, sourceGradient);
    assignProminence(targetBuild.groups, targetGradient);
    result.source_fragment_count = sourceBuild.fragmentCount;
    result.target_fragment_count = targetBuild.fragmentCount;
    result.source_line_group_count = static_cast<int>(sourceBuild.groups.size());
    result.target_line_group_count = static_cast<int>(targetBuild.groups.size());
    MainDirection sourceMain = estimateMainDirection(sourceBuild.groups,
                                                     warpedSourceMetrics.longSide,
                                                     options);
    MainDirection targetMain = estimateMainDirection(targetBuild.groups,
                                                     canvasTargetMetrics.longSide,
                                                     options);
    for (const auto& group : sourceBuild.groups) {
        if (group.valid) {
            ++result.source_valid_line_group_count;
            if (group.mainCandidate) {
                ++result.source_main_line_group_count;
            }
        }
    }
    for (const auto& group : targetBuild.groups) {
        if (group.valid) {
            ++result.target_valid_line_group_count;
            if (group.mainCandidate) {
                ++result.target_main_line_group_count;
            }
        }
    }
    result.source_main_direction_reliable = sourceMain.reliable;
    result.target_main_direction_reliable = targetMain.reliable;
    result.source_main_direction_degrees = sourceMain.angle;
    result.target_main_direction_degrees = targetMain.angle;
    result.source_main_direction_support_ratio = sourceMain.supportRatio;
    result.target_main_direction_support_ratio = targetMain.supportRatio;
    result.source_main_direction_spread_degrees = sourceMain.spread;
    result.target_main_direction_spread_degrees = targetMain.spread;
    result.source_main_direction_margin = sourceMain.margin;
    result.target_main_direction_margin = targetMain.margin;
    result.source_main_max_actual_length_ratio = sourceMain.maxActualLengthRatio;
    result.target_main_max_actual_length_ratio = targetMain.maxActualLengthRatio;
    if (!sourceMain.reliable || !targetMain.reliable) {
        result.status = "INSUFFICIENT";
        result.message = "[direction/line evidence] main direction evidence is insufficient or ambiguous";
        return true;
    }

    result.main_direction_difference_degrees =
        angleDifference180(sourceMain.angle, targetMain.angle);
    if (result.main_direction_difference_degrees >
        options.max_main_direction_difference_degrees) {
        result.status = "FAIL";
        result.message = "[direction/line evidence] source and target main directions disagree";
        result.reference_direction_degrees = targetMain.angle;
        return true;
    }
    // 闃舵�?5锛氫富鏂瑰悜涓€鑷村悗锛屽宸查厤鍑?source �?target 鏂藉姞鍚屼竴涓弬鑰冩棆杞�?
    // 杩欎細鎶婂叡鍚屼富鏂瑰悜鏀惧埌姘村钩杞达紝浣嗕笉浼氭秷闄や袱鍥惧凡鏈夌殑鐩稿鏃嬭浆銆佹硶鍚戝亸绉绘垨鍒囧悜閿欎綅�?
    result.reference_direction_degrees = targetMain.angle;
    ReferenceRotationCanvas referenceCanvas;
    std::string referenceCanvasError;
    if (!buildReferenceRotationCanvas(canvas.size,
                                      targetMain.angle,
                                      options,
                                      referenceCanvas,
                                      referenceCanvasError)) {
        result.status = "INSUFFICIENT";
        result.message = "[direction/line evidence] " + referenceCanvasError;
        return false;
    }
    cv::Mat referenceSourceGray;
    cv::Mat referenceTargetGray;
    cv::Mat referenceSourceVisibility;
    cv::Mat referenceTargetVisibility;
    if (!warpImage(warpedSourceGray,
                   referenceCanvas.size,
                   referenceCanvas.transform,
                   cv::INTER_LINEAR,
                   referenceSourceGray) ||
        !warpImage(canvasTargetGray,
                   referenceCanvas.size,
                   referenceCanvas.transform,
                   cv::INTER_LINEAR,
                   referenceTargetGray) ||
        !warpImage(warpedSourceVisibility,
                   referenceCanvas.size,
                   referenceCanvas.transform,
                   cv::INTER_NEAREST,
                   referenceSourceVisibility) ||
        !warpImage(canvasTargetVisibility,
                   referenceCanvas.size,
                   referenceCanvas.transform,
                   cv::INTER_NEAREST,
                   referenceTargetVisibility)) {
        result.status = "INSUFFICIENT";
        result.message =
            "[direction/line evidence] cannot rotate source and target into common reference frame";
        return false;
    }
    cv::Mat referenceSourceGradX;
    cv::Mat referenceSourceGradY;
    cv::Mat referenceSourceGradient;
    cv::Mat referenceTargetGradX;
    cv::Mat referenceTargetGradY;
    cv::Mat referenceTargetGradient;
    computeGradient(referenceSourceGray,
                    referenceSourceGradX,
                    referenceSourceGradY,
                    referenceSourceGradient);
    computeGradient(referenceTargetGray,
                    referenceTargetGradX,
                    referenceTargetGradY,
                    referenceTargetGradient);
    const std::vector<cv::Vec4f> sourceInitialSegments =
        transformLineSegments(sourceBuild.initialSegments, referenceCanvas.transform);
    const std::vector<cv::Vec4f> targetInitialSegments =
        transformLineSegments(targetBuild.initialSegments, referenceCanvas.transform);
    // initial 浠嶆槸妫€娴嬪櫒鍘熷鐗囨锛屽彧鏄悓姝ユ樉绀哄湪鍙屾柟鍏辩敤鐨勬棆杞潗鏍囩郴�?
    result.initial_source_line_segments = renderInitialSegments(referenceCanvas.size,
                                                                 sourceInitialSegments,
                                                                 kSourceLineColor);
    result.initial_target_line_segments = renderInitialSegments(referenceCanvas.size,
                                                                 targetInitialSegments,
                                                                 kTargetLineColor);
    sourceBuild = buildReferenceLineGroups(sourceInitialSegments, 0.0, options);
    targetBuild = buildReferenceLineGroups(targetInitialSegments, 0.0, options);
    // 鍙湁鍒濇鍒嗙粍宸茬粡浜х敓鈥滄柇瑁傛瘮渚嬭秴闄愨€濈殑鏃犳晥缁勬椂锛屾墠鍦ㄨ瘉鎹瓫閫夊墠
    // 灏濊瘯鍚堝苟鐩搁偦鏂缁勩€傛甯告牱鏈户缁部鐢ㄧ瓫閫夊悗鐨勪簩娆￠噸鍒嗙粍锛岄伩鍏嶆�?
    // 鏈潵鐙珛鐨勫钩琛岃竟鎻愬墠鍚堝苟�?
    const auto hasLargeGapGroup = [&](const std::vector<LineGroup>& groups) {
        return std::any_of(groups.begin(), groups.end(), [&](const LineGroup& group) {
            return !group.valid && group.gapRatio > options.max_line_group_gap_ratio;
        });
    };
    if (hasLargeGapGroup(sourceBuild.groups) || hasLargeGapGroup(targetBuild.groups)) {
        sourceBuild.groups = refitSeparatedLineGroups(std::move(sourceBuild.groups),
                                                       0.0,
                                                       true,
                                                       options);
        targetBuild.groups = refitSeparatedLineGroups(std::move(targetBuild.groups),
                                                       0.0,
                                                       true,
                                                       options);
    }
    result.source_fragment_count = sourceBuild.fragmentCount;
    result.target_fragment_count = targetBuild.fragmentCount;

    const DirectionProfiles sourceProfiles = buildDirectionProfiles(referenceSourceGradX,
                                                                     referenceSourceGradY,
                                                                     referenceSourceVisibility,
                                                                     0.0,
                                                                     options);
    const DirectionProfiles targetProfiles = buildDirectionProfiles(referenceTargetGradX,
                                                                     referenceTargetGradY,
                                                                      referenceTargetVisibility,
                                                                     0.0,
                                                                     options);
    applyProfileProminence(sourceBuild.groups,
                           0.0,
                           sourceProfiles,
                           options);
    applyProfileProminence(targetBuild.groups,
                           0.0,
                           targetProfiles,
                           options);
    markReferenceMainDirectionGroups(sourceBuild.groups,
                                     warpedSourceMetrics.longSide,
                                     options);
    markReferenceMainDirectionGroups(targetBuild.groups,
                                     canvasTargetMetrics.longSide,
                                     options);
    // 鍏卞悓鍙傝€冪郴宸蹭护闀胯竟娌挎按骞宠酱锛屾晠鍓嶆櫙璐ㄥ績鐨?y 鍧愭爣鍙垽鍒垚瀵圭矖闀胯竟鐨勫唴澶栦晶�?
    const cv::Moments sourceVisibilityMoments = cv::moments(referenceSourceVisibility, true);
    const cv::Moments targetVisibilityMoments = cv::moments(referenceTargetVisibility, true);
    const double sourceLongitudinalNormalCenter = sourceVisibilityMoments.m00 > kEpsilon
                                                      ? sourceVisibilityMoments.m01 /
                                                            sourceVisibilityMoments.m00
                                                      : referenceCanvas.size.height * 0.5;
    const double targetLongitudinalNormalCenter = targetVisibilityMoments.m00 > kEpsilon
                                                      ? targetVisibilityMoments.m01 /
                                                            targetVisibilityMoments.m00
                                                      : referenceCanvas.size.height * 0.5;
    auto sourceEvidenceAll = buildEvidence(sourceBuild.groups,
                                           0.0,
                                           sourceLongitudinalNormalCenter,
                                           options);
    auto targetEvidenceAll = buildEvidence(targetBuild.groups,
                                             0.0,
                                             targetLongitudinalNormalCenter,
                                             options);
    // filtered 鍥炬槸杩涘叆浜屾鍒嗙粍鍓嶇殑绛涢€夌粨鏋滐細淇濈暀鐪熷疄妫€娴嬬墖娈靛強鍏舵柇瑁傦�?
    // 涓嶈兘浣跨敤闅忓悗閲嶆柊鎷熷悎鐨勭嚎缁勶紝鍚﹀垯浼氫笌 fitted 鍥捐〃杈惧悓涓€鏉″畬鏁寸嚎銆?
    result.filtered_source_lines = renderGroups(referenceCanvas.size,
                                                sourceBuild.groups,
                                                sourceEvidenceAll,
                                                kSourceLineColor);
    result.filtered_target_lines = renderGroups(referenceCanvas.size,
                                                targetBuild.groups,
                                                targetEvidenceAll,
                                                kTargetLineColor);
    retainEvidenceGroups(sourceBuild.groups, sourceEvidenceAll);
    retainEvidenceGroups(targetBuild.groups, targetEvidenceAll);
    // 鍏堝畬鎴愬唴渚ч暱杈瑰拰绔栫洿宸︿晶杈硅繃婊わ紝鍐嶅鍓╀綑鐨勬嫙鍚堢嚎缁勫仛涓€娆″鏉鹃噸鍒嗛厤�?
    sourceBuild.groups = refitSeparatedLineGroups(std::move(sourceBuild.groups),
                                                   0.0,
                                                   false,
                                                   options);
    targetBuild.groups = refitSeparatedLineGroups(std::move(targetBuild.groups),
                                                   0.0,
                                                   false,
                                                   options);
    applyProfileProminence(sourceBuild.groups,
                           0.0,
                           sourceProfiles,
                           options);
    applyProfileProminence(targetBuild.groups,
                           0.0,
                           targetProfiles,
                           options);
    markReferenceMainDirectionGroups(sourceBuild.groups,
                                     warpedSourceMetrics.longSide,
                                     options);
    markReferenceMainDirectionGroups(targetBuild.groups,
                                     canvasTargetMetrics.longSide,
                                     options);
    // 閲嶅缓绱㈠紩锛屼娇鏈€缁堝浘銆佸尮閰嶅拰缁撴灉璁℃暟鍏卞悓寮曠敤 final_groups�?
    sourceEvidenceAll = buildEvidence(sourceBuild.groups,
                                      0.0,
                                      sourceLongitudinalNormalCenter,
                                      options);
    targetEvidenceAll = buildEvidence(targetBuild.groups,
                                       0.0,
                                       targetLongitudinalNormalCenter,
                                       options);
    result.source_line_group_count = static_cast<int>(sourceBuild.groups.size());
    result.target_line_group_count = static_cast<int>(targetBuild.groups.size());
    result.source_valid_line_group_count = result.source_line_group_count;
    result.target_valid_line_group_count = result.target_line_group_count;
    std::vector<EvidenceGroup> sourceHorizontal;
    std::vector<EvidenceGroup> targetHorizontal;
    std::vector<EvidenceGroup> sourceVertical;
    std::vector<EvidenceGroup> targetVertical;
    for (const auto& item : sourceEvidenceAll) {
        if (item.axis == Axis::HORIZONTAL) {
            if (item.mainDirection) {
                sourceHorizontal.push_back(item);
            }
        } else {
            sourceVertical.push_back(item);
        }
    }
    for (const auto& item : targetEvidenceAll) {
        if (item.axis == Axis::HORIZONTAL) {
            if (item.mainDirection) {
                targetHorizontal.push_back(item);
            }
        } else {
            targetVertical.push_back(item);
        }
    }
    const auto normalSort = [](const EvidenceGroup& first, const EvidenceGroup& second) {
        return first.normalPosition < second.normalPosition;
    };
    std::sort(sourceHorizontal.begin(), sourceHorizontal.end(), normalSort);
    std::sort(targetHorizontal.begin(), targetHorizontal.end(), normalSort);
    std::sort(sourceVertical.begin(), sourceVertical.end(), normalSort);
    std::sort(targetVertical.begin(), targetVertical.end(), normalSort);
    // 闃舵�?6锛氭寜娉曞悜浣嶇疆杩涜鍏佽璺宠繃鐨勬湁搴忓尮閰嶏紝骞舵眹鎬绘按骞?绔栫洿涓夋€佺粨鏋溿�?
    result.horizontal = evaluateDirection(sourceHorizontal,
                                          targetHorizontal,
                                          true,
                                          options);
    result.vertical = evaluateDirection(sourceVertical,
                                        targetVertical,
                                        false,
                                        options);
    result.fitted_source_lines = renderFittedGroups(referenceCanvas.size,
                                                    sourceBuild.groups,
                                                    sourceEvidenceAll,
                                                    0.0,
                                                    kSourceLineColor);
    result.fitted_target_lines = renderFittedGroups(referenceCanvas.size,
                                                    targetBuild.groups,
                                                    targetEvidenceAll,
                                                    0.0,
                                                    kTargetLineColor);
    renderOverlay(referenceCanvas.size,
                  sourceBuild.groups,
                  targetBuild.groups,
                  sourceEvidenceAll,
                  targetEvidenceAll,
                  options,
                  0.0,
                  result.matched_line_overlay);

    if (result.horizontal.status == "FAIL" || result.vertical.status == "FAIL") {
        result.status = "FAIL";
        result.message = "[summary] horizontal or vertical line-group structure conflicts";
    } else if (result.horizontal.status == "PASS" && result.vertical.status == "PASS") {
        result.status = "PASS";
        result.message = "[summary] horizontal and vertical line-group evidence agrees";
    } else {
        result.status = "INSUFFICIENT";
        result.message = "[summary] horizontal or vertical line-group evidence is insufficient";
    }
    return true;
}

} // namespace ir::edge_structure_diagnostic
