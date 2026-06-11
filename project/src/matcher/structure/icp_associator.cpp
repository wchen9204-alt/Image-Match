#include "matcher/structure/icp_associator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "matcher/structure/structure_point_set.h"
#include "utils/logger.h"
#include "utils/string_utils.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

struct NearestPoint {
    cv::Point2f point;
    double distance = std::numeric_limits<double>::infinity();
    bool found = false;
};

// 在线性点集中寻找 query 的最近邻；当前点数受 maxPoints 限制，简单扫描足够稳定。
NearestPoint findNearest(const cv::Point2f& query, const std::vector<cv::Point2f>& points) {
    NearestPoint best;
    double bestDist2 = std::numeric_limits<double>::infinity();
    for (const auto& p : points) {
        const double dx = static_cast<double>(p.x) - query.x;
        const double dy = static_cast<double>(p.y) - query.y;
        const double d2 = dx * dx + dy * dy;
        if (d2 < bestDist2) {
            bestDist2 = d2;
            best.point = p;
            best.found = true;
        }
    }
    best.distance = best.found ? std::sqrt(bestDist2) : std::numeric_limits<double>::infinity();
    return best;
}

} // namespace

IcpAssociator::IcpAssociator(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    _maxPoints = yaml_utils::getInt(params, "maxPoints", 2000);
    _maxIterations = yaml_utils::getInt(params, "maxIterations", 30);
    _minCorrespondences = yaml_utils::getInt(params, "minCorrespondences", 20);
    _maxCorrespondenceDistance =
        yaml_utils::getDouble(params, "maxCorrespondenceDistance", 10.0);
    _tolerance = yaml_utils::getDouble(params, "tolerance", 0.01);
    _scoreThreshold = yaml_utils::getDouble(params, "scoreThreshold", 2.0);
    _initialization =
        string_utils::toUpperAscii(yaml_utils::getString(params, "initialization", "CENTROID"));
}

bool IcpAssociator::associate(RegistrationContext& ctx) {
    auto& md = ctx.structure_match_data;
    md.clear();
    md.method = name();

    if (ctx.structure_data.first.response.empty() || ctx.structure_data.second.response.empty()) {
        md.message = "structure response images are empty";
        return false;
    }
    if (ctx.structure_data.first.response.size() != ctx.structure_data.second.response.size()) {
        md.message = "structure response images have different sizes";
        return false;
    }

    const std::vector<cv::Point2f> srcPoints =
        structure_points::collectPoints(ctx.structure_data.first.response, _maxPoints);
    const std::vector<cv::Point2f> dstPoints =
        structure_points::collectPoints(ctx.structure_data.second.response, _maxPoints);
    if (srcPoints.empty() || dstPoints.empty()) {
        md.message = "no structure points found";
        return false;
    }

    cv::Point2d translation(0.0, 0.0);
    if (_initialization == "CENTROID") {
        // 先用两组点的质心差做初始平移，降低 ICP 陷入局部错误对应的概率。
        translation = structure_points::centroid(dstPoints) - structure_points::centroid(srcPoints);
    }

    double score = std::numeric_limits<double>::infinity();
    int correspondences = 0;
    const int maxIterations = std::max(1, _maxIterations);
    for (int iter = 0; iter < maxIterations; ++iter) {
        // 每轮根据当前平移建立最近邻对应，再用平均残差更新整体平移。
        cv::Point2d deltaSum(0.0, 0.0);
        double distanceSum = 0.0;
        correspondences = 0;

        for (const auto& src : srcPoints) {
            const cv::Point2f moved(static_cast<float>(src.x + translation.x),
                                    static_cast<float>(src.y + translation.y));
            const NearestPoint nearest = findNearest(moved, dstPoints);
            if (!nearest.found || nearest.distance > _maxCorrespondenceDistance) {
                continue;
            }

            deltaSum.x += static_cast<double>(nearest.point.x) - moved.x;
            deltaSum.y += static_cast<double>(nearest.point.y) - moved.y;
            distanceSum += nearest.distance;
            ++correspondences;
        }

        if (correspondences < _minCorrespondences) {
            md.message = "not enough ICP correspondences: " + std::to_string(correspondences);
            return false;
        }

        const cv::Point2d delta(deltaSum.x / static_cast<double>(correspondences),
                                deltaSum.y / static_cast<double>(correspondences));
        translation += delta;
        score = distanceSum / static_cast<double>(correspondences);

        const double deltaNorm = std::sqrt(delta.x * delta.x + delta.y * delta.y);
        if (deltaNorm <= _tolerance) {
            break;
        }
    }

    md.translation = translation;
    md.score = score;
    md.valid = score <= _scoreThreshold;
    if (!md.valid) {
        md.message = "icp score above threshold: " + std::to_string(score);
        IR_LOG_WARN("IcpAssociator rejected match: ", md.message);
    }

    IR_LOG_INFO("IcpAssociator estimated translation dx=",
                translation.x,
                ", dy=",
                translation.y,
                ", score=",
                score,
                ", correspondences=",
                correspondences);
    return md.valid;
}

} // namespace ir
