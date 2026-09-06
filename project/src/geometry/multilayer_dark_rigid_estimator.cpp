#include "geometry/multilayer_dark_rigid_estimator.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "data/correspondence_view.h"
#include "geometry/partial_affine_utils.h"
#include "pipeline/base_pipeline_helpers.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

using VoteKey = std::tuple<int, int, int>;

/// 将方向差归一化到 [-180, 180)，避免等价旋转落入相距 360 度的不同 bin。
double normalizeAngleDegrees(double angle) {
    while (angle >= 180.0) {
        angle -= 360.0;
    }
    while (angle < -180.0) {
        angle += 360.0;
    }
    return angle;
}

/// 计算非零前景包围盒的像素几何中心；前景为空时回退到图像几何中心。
bool foregroundBoundingBoxCenter(const cv::Mat& gray, cv::Point2d& center) {
    if (gray.empty()) {
        return false;
    }

    cv::Mat foreground_mask;
    cv::compare(gray, cv::Scalar::all(0), foreground_mask, cv::CMP_GT);
    const cv::Rect bounds = cv::boundingRect(foreground_mask);
    if (bounds.empty()) {
        center = {
            0.5 * static_cast<double>(gray.cols - 1),
            0.5 * static_cast<double>(gray.rows - 1)
        };
        return true;
    }

    center = {
        static_cast<double>(bounds.x) + 0.5 * static_cast<double>(bounds.width - 1),
        static_cast<double>(bounds.y) + 0.5 * static_cast<double>(bounds.height - 1)
    };
    return true;
}

/// 计算非零前景包围盒宽度；图像为空或没有前景时返回 0。
double foregroundBoundingBoxWidth(const cv::Mat& gray) {
    if (gray.empty()) {
        return 0.0;
    }
    cv::Mat foregroundMask;
    cv::compare(gray, cv::Scalar::all(0), foregroundMask, cv::CMP_GT);
    return static_cast<double>(cv::boundingRect(foregroundMask).width);
}

/// 使用 2x3 刚体矩阵将一个二维点投影到目标坐标系。
cv::Point2f applyRigid(const cv::Mat& matrix, const cv::Point2f& point) {
    return {
        static_cast<float>(matrix.at<double>(0, 0) * point.x +
                           matrix.at<double>(0, 1) * point.y +
                           matrix.at<double>(0, 2)),
        static_cast<float>(matrix.at<double>(1, 0) * point.x +
                           matrix.at<double>(1, 1) * point.y +
                           matrix.at<double>(1, 2))
    };
}

/// 保存一次严格刚体 RANSAC 的模型、内点掩码和执行结果。
struct RansacResult {
    cv::Mat matrix;
    std::vector<unsigned char> mask;
    int inliers = 0;
    bool success = false;
};

/// 保存单个投票簇的 RANSAC、评分和调试输出信息。
struct ClusterReport {
    bool selected = false;
    bool judged = false;
    size_t matches = 0;
    size_t ransac_matches = 0;
    bool ransac_attempted = false;
    bool ransac_success = false;
    int best_inliers = 0;
    double score = 0.0;
    cv::Mat matrix;
    std::string status;
};

/// 保存一组关键点索引及其欧氏距离，供点对距离索引使用。
struct PointPairEntry {
    int first = -1;
    int second = -1;
    double distance = 0.0;
};

/// 按关键点索引生成距离满足最小间距要求的有序点对索引。
std::vector<PointPairEntry> buildPointPairIndex(std::span<const cv::KeyPoint> keypoints,
                                                const std::vector<int>& selected,
                                                double minimumDistance) {
    const double minimumDistanceSquared = minimumDistance * minimumDistance;
    std::vector<PointPairEntry> pairs;
    for (size_t i = 0; i + 1 < selected.size(); ++i) {
        const int first = selected[i];
        if (first < 0 || first >= static_cast<int>(keypoints.size())) {
            continue;
        }
        for (size_t j = i + 1; j < selected.size(); ++j) {
            const int second = selected[j];
            if (second < 0 || second >= static_cast<int>(keypoints.size())) {
                continue;
            }
            const double dx = static_cast<double>(keypoints[first].pt.x) - keypoints[second].pt.x;
            const double dy = static_cast<double>(keypoints[first].pt.y) - keypoints[second].pt.y;
            const double distanceSquared = dx * dx + dy * dy;
            if (distanceSquared < minimumDistanceSquared) {
                continue;
            }
            pairs.push_back({first, second, std::sqrt(distanceSquared)});
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const PointPairEntry& lhs, const PointPairEntry& rhs) {
        return lhs.distance < rhs.distance;
    });
    return pairs;
}

/// 计算 source 点对方向到 target 点对方向的有符号旋转角差，单位为度。
double pointPairRotationDegrees(const cv::Point2f& sourceFirst,
                                const cv::Point2f& sourceSecond,
                                const cv::Point2f& targetFirst,
                                const cv::Point2f& targetSecond) {
    const double sourceDx = static_cast<double>(sourceSecond.x) - sourceFirst.x;
    const double sourceDy = static_cast<double>(sourceSecond.y) - sourceFirst.y;
    const double targetDx = static_cast<double>(targetSecond.x) - targetFirst.x;
    const double targetDy = static_cast<double>(targetSecond.y) - targetFirst.y;
    return normalizeAngleDegrees(
        std::atan2(targetDy, targetDx) * 180.0 / CV_PI -
        std::atan2(sourceDy, sourceDx) * 180.0 / CV_PI);
}

/// 向簇中追加匹配，并避免同一 query/train 索引组合重复出现。
void appendUniqueMatch(std::vector<cv::DMatch>& matches, const cv::DMatch& candidate) {
    const auto duplicate = std::find_if(matches.begin(), matches.end(), [&](const cv::DMatch& match) {
        return match.queryIdx == candidate.queryIdx && match.trainIdx == candidate.trainIdx;
    });
    if (duplicate == matches.end()) {
        matches.push_back(candidate);
    }
}

/// 将投票簇的统计信息写入调试 CSV 文件。
void writeClusterReport(const RegistrationContext& ctx,
                        const std::vector<ClusterReport>& reports) {
    if (ctx.output_dir.empty()) {
        return;
    }

    const std::filesystem::path debug_dir = ctx.output_dir / "debug";
    std::error_code error;
    std::filesystem::create_directories(debug_dir, error);
    if (error) {
        IR_LOG_WARN("无法创建多层暗部簇诊断目录：", debug_dir.string(),
                    "，错误：", error.message());
        return;
    }

    std::ofstream output(debug_dir / "multilayer_clusters.csv", std::ios::trunc);
    if (!output) {
        IR_LOG_WARN("无法写入多层暗部簇诊断文件：",
                    (debug_dir / "multilayer_clusters.csv").string());
        return;
    }

    output << "rank,selected,judged,matches,ransac_attempted,ransac_matches,"
              "gray_diff_p90,gray_diff_p90_normalized,rotation_deg,tx,ty,status\n";
    output << std::fixed << std::setprecision(6);
    for (size_t index = 0; index < reports.size(); ++index) {
        const ClusterReport& report = reports[index];
        output << (index + 1) << ','
               << (report.selected ? "yes" : "no") << ','
               << (report.judged ? "yes" : "no") << ','
               << report.matches << ','
               << (report.ransac_attempted ? "yes" : "no") << ','
               << report.ransac_matches << ','
               << (report.ransac_success ? "true" : "false") << ','
               << report.best_inliers << ',';
        if (!report.ransac_success) {
            output << "n/a,n/a,n/a,n/a,";
        } else {
            const double rotation = std::atan2(
                report.matrix.at<double>(1, 0), report.matrix.at<double>(0, 0)) *
                180.0 / CV_PI;
            output << report.score << ',' << rotation << ','
                   << report.matrix.at<double>(0, 2) << ','
                   << report.matrix.at<double>(1, 2) << ',';
        }
        output << report.status << '\n';
    }
}

/// 将成功的簇模型应用到 source 图像，并保存对齐后的 false-color 调试图。
void writeClusterAlignedImage(const RegistrationContext& ctx,
                              const size_t rank,
                              const ClusterReport& report) {
    if (!report.ransac_success || report.matrix.empty() ||
        ctx.images.first_gray.empty() || ctx.images.second_gray.empty() ||
        ctx.output_dir.empty()) {
        return;
    }

    cv::Mat warped_source;
    cv::warpAffine(ctx.images.first_gray,
                   warped_source,
                   report.matrix,
                   ctx.images.second_gray.size(),
                   cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT,
                   cv::Scalar(0));
    cv::Mat aligned;
    if (!base_pipeline_helpers::buildFalseColorOverlay(
            warped_source, ctx.images.second_gray, 0, aligned)) {
        return;
    }

    const std::filesystem::path match_dir = ctx.output_dir / "debug" / "match";
    std::error_code error;
    std::filesystem::create_directories(match_dir, error);
    if (error) {
        return;
    }
    const std::filesystem::path output =
        match_dir / ("multilayer_cluster_" + std::to_string(rank) + "_aligned.png");
    if (!cv::imwrite(output.string(), aligned)) {
        IR_LOG_WARN("无法写入多层暗部簇对齐图：", output.string());
    }
}

/// 使用固定随机种子执行严格刚体 RANSAC，并在最佳内点集合上重新拟合模型。
RansacResult runReferenceRigidRansac(const std::vector<cv::Point2f>& source,
                                     const std::vector<cv::Point2f>& target,
                                     const int min_inliers,
                                     const double threshold,
                                     const int iterations) {
    RansacResult result;
    const size_t count = std::min(source.size(), target.size());
    if (count < static_cast<size_t>(min_inliers) || min_inliers < 2 || iterations <= 0) {
        return result;
    }

    // 固定种子保证每次 fallback 的抽样可复现。
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> distribution(0, count - 1);
    const double threshold_squared = threshold * threshold;
    std::vector<unsigned char> best_mask(count, 0);
    int best_count = 0;
    cv::Mat best_matrix;

    // 每次随机抽取两对点，直接估计严格刚体模型。
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const size_t first = distribution(rng);
        const size_t second = distribution(rng);
        if (first == second) {
            continue;
        }

        const std::vector<cv::Point2f> sampled_source = {source[first], source[second]};
        const std::vector<cv::Point2f> sampled_target = {target[first], target[second]};
        cv::Mat candidate;
        if (!partial_affine_utils::estimateRigidNoScale2D(
                sampled_source, sampled_target, candidate)) {
            continue;
        }

        std::vector<unsigned char> mask(count, 0);
        int inliers = 0;
        for (size_t index = 0; index < count; ++index) {
            const cv::Point2f projected = applyRigid(candidate, source[index]);
            const double dx = projected.x - target[index].x;
            const double dy = projected.y - target[index].y;
            if (dx * dx + dy * dy <= threshold_squared) {
                mask[index] = 1;
                ++inliers;
            }
        }

        // 只按内点数量保留模型，平局时保留先出现的模型。
        if (inliers > best_count) {
            best_count = inliers;
            best_mask = std::move(mask);
            best_matrix = candidate;
        }
    }

    // 最佳模型必须达到多层暗部配置的最少内点数，才能进入重拟合和候选评分。
    result.inliers = best_count;
    if (best_count < min_inliers || best_matrix.empty()) {
        return result;
    }

    // 在最佳内点集合上重新拟合一次严格刚体矩阵。
    std::vector<cv::Point2f> inlier_source;
    std::vector<cv::Point2f> inlier_target;
    for (size_t index = 0; index < count; ++index) {
        if (best_mask[index]) {
            inlier_source.push_back(source[index]);
            inlier_target.push_back(target[index]);
        }
    }
    if (!partial_affine_utils::estimateRigidNoScale2D(
            inlier_source, inlier_target, best_matrix)) {
        return result;
    }

    result.matrix = best_matrix;
    result.mask = std::move(best_mask);
    result.inliers = best_count;
    result.success = true;
    return result;
}

/// 计算灰度残差的 90 分位值，用于图像质量评分。
double percentile90(std::vector<double> values) {
    if (values.empty()) {
        return 255.0;
    }
    std::sort(values.begin(), values.end());
    const size_t index = std::min(
        values.size() - 1,
        static_cast<size_t>(std::ceil(0.90 * static_cast<double>(values.size()))) - 1);
    return values[index];
}

/// 根据前景重叠率和重叠区域灰度差评价刚体模型质量。
double scoreRegistration(const cv::Mat& source_gray,
                         const cv::Mat& target_gray,
                         const cv::Mat& matrix) {
    cv::Mat warped_source;
    cv::warpAffine(source_gray,
                   warped_source,
                   matrix,
                   target_gray.size(),
                   cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT,
                   cv::Scalar(0));

    const cv::Mat source_mask = source_gray > 0;
    const cv::Mat target_mask = target_gray > 0;
    cv::Mat warped_source_mask;
    cv::warpAffine(source_mask,
                   warped_source_mask,
                   matrix,
                   target_gray.size(),
                   cv::INTER_NEAREST,
                   cv::BORDER_CONSTANT,
                   cv::Scalar(0));
    cv::Mat overlap_mask;
    cv::bitwise_and(warped_source_mask, target_mask, overlap_mask);

    const double source_area = static_cast<double>(cv::countNonZero(source_mask));
    const double target_area = static_cast<double>(cv::countNonZero(target_mask));
    const double overlap_area = static_cast<double>(cv::countNonZero(overlap_mask));
    if (source_area <= 0.0 || target_area <= 0.0) {
        return 0.0;
    }

    std::vector<double> residuals;
    residuals.reserve(static_cast<size_t>(overlap_area));
    for (int y = 0; y < target_gray.rows; ++y) {
        for (int x = 0; x < target_gray.cols; ++x) {
            if (overlap_mask.at<unsigned char>(y, x) == 0) {
                continue;
            }
            residuals.push_back(std::abs(
                static_cast<double>(warped_source.at<unsigned char>(y, x)) -
                static_cast<double>(target_gray.at<unsigned char>(y, x))));
        }
    }
    return 0.4 * overlap_area / source_area +
           0.4 * overlap_area / target_area +
           0.2 * (1.0 - percentile90(std::move(residuals)) / 255.0);
}

} // namespace

/// 从 YAML 配置读取多层暗部刚体估计器参数并完成范围归一化。
MultilayerDarkRigidEstimator::MultilayerDarkRigidEstimator(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"];
    const YAML::Node options = params["multilayer_dark"];
    _min_inliers = std::max(2, yaml_utils::getInt(options, "min_inliers", 6));
    _rotation_bin_degrees = std::max(
        1e-6, yaml_utils::getDouble(options, "rotation_bin_degrees", 5.0));
    _translation_bin_size = std::max(
        1e-6, yaml_utils::getDouble(options, "translation_bin_size", 10.0));
    _reprojection_threshold = std::max(
        0.0, yaml_utils::getDouble(options, "reprojection_threshold", 5.0));
    _ransac_iterations = std::max(
        1, yaml_utils::getInt(options, "ransac_iterations", 1000));
    _max_clusters = std::max(1, yaml_utils::getInt(options, "max_clusters", 5));
    _voting_method = yaml_utils::getString(options, "voting_method", "KEYPOINT_ANGLE");
    std::transform(_voting_method.begin(), _voting_method.end(), _voting_method.begin(),
                   [](unsigned char value) { return static_cast<char>(std::toupper(value)); });
    if (_voting_method != "POINT_PAIR") {
        _voting_method = "KEYPOINT_ANGLE";
    }
    _point_pair_distance_tolerance = std::max(
        0.0, yaml_utils::getDouble(options, "point_pair_distance_tolerance", 3.0));
    _point_pair_max_candidates = std::max(
        1, yaml_utils::getInt(options, "point_pair_max_candidates", 64));
    _point_pair_min_distance_ratio = std::max(
        0.0, yaml_utils::getDouble(options, "point_pair_min_distance_ratio", 0.5));
    _point_pair_translation_tolerance = std::max(
        0.0, yaml_utils::getDouble(options, "point_pair_translation_tolerance", 10.0));
}

/// 执行多层暗部刚体估计：投票分簇、严格刚体 RANSAC、模型评分和结果回写。
bool MultilayerDarkRigidEstimator::estimate(RegistrationContext& ctx) {
    auto& geometry = ctx.geometry_data;
    geometry.clear();
    geometry.type = GeometryType::RIGID;

    const CorrespondenceView view = ensureCorrespondenceView(ctx);
    if (view.filtered.size() < static_cast<size_t>(_min_inliers)) {
        geometry.message = "多层暗部扩展匹配数量不足";
        return false;
    }

    // 1. 以 source 前景包围盒中心作为共同坐标原点，并按旋转角与平移量投票分簇。
    cv::Point2d source_rotation_center;
    if (!foregroundBoundingBoxCenter(ctx.images.first_gray, source_rotation_center)) {
        geometry.message = "多层暗部图像为空，无法确定前景旋转中心";
        return false;
    }

    std::map<VoteKey, std::vector<cv::DMatch>> bins;
    if (_voting_method == "POINT_PAIR") {
        std::map<std::pair<int, int>, cv::DMatch> matchLookup;
        for (const cv::DMatch& match : view.filtered) {
            if (match.queryIdx < 0 || match.trainIdx < 0 ||
                match.queryIdx >= static_cast<int>(view.first_keypoints.size()) ||
                match.trainIdx >= static_cast<int>(view.second_keypoints.size())) {
                continue;
            }
            matchLookup.emplace(std::make_pair(match.queryIdx, match.trainIdx), match);
        }
        std::vector<int> sourceSelected;
        std::vector<int> targetSelected;
        sourceSelected.reserve(matchLookup.size());
        targetSelected.reserve(matchLookup.size());
        for (const auto& [indices, match] : matchLookup) {
            sourceSelected.push_back(indices.first);
            targetSelected.push_back(indices.second);
        }
        std::sort(sourceSelected.begin(), sourceSelected.end());
        sourceSelected.erase(std::unique(sourceSelected.begin(), sourceSelected.end()), sourceSelected.end());
        std::sort(targetSelected.begin(), targetSelected.end());
        targetSelected.erase(std::unique(targetSelected.begin(), targetSelected.end()), targetSelected.end());
        const double minimumPairDistance = std::max(
            1.0, std::min(foregroundBoundingBoxWidth(ctx.images.first_gray),
                          foregroundBoundingBoxWidth(ctx.images.second_gray)) *
                     _point_pair_min_distance_ratio);
        const std::vector<PointPairEntry> sourcePairs = buildPointPairIndex(
            view.first_keypoints, sourceSelected, minimumPairDistance);
        const std::vector<PointPairEntry> targetPairs = buildPointPairIndex(
            view.second_keypoints, targetSelected, minimumPairDistance);

        // 将一组满足长度约束的 source/target 点对转换为角度、平移投票。
        const auto addPairVote = [&](const PointPairEntry& sourcePair,
                                     const PointPairEntry& targetPair,
                                     bool reverseTargetOrder) {
            const int targetFirst = reverseTargetOrder ? targetPair.second : targetPair.first;
            const int targetSecond = reverseTargetOrder ? targetPair.first : targetPair.second;
            const auto firstMatch = matchLookup.find({sourcePair.first, targetFirst});
            const auto secondMatch = matchLookup.find({sourcePair.second, targetSecond});
            if (firstMatch == matchLookup.end() || secondMatch == matchLookup.end() ||
                firstMatch->second.queryIdx == secondMatch->second.queryIdx ||
                firstMatch->second.trainIdx == secondMatch->second.trainIdx) {
                return;
            }
            const cv::Point2f& sourceFirst = view.first_keypoints[sourcePair.first].pt;
            const cv::Point2f& sourceSecond = view.first_keypoints[sourcePair.second].pt;
            const cv::Point2f& targetFirstPoint = view.second_keypoints[targetFirst].pt;
            const cv::Point2f& targetSecondPoint = view.second_keypoints[targetSecond].pt;
            const double angle = pointPairRotationDegrees(
                sourceFirst, sourceSecond, targetFirstPoint, targetSecondPoint);
            const double radians = angle * CV_PI / 180.0;
            const double cosine = std::cos(radians);
            const double sine = std::sin(radians);
            const double sourceRelativeX = sourceFirst.x - source_rotation_center.x;
            const double sourceRelativeY = sourceFirst.y - source_rotation_center.y;
            const double targetRelativeX = targetFirstPoint.x - source_rotation_center.x;
            const double targetRelativeY = targetFirstPoint.y - source_rotation_center.y;
            const double tx = targetRelativeX -
                (cosine * sourceRelativeX - sine * sourceRelativeY);
            const double ty = targetRelativeY -
                (sine * sourceRelativeX + cosine * sourceRelativeY);
            const double sourceSecondRelativeX = sourceSecond.x - source_rotation_center.x;
            const double sourceSecondRelativeY = sourceSecond.y - source_rotation_center.y;
            const double targetSecondRelativeX = targetSecondPoint.x - source_rotation_center.x;
            const double targetSecondRelativeY = targetSecondPoint.y - source_rotation_center.y;
            const double txSecond = targetSecondRelativeX -
                (cosine * sourceSecondRelativeX - sine * sourceSecondRelativeY);
            const double tySecond = targetSecondRelativeY -
                (sine * sourceSecondRelativeX + cosine * sourceSecondRelativeY);
            if (std::abs(tx - txSecond) > _point_pair_translation_tolerance ||
                std::abs(ty - tySecond) > _point_pair_translation_tolerance) {
                return;
            }
            const int rotationBinCount = std::max(
                1, static_cast<int>(std::lround(360.0 / _rotation_bin_degrees)));
            int angleBin = static_cast<int>(std::floor(
                (angle + 0.5 * _rotation_bin_degrees) / _rotation_bin_degrees));
            const int halfRotationBinCount = rotationBinCount / 2;
            while (angleBin >= halfRotationBinCount) {
                angleBin -= rotationBinCount;
            }
            while (angleBin < -halfRotationBinCount) {
                angleBin += rotationBinCount;
            }
            const VoteKey key{angleBin,
                              static_cast<int>(std::floor(tx / _translation_bin_size)),
                              static_cast<int>(std::floor(ty / _translation_bin_size))};
            appendUniqueMatch(bins[key], firstMatch->second);
            appendUniqueMatch(bins[key], secondMatch->second);
        };

        const bool sourceIsOuter = sourcePairs.size() <= targetPairs.size();
        const auto& outerPairs = sourceIsOuter ? sourcePairs : targetPairs;
        const auto& innerPairs = sourceIsOuter ? targetPairs : sourcePairs;
        for (const PointPairEntry& outerPair : outerPairs) {
            const auto lower = std::lower_bound(
                innerPairs.begin(), innerPairs.end(),
                outerPair.distance - _point_pair_distance_tolerance,
                [](const PointPairEntry& pair, double value) { return pair.distance < value; });
            const auto upper = std::upper_bound(
                lower, innerPairs.end(),
                outerPair.distance + _point_pair_distance_tolerance,
                [](double value, const PointPairEntry& pair) { return value < pair.distance; });
            std::vector<const PointPairEntry*> candidates;
            candidates.reserve(static_cast<size_t>(std::distance(lower, upper)));
            for (auto candidate = lower; candidate != upper; ++candidate) {
                candidates.push_back(&*candidate);
            }
            std::sort(candidates.begin(), candidates.end(), [&](const PointPairEntry* lhs,
                                                                const PointPairEntry* rhs) {
                const double leftDifference = std::abs(lhs->distance - outerPair.distance);
                const double rightDifference = std::abs(rhs->distance - outerPair.distance);
                if (leftDifference != rightDifference) {
                    return leftDifference < rightDifference;
                }
                return lhs->distance < rhs->distance;
            });
            const size_t candidateLimit = std::min(
                candidates.size(), static_cast<size_t>(_point_pair_max_candidates));
            for (size_t candidateIndex = 0; candidateIndex < candidateLimit; ++candidateIndex) {
                const PointPairEntry& candidate = *candidates[candidateIndex];
                if (sourceIsOuter) {
                    addPairVote(outerPair, candidate, false);
                    addPairVote(outerPair, candidate, true);
                } else {
                    addPairVote(candidate, outerPair, false);
                    addPairVote(candidate, outerPair, true);
                }
            }
        }
        IR_LOG_INFO("MultilayerDarkRigidEstimator point-pair voting: source_keypoints=",
                    sourceSelected.size(), ", target_keypoints=", targetSelected.size(),
                    ", source_pairs=", sourcePairs.size(), ", target_pairs=", targetPairs.size(),
                    ", bins=", bins.size());
    } else {
        // 默认方式：每个匹配根据关键点方向差和相对平移直接投票。
        for (const auto& match : view.filtered) {
            if (match.queryIdx < 0 || match.trainIdx < 0 ||
                match.queryIdx >= static_cast<int>(view.first_keypoints.size()) ||
                match.trainIdx >= static_cast<int>(view.second_keypoints.size())) {
                continue;
            }
            const auto& source_keypoint = view.first_keypoints[static_cast<size_t>(match.queryIdx)];
            const auto& target_keypoint = view.second_keypoints[static_cast<size_t>(match.trainIdx)];
            const double source_angle = source_keypoint.angle < 0.0f ? 0.0 : source_keypoint.angle;
            const double target_angle = target_keypoint.angle < 0.0f ? 0.0 : target_keypoint.angle;
            // 方向差先归一化，再参与旋转计算和投票分 bin。
            const double angle = normalizeAngleDegrees(target_angle - source_angle);
            const double radians = angle * CV_PI / 180.0;
            const double cosine = std::cos(radians);
            const double sine = std::sin(radians);
            const double source_relative_x = source_keypoint.pt.x - source_rotation_center.x;
            const double source_relative_y = source_keypoint.pt.y - source_rotation_center.y;
            const double target_relative_x = target_keypoint.pt.x - source_rotation_center.x;
            const double target_relative_y = target_keypoint.pt.y - source_rotation_center.y;
            const double tx = target_relative_x -
                (cosine * source_relative_x - sine * source_relative_y);
            const double ty = target_relative_y -
                (sine * source_relative_x + cosine * source_relative_y);
            bins[{static_cast<int>(std::floor(angle / _rotation_bin_degrees)),
                  static_cast<int>(std::floor(tx / _translation_bin_size)),
                  static_cast<int>(std::floor(ty / _translation_bin_size))}].push_back(match);
        }
    }

    std::vector<std::vector<cv::DMatch>> clusters;
    clusters.reserve(bins.size());
    for (auto& entry : bins) {
        clusters.push_back(std::move(entry.second));
    }
    std::sort(clusters.begin(), clusters.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.size() > rhs.size();
    });
    if (clusters.size() > static_cast<size_t>(_max_clusters)) {
        clusters.resize(static_cast<size_t>(_max_clusters));
    }

    // 2. 按投票数量排序候选簇，并对每个簇执行固定种子严格刚体 RANSAC。
    cv::Mat best_matrix;
    std::vector<unsigned char> best_mask;
    std::vector<cv::Point2f> best_source;
    std::vector<cv::Point2f> best_target;
    std::vector<cv::DMatch> best_matches;
    int best_inliers = 0;
    size_t best_report_index = std::numeric_limits<size_t>::max();
    std::vector<ClusterReport> reports;
    reports.reserve(clusters.size());
    struct ScoredCandidate {
        size_t report_index = 0;
        double score = 0.0;
        int inlier_count = 0;
        double inlier_ratio = 0.0;
        cv::Mat matrix;
        std::vector<unsigned char> mask;
        std::vector<cv::Point2f> source;
        std::vector<cv::Point2f> target;
        std::vector<cv::DMatch> matches;
    };
    std::vector<ScoredCandidate> scored_candidates;
    scored_candidates.reserve(clusters.size());
    for (const auto& cluster : clusters) {
        ClusterReport report;
        report.matches = cluster.size();
        if (cluster.size() < static_cast<size_t>(_min_inliers)) {
            report.status = "skipped_too_few_matches";
            reports.push_back(std::move(report));
            continue;
        }
        report.judged = true;
        report.ransac_attempted = true;
        std::vector<cv::Point2f> source;
        std::vector<cv::Point2f> target;
        std::vector<cv::DMatch> valid_matches;
        valid_matches.reserve(cluster.size());
        for (const auto& match : cluster) {
            if (match.queryIdx < 0 || match.trainIdx < 0 ||
                match.queryIdx >= static_cast<int>(view.first_keypoints.size()) ||
                match.trainIdx >= static_cast<int>(view.second_keypoints.size())) {
                continue;
            }
            source.push_back(view.first_keypoints[static_cast<size_t>(match.queryIdx)].pt);
            target.push_back(view.second_keypoints[static_cast<size_t>(match.trainIdx)].pt);
            valid_matches.push_back(match);
        }
        report.ransac_matches = valid_matches.size();
        const RansacResult candidate = runReferenceRigidRansac(
            source, target, _min_inliers, _reprojection_threshold, _ransac_iterations);
        if (!candidate.success) {
            report.status = "ransac_failed";
            reports.push_back(std::move(report));
            continue;
        }
        report.ransac_success = true;
        report.best_inliers = candidate.inliers;
        report.matrix = candidate.matrix;
        report.score = scoreRegistration(
            ctx.images.first_gray, ctx.images.second_gray, candidate.matrix);
        report.status = "completed";
        const double inlier_ratio = valid_matches.empty()
            ? 0.0
            : static_cast<double>(candidate.inliers) /
                  static_cast<double>(valid_matches.size());
        const size_t report_index = reports.size();
        reports.push_back(std::move(report));
        scored_candidates.push_back(ScoredCandidate{
            report_index,
            reports.back().score,
            candidate.inliers,
            inlier_ratio,
            candidate.matrix.clone(),
            candidate.mask,
            std::move(source),
            std::move(target),
            std::move(valid_matches)});
    }

    // 3. 先按图像评分筛选候选；评分接近时依次以内点数、内点率和评分决胜。
    if (!scored_candidates.empty()) {
        const auto best_score_it = std::max_element(
            scored_candidates.begin(), scored_candidates.end(),
            [](const ScoredCandidate& lhs, const ScoredCandidate& rhs) {
                return lhs.score < rhs.score;
            });
        const size_t best_score_index = static_cast<size_t>(
            std::distance(scored_candidates.begin(), best_score_it));
        const ScoredCandidate& best_score_candidate = *best_score_it;
        size_t selected_index = best_score_index;

        std::vector<size_t> eligible_indices;
        for (size_t index = 0; index < scored_candidates.size(); ++index) {
            const auto& candidate = scored_candidates[index];
            if (candidate.score + 0.01 < best_score_candidate.score ||
                candidate.inlier_count <= best_score_candidate.inlier_count ||
                candidate.inlier_ratio <= best_score_candidate.inlier_ratio) {
                continue;
            }
            eligible_indices.push_back(index);
        }
        if (!eligible_indices.empty()) {
            selected_index = *std::max_element(
                eligible_indices.begin(), eligible_indices.end(),
                [&](const size_t lhs, const size_t rhs) {
                    const auto& left = scored_candidates[lhs];
                    const auto& right = scored_candidates[rhs];
                    if (left.inlier_count != right.inlier_count) {
                        return left.inlier_count < right.inlier_count;
                    }
                    if (left.inlier_ratio != right.inlier_ratio) {
                        return left.inlier_ratio < right.inlier_ratio;
                    }
                    return left.score < right.score;
                });
        }

        auto& selected = scored_candidates[selected_index];
        best_matrix = selected.matrix.clone();
        best_mask = selected.mask;
        best_source = std::move(selected.source);
        best_target = std::move(selected.target);
        best_matches = std::move(selected.matches);
        best_inliers = selected.inlier_count;
        best_report_index = selected.report_index;
    }

    // 4. 保存排序后投票簇的诊断信息，便于定位匹配簇和 RANSAC 问题。
    if (!reports.empty()) {
        if (best_report_index < reports.size()) {
            reports[best_report_index].selected = true;
        }
        writeClusterReport(ctx, reports);
        for (size_t index = 0; index < reports.size(); ++index) {
            writeClusterAlignedImage(ctx, index + 1, reports[index]);
        }
    }

    if (best_matrix.empty() || best_inliers < _min_inliers) {
        geometry.message = "多层暗部严格刚体 RANSAC 未得到有效模型";
        return false;
    }

    // 5. 将候选簇内的原始索引和 mask 写回通用上下文，供后续刷新快照、warp、验证和可视化使用。
    ctx.keypoint_match_data.filtered_matches = best_matches;
    ctx.keypoint_match_data.inlier_mask = best_mask;
    ctx.keypoint_match_data.inlier_matches.clear();
    for (size_t index = 0; index < best_matches.size() && index < best_mask.size(); ++index) {
        if (best_mask[index]) {
            ctx.keypoint_match_data.inlier_matches.push_back(best_matches[index]);
        }
    }
    ctx.geometry_data.A = best_matrix;
    ctx.geometry_data.valid = true;
    ctx.geometry_data.inlier_mask = best_mask;
    ctx.geometry_data.num_inliers = best_inliers;
    ctx.geometry_data.inlier_ratio = best_source.empty()
        ? 0.0
        : static_cast<double>(best_inliers) / static_cast<double>(best_source.size());
    ctx.geometry_data.num_correspondences = static_cast<int>(best_source.size());
    ctx.geometry_data.correspondence_source = "KEYPOINT_DARK_MULTILAYER";
    ctx.geometry_data.message = "多层暗部刚体投票估计完成";
    return true;
}

} // namespace ir
