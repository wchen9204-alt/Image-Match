#include "geometry/multilayer_dark_rigid_estimator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "data/correspondence_view.h"
#include "geometry/partial_affine_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

using VoteKey = std::tuple<int, int, int>;

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

struct RansacResult {
    cv::Mat matrix;
    std::vector<unsigned char> mask;
    int inliers = 0;
    bool success = false;
};

RansacResult runReferenceRigidRansac(const std::vector<cv::Point2f>& source,
                                     const std::vector<cv::Point2f>& target,
                                     const double threshold,
                                     const int iterations) {
    RansacResult result;
    const size_t count = std::min(source.size(), target.size());
    if (count < 2 || iterations <= 0) {
        return result;
    }

    // 参考实现固定使用 42，保证每次 fallback 的抽样可复现。
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

        // 参考流程只按内点数量保留模型，平局时保留先出现的模型。
        if (inliers > best_count) {
            best_count = inliers;
            best_mask = std::move(mask);
            best_matrix = candidate;
        }
    }

    if (best_count < 2 || best_matrix.empty()) {
        return result;
    }

    // 参考流程在最佳内点集合上重新拟合一次严格刚体矩阵。
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
}

bool MultilayerDarkRigidEstimator::estimate(RegistrationContext& ctx) {
    auto& geometry = ctx.geometry_data;
    geometry.clear();
    geometry.type = GeometryType::RIGID;

    const CorrespondenceView view = ensureCorrespondenceView(ctx);
    if (view.filtered.size() < static_cast<size_t>(_min_inliers)) {
        geometry.message = "多层暗部扩展匹配数量不足";
        return false;
    }

    // 1. 按每个匹配隐含的旋转角与平移量投票分簇。
    std::map<VoteKey, std::vector<cv::DMatch>> bins;
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
        const double angle = target_angle - source_angle;
        const double radians = angle * CV_PI / 180.0;
        const double cosine = std::cos(radians);
        const double sine = std::sin(radians);
        const double tx = target_keypoint.pt.x -
            (cosine * source_keypoint.pt.x - sine * source_keypoint.pt.y);
        const double ty = target_keypoint.pt.y -
            (sine * source_keypoint.pt.x + cosine * source_keypoint.pt.y);
        bins[{static_cast<int>(std::floor(angle / _rotation_bin_degrees)),
              static_cast<int>(std::floor(tx / _translation_bin_size)),
              static_cast<int>(std::floor(ty / _translation_bin_size))}].push_back(match);
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

    // 2. 每个投票簇执行固定种子严格刚体 RANSAC，按图像质量分数选择最终模型。
    cv::Mat best_matrix;
    std::vector<unsigned char> best_mask;
    std::vector<cv::Point2f> best_source;
    std::vector<cv::Point2f> best_target;
    std::vector<cv::DMatch> best_matches;
    int best_inliers = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (const auto& cluster : clusters) {
        if (cluster.size() < static_cast<size_t>(_min_inliers)) {
            continue;
        }
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
        const RansacResult candidate = runReferenceRigidRansac(
            source, target, _reprojection_threshold, _ransac_iterations);
        if (!candidate.success) {
            continue;
        }
        const double score = scoreRegistration(
            ctx.images.first_gray, ctx.images.second_gray, candidate.matrix);
        if (best_matrix.empty() || score > best_score) {
            best_matrix = candidate.matrix;
            best_mask = candidate.mask;
            best_source = std::move(source);
            best_target = std::move(target);
            best_matches = std::move(valid_matches);
            best_inliers = candidate.inliers;
            best_score = score;
        }
    }

    if (best_matrix.empty() || best_inliers < _min_inliers) {
        geometry.message = "多层暗部严格刚体 RANSAC 未得到有效模型";
        return false;
    }

    // 3. 将候选簇内的原始索引和 mask 写回通用上下文，供后续刷新快照、warp、验证和可视化使用。
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
