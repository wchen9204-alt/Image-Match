#include "geometry/multilayer_dark_rigid_estimator.h"

#include <algorithm>
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

/// 计算非零前景包围盒的几何中心，作为多层暗部投票时的旋转参考点。
bool foregroundBoundingBoxCenter(const cv::Mat& gray, cv::Point2d& center) {
    std::vector<cv::Point> foreground_points;
    cv::findNonZero(gray, foreground_points);
    if (foreground_points.empty()) {
        return false;
    }

    const cv::Rect bounds = cv::boundingRect(foreground_points);
    center = {
        static_cast<double>(bounds.x) + static_cast<double>(bounds.width) * 0.5,
        static_cast<double>(bounds.y) + static_cast<double>(bounds.height) * 0.5
    };
    return true;
}

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

    // 1. 以 source 前景包围盒中心为旋转参考点，按每个匹配隐含的旋转角与平移量投票分簇。
    cv::Point2d rotation_center;
    if (!foregroundBoundingBoxCenter(ctx.images.first_gray, rotation_center)) {
        geometry.message = "多层暗部 source 前景为空，无法确定旋转中心";
        return false;
    }

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
        const double relative_x = source_keypoint.pt.x - rotation_center.x;
        const double relative_y = source_keypoint.pt.y - rotation_center.y;
        const double rotated_x = rotation_center.x + cosine * relative_x - sine * relative_y;
        const double rotated_y = rotation_center.y + sine * relative_x + cosine * relative_y;
        const double tx = target_keypoint.pt.x - rotated_x;
        const double ty = target_keypoint.pt.y - rotated_y;
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
    size_t best_report_index = std::numeric_limits<size_t>::max();
    std::vector<ClusterReport> reports;
    reports.reserve(clusters.size());
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
            source, target, _reprojection_threshold, _ransac_iterations);
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
        const double score = report.score;
        if (best_matrix.empty() || score > best_score) {
            best_matrix = candidate.matrix;
            best_mask = candidate.mask;
            best_source = std::move(source);
            best_target = std::move(target);
            best_matches = std::move(valid_matches);
            best_inliers = candidate.inliers;
            best_score = score;
            best_report_index = reports.size();
        }
        reports.push_back(std::move(report));
    }

    // 3. 保存排序后前十个投票簇的诊断信息，便于定位匹配簇和 RANSAC 的问题。
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

    // 4. 将候选簇内的原始索引和 mask 写回通用上下文，供后续刷新快照、warp、验证和可视化使用。
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
