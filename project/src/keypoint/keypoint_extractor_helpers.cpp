#include "keypoint/keypoint_extractor_helpers.h"

#include <algorithm>

#include <opencv2/imgproc.hpp>

#include "utils/image_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

BoundaryCornerMethod parseBoundaryCornerMethod(const YAML::Node& boundary) {
    const std::string method =
        yaml_utils::getString(boundary, "corner_method", "");
    if (method == "harris" || method == "HARRIS") {
        return BoundaryCornerMethod::HARRIS;
    }
    if (method == "shi_tomasi" || method == "SHI_TOMASI" || method == "shi-tomasi") {
        return BoundaryCornerMethod::SHI_TOMASI;
    }

    // 兼容旧配置：未切换到 corner_method 时，继续读取 use_harris。
    return yaml_utils::getBool(boundary, "use_harris", false)
               ? BoundaryCornerMethod::HARRIS
               : BoundaryCornerMethod::SHI_TOMASI;
}

cv::Mat buildForegroundMask(const cv::Mat& gray, int foreground_threshold) {
    cv::Mat mask;
    if (gray.empty()) {
        return mask;
    }
    cv::threshold(gray, mask, foreground_threshold, 255, cv::THRESH_BINARY);
    return mask;
}

cv::Mat buildBoundaryBandMask(const cv::Mat& foreground_mask, int boundary_band) {
    cv::Mat boundary_band_mask;
    if (foreground_mask.empty()) {
        return boundary_band_mask;
    }

    const int safe_band = std::max(1, boundary_band);
    const int erode_radius = std::max(1, safe_band / 2);
    const cv::Size erode_kernel_size(erode_radius * 2 + 1, erode_radius * 2 + 1);
    const cv::Size dilate_kernel_size(safe_band * 2 + 1, safe_band * 2 + 1);

    cv::Mat eroded_mask;
    cv::Mat boundary_mask;
    const cv::Mat erode_kernel =
        cv::getStructuringElement(cv::MORPH_RECT, erode_kernel_size);
    const cv::Mat dilate_kernel =
        cv::getStructuringElement(cv::MORPH_RECT, dilate_kernel_size);
    cv::erode(foreground_mask, eroded_mask, erode_kernel);
    cv::subtract(foreground_mask, eroded_mask, boundary_mask);
    cv::dilate(boundary_mask, boundary_band_mask, dilate_kernel);
    return boundary_band_mask;
}

bool isFarEnoughFromExistingKeypoints(const cv::Point2f& point,
                                      const std::vector<cv::KeyPoint>& keypoints,
                                      double min_distance) {
    if (min_distance <= 0.0) {
        return true;
    }

    const double min_distance_sq = min_distance * min_distance;
    for (const auto& keypoint : keypoints) {
        const double dx = point.x - keypoint.pt.x;
        const double dy = point.y - keypoint.pt.y;
        if (dx * dx + dy * dy < min_distance_sq) {
            return false;
        }
    }
    return true;
}

} // namespace

bool prepareKeypointExtractionContext(RegistrationContext& ctx,
                                      KeypointType type,
                                      NormType norm,
                                      const std::string& extractor_name) {
    auto& fd = ctx.keypoint_data;
    auto& images = ctx.images;
    fd.type = type;
    fd.norm_type = norm;

    if (images.first.empty() || images.second.empty()) {
        IR_LOG_ERROR(extractor_name, "::extract - source images are empty.");
        return false;
    }
    if (!image_utils::ensureGray(images.first, images.first_gray) ||
        !image_utils::ensureGray(images.second, images.second_gray)) {
        IR_LOG_ERROR(extractor_name, "::extract - failed to prepare grayscale images.");
        return false;
    }
    return true;
}

BoundaryCornerAugmentationConfig loadBoundaryCornerAugmentationConfig(const YAML::Node& cfg) {
    BoundaryCornerAugmentationConfig config;
    if (!cfg || !cfg.IsMap()) {
        return config;
    }

    const YAML::Node augmentation = cfg["augmentation"];
    const YAML::Node boundary =
        augmentation && augmentation.IsMap() ? augmentation["foreground_boundary_corners"]
                                             : YAML::Node{};
    if (!boundary || !boundary.IsMap()) {
        return config;
    }

    config.enabled = yaml_utils::getBool(boundary, "enabled", false);
    config.corner_method = parseBoundaryCornerMethod(boundary);
    config.max_corners = std::max(0, yaml_utils::getInt(boundary, "max_corners", 40));
    config.quality_level =
        std::max(1e-6, yaml_utils::getDouble(boundary, "quality_level", 0.01));
    config.min_distance = std::max(0.0, yaml_utils::getDouble(boundary, "min_distance", 8.0));
    config.block_size = std::max(2, yaml_utils::getInt(boundary, "block_size", 3));
    config.harris_k = yaml_utils::getDouble(boundary, "harris_k", 0.04);
    config.boundary_band = std::max(1, yaml_utils::getInt(boundary, "boundary_band", 6));
    config.foreground_threshold =
        std::max(0, yaml_utils::getInt(boundary, "foreground_threshold", 10));
    config.keypoint_size =
        std::max(1.0f, yaml_utils::getFloat(boundary, "keypoint_size", 12.0f));
    config.max_total_keypoints = yaml_utils::getInt(boundary, "max_total_keypoints", -1);
    return config;
}

int augmentKeypointsWithBoundaryCorners(const cv::Mat& gray,
                                        std::vector<cv::KeyPoint>& keypoints,
                                        const BoundaryCornerAugmentationConfig& config) {
    if (!config.enabled || gray.empty() || config.max_corners <= 0) {
        return 0;
    }

    cv::Mat foreground_mask = buildForegroundMask(gray, config.foreground_threshold);
    cv::Mat boundary_band_mask = buildBoundaryBandMask(foreground_mask, config.boundary_band);
    if (boundary_band_mask.empty() || cv::countNonZero(boundary_band_mask) == 0) {
        IR_LOG_WARN("Boundary corner augmentation skipped: empty boundary band mask.");
        return 0;
    }

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray,
                            corners,
                            config.max_corners,
                            config.quality_level,
                            config.min_distance,
                            boundary_band_mask,
                            config.block_size,
                            config.corner_method == BoundaryCornerMethod::HARRIS,
                            config.harris_k);

    int added = 0;
    for (const auto& corner : corners) {
        if (!isFarEnoughFromExistingKeypoints(corner, keypoints, config.min_distance)) {
            continue;
        }
        keypoints.emplace_back(corner, config.keypoint_size);
        ++added;
        if (config.max_total_keypoints > 0 &&
            static_cast<int>(keypoints.size()) >= config.max_total_keypoints) {
            break;
        }
    }

    if (added > 0) {
        IR_LOG_INFO("Boundary corner augmentation added ", added, " keypoints.");
    }
    return added;
}

bool extractKeypointsWithBoundaryAugmentation(RegistrationContext& ctx,
                                              KeypointType type,
                                              NormType norm,
                                              const std::string& extractor_name,
                                              cv::Feature2D& extractor,
                                              const BoundaryCornerAugmentationConfig& config) {
    if (!prepareKeypointExtractionContext(ctx, type, norm, extractor_name)) {
        return false;
    }

    auto& fd = ctx.keypoint_data;
    auto& images = ctx.images;

    if (!config.enabled) {
        extractor.detectAndCompute(
            images.first_gray, cv::noArray(), fd.first.keypoints, fd.first.descriptors);
        extractor.detectAndCompute(
            images.second_gray, cv::noArray(), fd.second.keypoints, fd.second.descriptors);
        IR_LOG_INFO(extractor_name,
                    " extracted ",
                    fd.first.keypoints.size(),
                    " / ",
                    fd.second.keypoints.size(),
                    " keypoints");
        return !fd.empty();
    }

    extractor.detect(images.first_gray, fd.first.keypoints);
    extractor.detect(images.second_gray, fd.second.keypoints);

    const int added_first = augmentKeypointsWithBoundaryCorners(images.first_gray,
                                                                fd.first.keypoints,
                                                                config);
    const int added_second = augmentKeypointsWithBoundaryCorners(images.second_gray,
                                                                 fd.second.keypoints,
                                                                 config);

    extractor.compute(images.first_gray, fd.first.keypoints, fd.first.descriptors);
    extractor.compute(images.second_gray, fd.second.keypoints, fd.second.descriptors);

    IR_LOG_INFO(extractor_name,
                " extracted ",
                fd.first.keypoints.size(),
                " / ",
                fd.second.keypoints.size(),
                " keypoints",
                " (boundary corners added: ",
                added_first,
                " / ",
                added_second,
                ")");
    return !fd.empty();
}

} // namespace ir
