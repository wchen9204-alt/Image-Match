#include "keypoint/multilayer_dark_keypoint_extractor.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <tuple>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "core/factory.h"
#include "utils/image_utils.h"
#include "utils/logger.h"

namespace ir {

namespace {

/// 多层暗部流程：保留阈值以下像素，再拉伸有效暗部灰度范围。
cv::Mat buildDarkFeatureLayer(const cv::Mat& gray, const int threshold) {
    cv::Mat layer = cv::Mat::zeros(gray.size(), CV_8UC1);
    const cv::Mat dark_mask = gray <= threshold;
    gray.copyTo(layer, dark_mask);

    double max_value = 0.0;
    cv::minMaxLoc(layer, nullptr, &max_value);
    if (max_value <= 0.0) {
        return {};
    }

    cv::Mat normalized;
    cv::normalize(layer, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    normalized.setTo(0, layer == 0);
    return normalized;
}

std::tuple<int, int, int, int> makeKeypointKey(const cv::KeyPoint& keypoint) {
    return {
        static_cast<int>(std::lround(keypoint.pt.x * 10.0f)),
        static_cast<int>(std::lround(keypoint.pt.y * 10.0f)),
        static_cast<int>(std::lround(keypoint.size * 10.0f)),
        static_cast<int>(std::lround(keypoint.angle))
    };
}

void appendUniqueKeypoints(const std::vector<cv::KeyPoint>& layer_keypoints,
                           std::vector<cv::KeyPoint>& merged_keypoints,
                           std::set<std::tuple<int, int, int, int>>& seen) {
    for (const cv::KeyPoint& keypoint : layer_keypoints) {
        if (!seen.insert(makeKeypointKey(keypoint)).second) {
            continue;
        }
        merged_keypoints.push_back(keypoint);
    }
}

bool extractLayerKeypoints(const cv::Mat& gray,
                           const std::vector<int>& thresholds,
                           const YAML::Node& keypoint_config,
                           std::vector<cv::KeyPoint>& keypoints) {
    std::set<std::tuple<int, int, int, int>> seen;
    for (const int threshold : thresholds) {
        const cv::Mat layer = buildDarkFeatureLayer(gray, threshold);
        if (layer.empty()) {
            continue;
        }

        // 每层创建独立算法对象，避免不同层之间共享 OpenCV 内部状态。
        const auto layer_extractor = Factory::createKeypointExtractor(keypoint_config);
        if (!layer_extractor) {
            continue;
        }
        const auto detector = layer_extractor->feature2D();
        if (detector.empty()) {
            continue;
        }
        std::vector<cv::KeyPoint> layer_keypoints;
        cv::Mat discarded_descriptors;
        detector->detectAndCompute(
            layer, cv::noArray(), layer_keypoints, discarded_descriptors);
        appendUniqueKeypoints(layer_keypoints, keypoints, seen);
    }
    return !keypoints.empty();
}

std::vector<int> buildLayerThresholds(const int layer_count) {
    std::vector<int> thresholds;
    thresholds.reserve(static_cast<size_t>(layer_count));
    for (int index = 0; index < layer_count; ++index) {
        thresholds.push_back(255 * (layer_count - index) / layer_count);
    }
    return thresholds;
}

bool computeOriginalDescriptors(const cv::Mat& gray,
                                const YAML::Node& keypoint_config,
                                std::vector<cv::KeyPoint>& keypoints,
                                cv::Mat& descriptors) {
    if (keypoints.empty()) {
        return false;
    }
    const auto original_extractor = Factory::createKeypointExtractor(keypoint_config);
    if (!original_extractor) {
        return false;
    }
    const auto extractor = original_extractor->feature2D();
    if (extractor.empty()) {
        return false;
    }
    // 描述子回到原始灰度图计算，保证不同暗部层的描述子处于同一灰度基准。
    extractor->compute(gray, keypoints, descriptors);
    return !keypoints.empty() && !descriptors.empty() &&
           descriptors.rows == static_cast<int>(keypoints.size());
}

} // namespace

MultilayerDarkKeypointExtractor::MultilayerDarkKeypointExtractor(
    const YAML::Node& keypoint_config,
    const int layer_count)
    : _extractor(Factory::createKeypointExtractor(keypoint_config)),
      _keypoint_config(keypoint_config),
      _name(_extractor->name() + "_DARK_MULTILAYER"),
      _layer_count(std::max(1, layer_count)) {}

bool MultilayerDarkKeypointExtractor::extract(RegistrationContext& ctx) {
    auto& features = ctx.keypoint_data;
    features.clear();
    features.type = _extractor->type();
    features.norm_type = _extractor->normType();

    // 1. 准备原图灰度，暗部图层只用于本提取器，不改写配准流程中的原始图像。
    if (ctx.images.first.empty() || ctx.images.second.empty()) {
        IR_LOG_ERROR("多层暗部提取缺少输入图像。");
        return false;
    }
    if (!image_utils::ensureGray(ctx.images.first, ctx.images.first_gray) ||
        !image_utils::ensureGray(ctx.images.second, ctx.images.second_gray)) {
        IR_LOG_ERROR("多层暗部提取无法准备灰度图。");
        return false;
    }

    // 2. 源图和目标图分别处理；某一张图缺少某个阈值层，不影响另一张图继续提取。
    std::vector<cv::KeyPoint> first_keypoints;
    std::vector<cv::KeyPoint> second_keypoints;
    const std::vector<int> thresholds = buildLayerThresholds(_layer_count);
    extractLayerKeypoints(ctx.images.first_gray, thresholds, _keypoint_config, first_keypoints);
    extractLayerKeypoints(ctx.images.second_gray, thresholds, _keypoint_config, second_keypoints);

    // 3. 合并去重后的关键点，再在每张对应原灰度图上独立计算描述子。
    const bool first_ok = computeOriginalDescriptors(
        ctx.images.first_gray, _keypoint_config, first_keypoints, features.first.descriptors);
    const bool second_ok = computeOriginalDescriptors(
        ctx.images.second_gray, _keypoint_config, second_keypoints, features.second.descriptors);
    features.first.keypoints = std::move(first_keypoints);
    features.second.keypoints = std::move(second_keypoints);

    // 4. 输出的关键点与描述子始终按行对齐，可直接进入既有匹配、过滤和几何流程。
    IR_LOG_INFO("多层暗部 ",
                _extractor->name(),
                " 提取完成：",
                features.first.keypoints.size(),
                " / ",
                features.second.keypoints.size(),
                " 个关键点，共 ",
                _layer_count,
                " 层。");
    return first_ok && second_ok && !features.empty();
}

} // namespace ir
