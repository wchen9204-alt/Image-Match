#pragma once

#include <cmath>
#include <string>

#include <opencv2/video/tracking.hpp>

#include "core/context.h"
#include "core/types.h"
#include "pipeline/base_pipeline_helpers.h"

namespace ir::ecc_common {

/// 功能：将配置中的运动模型字符串映射到 OpenCV ECC 的 motionType。
/// 作用：统一兼容 TRANSLATION / RIGID(EUCLIDEAN) / AFFINE / HOMOGRAPHY 等配置写法。
inline int motionTypeFromString(const std::string& raw) {
    if (raw == "TRANSLATION" || raw == "translation") {
        return cv::MOTION_TRANSLATION;
    }
    if (raw == "EUCLIDEAN" || raw == "euclidean" || raw == "RIGID" || raw == "rigid") {
        return cv::MOTION_EUCLIDEAN;
    }
    if (raw == "HOMOGRAPHY" || raw == "homography") {
        return cv::MOTION_HOMOGRAPHY;
    }
    return cv::MOTION_AFFINE;
}

/// 功能：将 ECC motionType 映射回平台统一几何类型。
/// 作用：让后续 warp、验证和结果摘要复用统一的 GeometryType 分支。
inline GeometryType geometryTypeFromMotionType(int motionType) {
    if (motionType == cv::MOTION_HOMOGRAPHY) {
        return GeometryType::HOMOGRAPHY;
    }
    if (motionType == cv::MOTION_EUCLIDEAN) {
        return GeometryType::RIGID;
    }
    return GeometryType::AFFINE;
}

/// 功能：把 similarity/affine 初值收紧为 ECC 的欧式刚体矩阵。
/// 作用：去掉统一缩放，只保留旋转和平移，满足 OpenCV MOTION_EUCLIDEAN 的约束。
inline bool normalizeEuclideanAffine(cv::Mat& affine) {
    if (affine.empty() || affine.rows < 2 || affine.cols < 3) {
        return false;
    }

    cv::Mat affine64;
    affine.convertTo(affine64, CV_64F);
    const double a00 = affine64.at<double>(0, 0);
    const double a10 = affine64.at<double>(1, 0);
    const double scale = std::sqrt(a00 * a00 + a10 * a10);
    if (!std::isfinite(scale) || scale <= 1e-9) {
        return false;
    }

    affine64.at<double>(0, 0) /= scale;
    affine64.at<double>(0, 1) /= scale;
    affine64.at<double>(1, 0) /= scale;
    affine64.at<double>(1, 1) /= scale;
    affine64.convertTo(affine, affine.type());
    return true;
}

/// 功能：把允许作为 seed 的点特征粗估结果转换成 ECC 可直接使用的初始 warp。
/// 作用：平台统一存的是 source -> target，而 ECC 内部需要 target(template) -> source(input)，
/// 因此这里负责补齐矩阵形状并做一次求逆。
inline bool initialWarpFromFeatureInitializer(const RegistrationContext& ctx,
                                              int motionType,
                                              cv::Mat& warp) {
    const auto& init = ctx.feature_initializer_data;
    if (!init.seed_available) {
        return false;
    }

    cv::Mat forward;
    if (motionType == cv::MOTION_HOMOGRAPHY) {
        if (!init.H.empty()) {
            init.H.convertTo(forward, CV_64F);
        } else if (!init.A.empty()) {
            forward = cv::Mat::eye(3, 3, CV_64F);
            cv::Mat affine64;
            init.A.convertTo(affine64, CV_64F);
            affine64(cv::Rect(0, 0, 3, 2)).copyTo(forward(cv::Rect(0, 0, 3, 2)));
        }
    } else if (!init.A.empty()) {
        init.A.convertTo(forward, CV_64F);
        forward = forward(cv::Rect(0, 0, 3, 2)).clone();
        if (motionType == cv::MOTION_TRANSLATION) {
            forward.at<double>(0, 0) = 1.0;
            forward.at<double>(0, 1) = 0.0;
            forward.at<double>(1, 0) = 0.0;
            forward.at<double>(1, 1) = 1.0;
        } else if (motionType == cv::MOTION_EUCLIDEAN) {
            normalizeEuclideanAffine(forward);
        }
    }

    if (forward.empty()) {
        return false;
    }

    cv::Mat inverse;
    if (!base_pipeline_helpers::invertTransformMatrix(forward, inverse)) {
        return false;
    }

    inverse.convertTo(warp, CV_32F);
    return !warp.empty();
}

} // namespace ir::ecc_common
