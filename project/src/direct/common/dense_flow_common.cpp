#include "direct/common/dense_flow_common.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/image_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {
namespace dense_flow_common {
namespace {

bool sampleFlowBilinear(const cv::Mat& flow, const cv::Point2f& pt, cv::Point2f& value) {
    if (flow.empty() || flow.type() != CV_32FC2 || pt.x < 0.0f || pt.y < 0.0f ||
        pt.x >= static_cast<float>(flow.cols - 1) || pt.y >= static_cast<float>(flow.rows - 1)) {
        return false;
    }

    const int x0 = static_cast<int>(std::floor(pt.x));
    const int y0 = static_cast<int>(std::floor(pt.y));
    const float dx = pt.x - static_cast<float>(x0);
    const float dy = pt.y - static_cast<float>(y0);

    const cv::Point2f v00 = flow.at<cv::Point2f>(y0, x0);
    const cv::Point2f v10 = flow.at<cv::Point2f>(y0, x0 + 1);
    const cv::Point2f v01 = flow.at<cv::Point2f>(y0 + 1, x0);
    const cv::Point2f v11 = flow.at<cv::Point2f>(y0 + 1, x0 + 1);
    value = v00 * ((1.0f - dx) * (1.0f - dy)) + v10 * (dx * (1.0f - dy)) +
            v01 * ((1.0f - dx) * dy) + v11 * (dx * dy);
    return std::isfinite(value.x) && std::isfinite(value.y);
}

cv::Mat computeGradientMagnitude(const cv::Mat& gray) {
    cv::Mat gradX;
    cv::Mat gradY;
    cv::Sobel(gray, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gradY, CV_32F, 0, 1, 3);

    cv::Mat magnitude;
    cv::magnitude(gradX, gradY, magnitude);
    return magnitude;
}

std::string labelOrDefault(const std::string& methodLabel) {
    return methodLabel.empty() ? std::string("DENSE_FLOW") : methodLabel;
}

} // namespace

PostprocessOptions readPostprocessOptions(const YAML::Node& params) {
    PostprocessOptions options;
    options.gradient_threshold = yaml_utils::getDouble(params, "gradient_threshold", 0.0);
    options.fit_global_transform = yaml_utils::getBool(params, "fit_global_transform", true);
    options.fit_model = yaml_utils::getString(params, "fit_model", "RIGID");
    options.sample_step = yaml_utils::getInt(params, "sample_step", 8);
    options.min_flow_magnitude = yaml_utils::getDouble(params, "min_flow_magnitude", -1.0);
    options.max_flow_magnitude = yaml_utils::getDouble(params, "max_flow_magnitude", -1.0);
    options.border_margin = yaml_utils::getInt(params, "border_margin", 0);
    options.forward_backward_check = yaml_utils::getBool(params, "forward_backward_check", false);
    options.fb_threshold = yaml_utils::getDouble(params, "fb_threshold", 1.5);
    options.robust.threshold = yaml_utils::getDouble(params, "ransac_threshold", 3.0);
    options.robust.method = yaml_utils::getString(params, "robust_method", "RANSAC");
    options.robust.max_iters = yaml_utils::getInt(params, "ransac_max_iters", 2000);
    options.robust.confidence = yaml_utils::getDouble(params, "ransac_confidence", 0.99);
    options.robust.refine_iters = yaml_utils::getInt(params, "ransac_refine_iters", 10);
    options.min_inliers = yaml_utils::getInt(params, "min_inliers", 20);
    return options;
}

cv::Mat prepareGray(const cv::Mat& gray, int blurKernel) {
    if (gray.empty()) {
        return cv::Mat();
    }

    cv::Mat prepared;
    // 预平滑用于降低光流估计对局部噪声和锯齿边缘的敏感性。
    image_utils::applyOptionalGaussianBlur(gray, prepared, blurKernel);
    return prepared;
}

bool prepareFlowInputs(RegistrationContext& ctx,
                       const std::string& methodLabel,
                       int blurKernel,
                       cv::Mat& firstGray,
                       cv::Mat& secondGray) {
    auto& dd = ctx.direct_data;
    auto& gd = ctx.geometry_data;
    const std::string label = labelOrDefault(methodLabel);

    firstGray.release();
    secondGray.release();
    if (ctx.images.first_gray.empty() || ctx.images.second_gray.empty()) {
        dd.message = label + " requires non-empty grayscale images";
        gd.message = dd.message;
        return false;
    }
    if (ctx.images.first_gray.size() != ctx.images.second_gray.size()) {
        dd.message = label + " requires grayscale images with the same size";
        gd.message = dd.message;
        return false;
    }

    firstGray = prepareGray(ctx.images.first_gray, blurKernel);
    secondGray = prepareGray(ctx.images.second_gray, blurKernel);
    if (firstGray.empty() || secondGray.empty()) {
        dd.message = label + " failed to prepare grayscale images";
        gd.message = dd.message;
        return false;
    }
    return true;
}

bool finalizeFlowAlignment(RegistrationContext& ctx,
                           const cv::Mat& firstGray,
                           const cv::Mat& backwardFlow,
                           const PostprocessOptions& options,
                           const std::string& methodLabel) {
    auto& dd = ctx.direct_data;
    auto& gd = ctx.geometry_data;
    const std::string label = labelOrDefault(methodLabel);

    if (dd.flow.empty()) {
        dd.message = label + " produced empty flow";
        gd.message = dd.message;
        return false;
    }

    if (!options.fit_global_transform) {
        // 仅保留稠密光流时不生成可用于 warp 的全局矩阵，调用方需要检查 geometry_data.valid。
        dd.valid = true;
        dd.score = 1.0;
        gd.type = GeometryType::UNKNOWN;
        gd.message = label + " flow produced without global transform";
        return true;
    }

    std::vector<cv::Point2f> srcPts;
    std::vector<cv::Point2f> dstPts;
    const int step = std::max(1, options.sample_step);
    const double minMagnitude = std::max(-1.0, options.min_flow_magnitude);
    const double maxMagnitude = std::max(-1.0, options.max_flow_magnitude);
    const double minMagnitude2 = minMagnitude >= 0.0 ? minMagnitude * minMagnitude : -1.0;
    const double maxMagnitude2 = maxMagnitude >= 0.0 ? maxMagnitude * maxMagnitude : -1.0;
    const double fbThreshold = std::max(0.0, options.fb_threshold);
    const cv::Mat gradientMagnitude =
        options.gradient_threshold > 0.0 ? computeGradientMagnitude(firstGray) : cv::Mat();
    const int estimatedRows = std::max(0, (dd.flow.rows - step / 2 + step - 1) / step);
    const int estimatedCols = std::max(0, (dd.flow.cols - step / 2 + step - 1) / step);
    const size_t estimatedSamples =
        static_cast<size_t>(std::max(0, estimatedRows) * std::max(0, estimatedCols));
    srcPts.reserve(estimatedSamples);
    dstPts.reserve(estimatedSamples);
    dd.matches.clear();
    dd.matches.reserve(estimatedSamples);
    int candidateCount = 0;
    int rejectedByGradient = 0;
    int rejectedByMagnitude = 0;
    int rejectedByBorder = 0;
    int rejectedByConsistency = 0;

    // 从稠密光流按固定步长采样点对，避免把每个像素都交给 RANSAC 造成无必要的计算开销。
    for (int y = step / 2; y < dd.flow.rows; y += step) {
        for (int x = step / 2; x < dd.flow.cols; x += step) {
            ++candidateCount;
            const cv::Point2f src(static_cast<float>(x), static_cast<float>(y));
            if (!direct_geometry_common::isPointInsideWithMargin(
                    src, dd.flow.size(), options.border_margin)) {
                ++rejectedByBorder;
                continue;
            }
            if (!gradientMagnitude.empty() &&
                gradientMagnitude.at<float>(y, x) < static_cast<float>(options.gradient_threshold)) {
                // 梯度过低的区域通常约束太弱，不适合参与全局几何拟合。
                ++rejectedByGradient;
                continue;
            }

            const cv::Point2f flow = dd.flow.at<cv::Point2f>(y, x);
            if (!std::isfinite(flow.x) || !std::isfinite(flow.y)) {
                continue;
            }
            const double magnitude2 = static_cast<double>(flow.x) * flow.x +
                                      static_cast<double>(flow.y) * flow.y;
            if ((minMagnitude2 >= 0.0 && magnitude2 < minMagnitude2) ||
                (maxMagnitude2 >= 0.0 && magnitude2 > maxMagnitude2)) {
                // 光流幅值过小或过大都可能是无效样本或异常估计，先在采样阶段剔除。
                ++rejectedByMagnitude;
                continue;
            }
            const float magnitude = static_cast<float>(std::sqrt(std::max(0.0, magnitude2)));

            // 将光流位移加到源点坐标上，得到目标图中的对应位置。
            const cv::Point2f dst(src.x + flow.x, src.y + flow.y);
            if (!direct_geometry_common::isPointInsideWithMargin(
                    dst, dd.flow.size(), options.border_margin)) {
                ++rejectedByBorder;
                continue;
            }
            if (options.forward_backward_check) {
                cv::Point2f backFlow;
                if (!sampleFlowBilinear(backwardFlow, dst, backFlow) ||
                    cv::norm(flow + backFlow) > fbThreshold) {
                    // 前后向不一致的样本通常不自洽，会破坏后续 RANSAC 拟合稳定性。
                    ++rejectedByConsistency;
                    continue;
                }
            }

            srcPts.push_back(src);
            dstPts.push_back(dst);
            const int idx = static_cast<int>(dd.matches.size());
            dd.matches.emplace_back(idx, idx, static_cast<float>(magnitude));
        }
    }

    if (srcPts.empty()) {
        dd.message = label + " sampled no valid flow correspondences";
        gd.message = dd.message;
        IR_LOG_WARN(label,
                    " rejected all sampled flow points: candidates=",
                    candidateCount,
                    ", gradient=",
                    rejectedByGradient,
                    ", magnitude=",
                    rejectedByMagnitude,
                    ", border=",
                    rejectedByBorder,
                    ", consistency=",
                    rejectedByConsistency);
        return false;
    }

    cv::Mat A;
    cv::Mat H;
    std::vector<unsigned char> mask;
    GeometryType type = GeometryType::UNKNOWN;
    // 从光流采样点对估计全局变换，RIGID 配置会在 RANSAC 内点上强制回归无缩放矩阵。
    if (!direct_geometry_common::fitGlobalTransform(srcPts,
                                                    dstPts,
                                                    options.fit_model,
                                                    options.robust,
                                                    A,
                                                    H,
                                                    mask,
                                                    type,
                                                    "Dense flow direct")) {
        dd.message = label + " failed to fit global transform from flow";
        gd.message = dd.message;
        return false;
    }

    int inliers = 0;
    for (unsigned char v : mask) {
        if (v) {
            ++inliers;
        }
    }
    if (inliers < options.min_inliers) {
        dd.message = label + " global transform has " + std::to_string(inliers) +
                     " inliers, below min_inliers=" + std::to_string(options.min_inliers);
        gd.message = dd.message;
        IR_LOG_WARN(label, " rejected: ", dd.message);
        return false;
    }

    // 将光流采样点、内点掩码和全局矩阵同时写回，供统一的 warp 和摘要流程复用。
    // 同时写入 direct_data 与 geometry_data：前者保留光流/点对，后者供统一 warp/blend 使用。
    dd.points1 = srcPts;
    dd.points2 = dstPts;
    dd.inlier_mask = mask;
    dd.valid = true;
    dd.score = srcPts.empty() ? 0.0 : static_cast<double>(inliers) / srcPts.size();

    gd.type = type;
    gd.valid = true;
    gd.num_inliers = inliers;
    gd.inlier_ratio = dd.score;
    gd.inlier_mask = mask;
    gd.correspondence_source = "DIRECT";
    gd.num_correspondences = static_cast<int>(srcPts.size());
    if (type == GeometryType::HOMOGRAPHY) {
        H.convertTo(gd.H, CV_64F);
        dd.H = gd.H.clone();
    } else {
        A.convertTo(gd.A, CV_64F);
        dd.A = gd.A.clone();
    }

    IR_LOG_INFO(label,
                " dense flow candidates=",
                candidateCount,
                ", sampled=",
                srcPts.size(),
                ", inliers=",
                inliers,
                ", rejected_gradient=",
                rejectedByGradient,
                ", rejected_magnitude=",
                rejectedByMagnitude,
                ", rejected_border=",
                rejectedByBorder,
                ", rejected_consistency=",
                rejectedByConsistency);
    return true;
}

} // namespace dense_flow_common
} // namespace ir

