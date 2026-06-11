#include "direct/dense/dense_flow_common.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "geometry/partial_affine_utils.h"
#include "utils/image_utils.h"
#include "utils/logger.h"
#include "utils/string_utils.h"
#include "utils/yaml_utils.h"

namespace ir {
namespace dense_flow_common {
namespace {

int normalizedRobustMethod(const std::string& method, bool allowRho) {
    const int robustMethod = robustMethodFromString(method);
    if (robustMethod == cv::RANSAC || robustMethod == cv::LMEDS) {
        return robustMethod;
    }
    if (allowRho && robustMethod == cv::RHO) {
        return robustMethod;
    }
    return cv::RANSAC;
}

bool isPointInsideWithMargin(const cv::Point2f& pt, const cv::Size& size, int borderMargin) {
    const float margin = static_cast<float>(std::max(0, borderMargin));
    return pt.x >= margin && pt.y >= margin && pt.x < static_cast<float>(size.width) - margin &&
           pt.y < static_cast<float>(size.height) - margin;
}

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

bool fitGlobalTransform(const std::vector<cv::Point2f>& srcPts,
                        const std::vector<cv::Point2f>& dstPts,
                        const std::string& fitModel,
                        const RobustFitOptions& options,
                        cv::Mat& A,
                        cv::Mat& H,
                        std::vector<unsigned char>& mask,
                        GeometryType& type) {
    A.release();
    H.release();
    mask.clear();
    const std::string model = string_utils::toUpperAscii(fitModel);
    const double ransacThreshold = std::max(0.0, options.threshold);
    const int maxIters = std::max(1, options.max_iters);
    const double confidence = std::clamp(options.confidence, 1e-6, 0.999999);
    const int refineIters = std::max(0, options.refine_iters);

    // HOMOGRAPHY 分支保留给存在透视形变的配置；当前平移+旋转主场景通常不走该分支。
    if (model == "HOMOGRAPHY") {
        if (srcPts.size() < 4) {
            return false;
        }
        H = cv::findHomography(srcPts,
                               dstPts,
                               normalizedRobustMethod(options.method, true),
                               ransacThreshold,
                               mask,
                               maxIters,
                               confidence);
        type = GeometryType::HOMOGRAPHY;
        return !H.empty();
    }

    // RIGID/SIMILARITY 先复用 OpenCV 的 partial affine RANSAC 做鲁棒内点筛选。
    // 注意 partial affine 本身允许等比缩放，因此 RIGID 还会在内点上回归严格旋转+平移矩阵。
    if (model == "RIGID" || model == "SIMILARITY") {
        if (srcPts.size() < 2) {
            return false;
        }
        A = cv::estimateAffinePartial2D(srcPts,
                                        dstPts,
                                        mask,
                                        normalizedRobustMethod(options.method, false),
                                        ransacThreshold,
                                        static_cast<size_t>(maxIters),
                                        confidence,
                                        static_cast<size_t>(refineIters));
        type = model == "SIMILARITY" ? GeometryType::SIMILARITY : GeometryType::RIGID;
        if (model == "RIGID" &&
            // 当前业务只接受平移+旋转，最终矩阵不能直接使用带 scale 自由度的 OpenCV 结果。
            !partial_affine_utils::refineRigidFromMask(srcPts, dstPts, ransacThreshold, mask, A)) {
            return false;
        }
        return !A.empty();
    }

    // AFFINE 明确允许剪切和非等比缩放，只在配置显式要求时启用，避免误把未知模型当 AFFINE。
    if (model == "AFFINE") {
        if (srcPts.size() < 3) {
            return false;
        }
        A = cv::estimateAffine2D(srcPts,
                                 dstPts,
                                 mask,
                                 normalizedRobustMethod(options.method, false),
                                 ransacThreshold,
                                 static_cast<size_t>(maxIters),
                                 confidence,
                                 static_cast<size_t>(refineIters));
        type = GeometryType::AFFINE;
        return !A.empty();
    }

    // 未知模型直接拒绝，避免静默回退造成配置拼写错误时输出不符合预期的几何结果。
    IR_LOG_WARN("Dense flow direct: unsupported fit_model=", fitModel);
    return false;
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
    const double fbThreshold = std::max(0.0, options.fb_threshold);
    const cv::Mat gradientMagnitude =
        options.gradient_threshold > 0.0 ? computeGradientMagnitude(firstGray) : cv::Mat();
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
            if (!isPointInsideWithMargin(src, dd.flow.size(), options.border_margin)) {
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
            const double magnitude = cv::norm(flow);
            if ((minMagnitude >= 0.0 && magnitude < minMagnitude) ||
                (maxMagnitude >= 0.0 && magnitude > maxMagnitude)) {
                // 光流幅值过小或过大都可能是无效样本或异常估计，先在采样阶段剔除。
                ++rejectedByMagnitude;
                continue;
            }

            // 将光流位移加到源点坐标上，得到目标图中的对应位置。
            const cv::Point2f dst(src.x + flow.x, src.y + flow.y);
            if (!isPointInsideWithMargin(dst, dd.flow.size(), options.border_margin)) {
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
    if (!fitGlobalTransform(srcPts, dstPts, options.fit_model, options.robust, A, H, mask, type)) {
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
