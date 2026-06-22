#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "core/types.h"
#include "geometry/partial_affine_utils.h"
#include "utils/logger.h"
#include "utils/string_utils.h"

namespace ir::direct_geometry_common {

/// 直接法从点对拟合全局几何时共用的鲁棒估计参数。
struct RobustFitOptions {
    /// 鲁棒估计方法，可选 RANSAC / LMEDS；HOMOGRAPHY 额外支持 RHO。
    std::string method = "RANSAC";

    /// RANSAC 重投影内点阈值，单位为像素。
    double threshold = 3.0;

    /// RANSAC 最大迭代次数。
    int max_iters = 2000;

    /// RANSAC 置信度。
    double confidence = 0.99;

    /// OpenCV affine 估计后的 refine 迭代次数。
    int refine_iters = 10;
};

/// 功能：把配置字符串规范化为当前几何拟合允许使用的 OpenCV 鲁棒估计方法。
/// 作用：统一兼容 RANSAC / LMEDS / RHO 等写法，并在不支持 RHO 的分支回退到 RANSAC。
inline int normalizedRobustMethod(const std::string& method, bool allowRho) {
    const int robustMethod = robustMethodFromString(method);
    if (robustMethod == cv::RANSAC || robustMethod == cv::LMEDS) {
        return robustMethod;
    }
    if (allowRho && robustMethod == cv::RHO) {
        return robustMethod;
    }
    return cv::RANSAC;
}

/// 功能：判断点是否落在图像内部，同时预留边界安全距离。
/// 作用：避免采样点或跟踪点过于靠近边缘，减少越界采样和边界插值带来的不稳定性。
inline bool isPointInsideWithMargin(const cv::Point2f& pt,
                                    const cv::Size& size,
                                    int borderMargin) {
    const float margin = static_cast<float>(std::max(0, borderMargin));
    return pt.x >= margin && pt.y >= margin && pt.x < static_cast<float>(size.width) - margin &&
           pt.y < static_cast<float>(size.height) - margin;
}

/// 功能：根据点对和目标模型类型拟合全局变换矩阵。
/// 作用：统一封装直接法点对上的 RANSAC / LMEDS 几何估计入口，并在 RIGID 模式下强制回归无尺度刚体矩阵。
inline bool fitGlobalTransform(const std::vector<cv::Point2f>& srcPts,
                               const std::vector<cv::Point2f>& dstPts,
                               const std::string& fitModel,
                               const RobustFitOptions& options,
                               cv::Mat& A,
                               cv::Mat& H,
                               std::vector<unsigned char>& mask,
                               GeometryType& type,
                               const std::string& logPrefix) {
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
            !partial_affine_utils::refineRigidFromMask(
                srcPts, dstPts, ransacThreshold, mask, A)) {
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
    IR_LOG_WARN(logPrefix, ": unsupported fit_model=", fitModel);
    return false;
}

} // namespace ir::direct_geometry_common
