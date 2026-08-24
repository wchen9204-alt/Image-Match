#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

#include "core/context.h"
#include "data/correspondence_view.h"
#include "utils/logger.h"

namespace ir::partial_affine_utils {

inline bool refineRigidFromMask(const std::vector<cv::Point2f>& src,
                                const std::vector<cv::Point2f>& dst,
                                double threshold,
                                std::vector<unsigned char>& mask,
                                cv::Mat& A,
                                bool logIterations);

/// 从统一对应关系视图中提取点坐标，供几何估计器复用。
inline void extractPoints(const CorrespondenceView& view,
                          std::vector<cv::Point2f>& pts1,
                          std::vector<cv::Point2f>& pts2) {
    pts1.clear();
    pts2.clear();
    pts1.reserve(view.filtered.size());
    pts2.reserve(view.filtered.size());
    for (const auto& m : view.filtered) {
        if (m.queryIdx < 0 || m.trainIdx < 0 ||
            m.queryIdx >= static_cast<int>(view.first_keypoints.size()) ||
            m.trainIdx >= static_cast<int>(view.second_keypoints.size())) {
            continue;
        }
        pts1.push_back(view.first_keypoints[m.queryIdx].pt);
        pts2.push_back(view.second_keypoints[m.trainIdx].pt);
    }
}

/// 将几何估计阶段得到的内点掩码写回通用结果，并在点特征/学习法中同步生成内点匹配列表。
inline void promoteInliers(RegistrationContext& ctx,
                           const CorrespondenceView& view,
                           const std::vector<unsigned char>& mask) {
    auto& gd = ctx.geometry_data;
    gd.inlier_mask = mask;
    gd.correspondence_source = toString(view.source);
    gd.num_correspondences = view.filteredCount();

    if (view.source == CorrespondenceSource::KEYPOINT || view.source == CorrespondenceSource::LEARNING) {
        auto& md = ctx.keypoint_match_data;
        md.inlier_mask = mask;
        md.inlier_matches.clear();
        md.inlier_matches.reserve(mask.size());
        for (size_t i = 0; i < view.filtered.size() && i < mask.size(); ++i) {
            if (mask[i]) {
                md.inlier_matches.push_back(view.filtered[i]);
            }
        }
    }
}

/// 根据掩码收集当前内点对应，用于后续刚体重估或再次筛选。
inline void collectMaskedPoints(const std::vector<cv::Point2f>& src,
                                const std::vector<cv::Point2f>& dst,
                                const std::vector<unsigned char>& mask,
                                std::vector<cv::Point2f>& inlierSrc,
                                std::vector<cv::Point2f>& inlierDst) {
    inlierSrc.clear();
    inlierDst.clear();
    const size_t n = std::min(src.size(), dst.size());
    inlierSrc.reserve(n);
    inlierDst.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!mask.empty() && (i >= mask.size() || !mask[i])) {
            continue;
        }
        inlierSrc.push_back(src[i]);
        inlierDst.push_back(dst[i]);
    }
}

/// 用参考实现的质心、叉积和点积估计无缩放刚体变换。
/// 这是纯 rigid 约束的最小二乘回归步骤，不包含 RANSAC。
inline bool estimateRigidNoScale2D(const std::vector<cv::Point2f>& src,
                                   const std::vector<cv::Point2f>& dst,
                                   cv::Mat& A) {
    if (src.size() != dst.size() || src.size() < 2) {
        return false;
    }

    cv::Point2d c1(0.0, 0.0);
    cv::Point2d c2(0.0, 0.0);
    for (size_t i = 0; i < src.size(); ++i) {
        c1.x += src[i].x;
        c1.y += src[i].y;
        c2.x += dst[i].x;
        c2.y += dst[i].y;
    }
    const double count = static_cast<double>(src.size());
    c1.x /= count;
    c1.y /= count;
    c2.x /= count;
    c2.y /= count;

    double sumCross = 0.0;
    double sumDot = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        const double sx = static_cast<double>(src[i].x) - c1.x;
        const double sy = static_cast<double>(src[i].y) - c1.y;
        const double tx = static_cast<double>(dst[i].x) - c2.x;
        const double ty = static_cast<double>(dst[i].y) - c2.y;
        sumCross += sx * ty - sy * tx;
        sumDot += sx * tx + sy * ty;
    }

    if (std::abs(sumCross) < 1e-12 && std::abs(sumDot) < 1e-12) {
        return false;
    }

    const double angle = std::atan2(sumCross, sumDot);
    const double cosAngle = std::cos(angle);
    const double sinAngle = std::sin(angle);
    const double tx = c2.x - (cosAngle * c1.x - sinAngle * c1.y);
    const double ty = c2.y - (sinAngle * c1.x + cosAngle * c1.y);

    A = (cv::Mat_<double>(2, 3) <<
         cosAngle, -sinAngle, tx,
         sinAngle, cosAngle, ty);
    return true;
}
/// 按给定模型做重投影筛选，生成新的内点掩码。
/// 该函数用于把初始模型或精修模型重新投影回全部候选点。
inline std::vector<unsigned char> maskByReprojection(const std::vector<cv::Point2f>& src,
                                                     const std::vector<cv::Point2f>& dst,
                                                     const cv::Mat& A,
                                                     double threshold) {
    std::vector<unsigned char> mask(std::min(src.size(), dst.size()), 0);
    if (A.empty() || A.rows < 2 || A.cols < 3) {
        return mask;
    }

    const double threshold2 = threshold * threshold;
    for (size_t i = 0; i < mask.size(); ++i) {
        const double x = src[i].x;
        const double y = src[i].y;
        const double dx = A.at<double>(0, 0) * x + A.at<double>(0, 1) * y +
                          A.at<double>(0, 2) - dst[i].x;
        const double dy = A.at<double>(1, 0) * x + A.at<double>(1, 1) * y +
                          A.at<double>(1, 2) - dst[i].y;
        if (dx * dx + dy * dy <= threshold2) {
            mask[i] = 1;
        }
    }
    return mask;
}

/// 计算当前模型在内点上的重投影平方误差，用于多轮 refine 时选择更稳定的结果。
inline double reprojectionErrorSum(const std::vector<cv::Point2f>& src,
                                   const std::vector<cv::Point2f>& dst,
                                   const std::vector<unsigned char>& mask,
                                   const cv::Mat& A) {
    if (A.empty() || A.rows < 2 || A.cols < 3) {
        return std::numeric_limits<double>::infinity();
    }

    const size_t n = std::min(src.size(), dst.size());
    double error = 0.0;
    bool hasInlier = false;
    for (size_t i = 0; i < n; ++i) {
        if (!mask.empty() && (i >= mask.size() || !mask[i])) {
            continue;
        }

        const double x = src[i].x;
        const double y = src[i].y;
        const double dx = A.at<double>(0, 0) * x + A.at<double>(0, 1) * y +
                          A.at<double>(0, 2) - dst[i].x;
        const double dy = A.at<double>(1, 0) * x + A.at<double>(1, 1) * y +
                          A.at<double>(1, 2) - dst[i].y;
        error += dx * dx + dy * dy;
        hasInlier = true;
    }
    return hasInlier ? error : std::numeric_limits<double>::infinity();
}

/// 统计掩码中的内点数量，供日志、阈值判断和结果回写复用。
inline int countInliers(const std::vector<unsigned char>& mask) {
    int count = 0;
    for (unsigned char v : mask) {
        if (v) {
            ++count;
        }
    }
    return count;
}

/// 直接在 rigid 模型下执行自定义 RANSAC，不再经过 estimateAffinePartial2D。
/// 最小样本使用两对点，模型始终满足 s=1，只允许旋转和平移。
inline bool estimateRigidRansacNoScale2D(const std::vector<cv::Point2f>& src,
                                         const std::vector<cv::Point2f>& dst,
                                         double threshold,
                                         int maxIters,
                                         double confidence,
                                         cv::Mat& A,
                                         std::vector<unsigned char>& mask,
                                         bool logIterations = false) {
    const size_t n = std::min(src.size(), dst.size());
    if (n < 2) {
        return false;
    }

    auto pairDistance2 = [](const cv::Point2f& a, const cv::Point2f& b) {
        const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
        const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
        return dx * dx + dy * dy;
    };

    auto updateRequiredIters = [&](int inliers) {
        if (inliers < 2) {
            return std::max(1, maxIters);
        }

        const double ratio = static_cast<double>(inliers) / static_cast<double>(n);
        const double sampleSuccess = ratio * ratio;
        if (sampleSuccess >= 1.0) {
            return 1;
        }
        if (sampleSuccess <= 0.0) {
            return std::max(1, maxIters);
        }

        const double safeConfidence = std::clamp(confidence, 1e-9, 1.0 - 1e-9);
        const double denom = std::log(1.0 - sampleSuccess);
        if (!std::isfinite(denom) || std::abs(denom) < 1e-12) {
            return std::max(1, maxIters);
        }

        const double numer = std::log(1.0 - safeConfidence);
        const int required = static_cast<int>(std::ceil(numer / denom));
        return std::clamp(required, 1, std::max(1, maxIters));
    };

    cv::Mat bestA;
    std::vector<unsigned char> bestMask;
    int bestInliers = 0;
    double bestError = std::numeric_limits<double>::infinity();

    cv::RNG rng(0x5EED1234u);
    int requiredIters = std::max(1, maxIters);
    for (int iter = 0; iter < requiredIters; ++iter) {
        int i = 0;
        int j = 1;
        if (n > 2) {
            i = rng.uniform(0, static_cast<int>(n));
            j = rng.uniform(0, static_cast<int>(n - 1));
            if (j >= i) {
                ++j;
            }
        }

        // 两对点过近时会让旋转方向不稳定，直接跳过该假设。
        if (pairDistance2(src[i], src[j]) <= 1e-6 || pairDistance2(dst[i], dst[j]) <= 1e-6) {
            continue;
        }

        std::vector<cv::Point2f> sampleSrc = {src[i], src[j]};
        std::vector<cv::Point2f> sampleDst = {dst[i], dst[j]};
        cv::Mat candidateA;
        if (!estimateRigidNoScale2D(sampleSrc, sampleDst, candidateA)) {
            continue;
        }

        std::vector<unsigned char> candidateMask =
            maskByReprojection(src, dst, candidateA, threshold);
        const int candidateInliers = countInliers(candidateMask);
        const double candidateError =
            reprojectionErrorSum(src, dst, candidateMask, candidateA);

        if (logIterations) {
            IR_LOG_TRACE("Rigid custom RANSAC iter=",
                        iter,
                        ", pair=(",
                        i,
                        ",",
                        j,
                        "), inliers=",
                        candidateInliers,
                        ", error=",
                        candidateError,
                        ", best_inliers=",
                        bestInliers,
                        ", best_error=",
                        bestError);
        }

        if (candidateInliers > bestInliers ||
            (candidateInliers == bestInliers && candidateError < bestError)) {
            bestInliers = candidateInliers;
            bestError = candidateError;
            bestA = candidateA;
            bestMask = candidateMask;
            requiredIters = std::min(requiredIters, updateRequiredIters(candidateInliers));
        }
    }

    if (bestInliers < 2 || bestA.empty()) {
        return false;
    }

    // 在 best mask 上再做一次严格 rigid 回归，提升最终矩阵稳定性。
    if (!refineRigidFromMask(src, dst, threshold, bestMask, bestA, logIterations)) {
        return false;
    }

    A = bestA;
    mask = bestMask;
    return true;
}

/// 基于已有内点掩码回归一次严格刚体模型，不迭代更新内点掩码。
/// 这是参考实现的最终精修路径。
inline bool refineRigidFromMask(const std::vector<cv::Point2f>& src,
                                const std::vector<cv::Point2f>& dst,
                                double threshold,
                                std::vector<unsigned char>& mask,
                                cv::Mat& A,
                                bool logIterations = false) {
    const size_t n = std::min(src.size(), dst.size());
    if (n < 2) {
        return false;
    }

    // 归一化已有掩码；若没有可用掩码，则按当前模型重新分类一次。
    std::vector<unsigned char> currentMask = mask;
    if (currentMask.size() != n) {
        currentMask = maskByReprojection(src, dst, A, threshold);
    }
    if (currentMask.size() != n) {
        currentMask.assign(n, 1);
    }

    std::vector<cv::Point2f> inlierSrc;
    std::vector<cv::Point2f> inlierDst;
    collectMaskedPoints(src, dst, currentMask, inlierSrc, inlierDst);
    if (inlierSrc.size() < 2 || !estimateRigidNoScale2D(inlierSrc, inlierDst, A)) {
        return false;
    }

    if (logIterations) {
        IR_LOG_TRACE("Rigid reference refine inliers=", inlierSrc.size());
    }
    mask = currentMask;
    return true;
}
/// 统一生成“内点数不足”的拒绝信息，便于不同几何估计器复用。
inline std::string rejectMessage(const std::string& kind, int inliers, int minInliers) {
    return "estimated " + kind + " with " + std::to_string(inliers) +
           " inliers, below minInliers=" + std::to_string(minInliers);
}

} // namespace ir::partial_affine_utils

