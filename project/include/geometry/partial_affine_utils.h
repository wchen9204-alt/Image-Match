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

/// 用 SVD 在二维点集上估计无缩放刚体变换，只保留旋转和平移。
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
    c1.x /= static_cast<double>(src.size());
    c1.y /= static_cast<double>(src.size());
    c2.x /= static_cast<double>(dst.size());
    c2.y /= static_cast<double>(dst.size());

    cv::Mat H = cv::Mat::zeros(2, 2, CV_64F);
    for (size_t i = 0; i < src.size(); ++i) {
        const cv::Point2d p(src[i].x - c1.x, src[i].y - c1.y);
        const cv::Point2d q(dst[i].x - c2.x, dst[i].y - c2.y);
        H.at<double>(0, 0) += p.x * q.x;
        H.at<double>(0, 1) += p.x * q.y;
        H.at<double>(1, 0) += p.y * q.x;
        H.at<double>(1, 1) += p.y * q.y;
    }

    // SVD 求最接近当前点集关系的旋转矩阵；若 det<0 则修正反射，保证结果仍是旋转。
    cv::SVD svd(H, cv::SVD::FULL_UV);
    cv::Mat R = svd.vt.t() * svd.u.t();
    if (cv::determinant(R) < 0.0) {
        cv::Mat V = svd.vt.t();
        V.col(1) *= -1.0;
        R = V * svd.u.t();
    }

    const cv::Mat c1m = (cv::Mat_<double>(2, 1) << c1.x, c1.y);
    const cv::Mat c2m = (cv::Mat_<double>(2, 1) << c2.x, c2.y);
    const cv::Mat t = c2m - R * c1m;

    A = cv::Mat::zeros(2, 3, CV_64F);
    R.copyTo(A(cv::Rect(0, 0, 2, 2)));
    A.at<double>(0, 2) = t.at<double>(0, 0);
    A.at<double>(1, 2) = t.at<double>(1, 0);
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

/// 基于 OpenCV RANSAC 筛出的内点，迭代回归严格刚体模型，并重新筛选内点。
/// 这是“先用 partial affine 找内点，再用最小二乘压回 s=1”的主路径。
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

    // 1. 归一化初始 mask：正常情况下它来自 OpenCV partial affine RANSAC。
    //    若 mask 长度异常，则先用当前 A 重新筛一次；仍不可用时退化为全量候选。
    std::vector<unsigned char> currentMask = mask;
    if (currentMask.size() != n) {
        currentMask = maskByReprojection(src, dst, A, threshold);
    }
    if (currentMask.size() != n) {
        currentMask.assign(n, 1);
    }

    cv::Mat bestA;
    std::vector<unsigned char> bestMask;
    int bestInliers = 0;
    double bestError = std::numeric_limits<double>::infinity();

    constexpr int kMaxRefineIters = 10;
    for (int iter = 0; iter < kMaxRefineIters; ++iter) {
        std::vector<cv::Point2f> inlierSrc;
        std::vector<cv::Point2f> inlierDst;
        // 2. 用当前内点回归 s=1 的刚体模型。
        collectMaskedPoints(src, dst, currentMask, inlierSrc, inlierDst);
        if (inlierSrc.size() < 2) {
            break;
        }

        cv::Mat candidateA;
        if (!estimateRigidNoScale2D(inlierSrc, inlierDst, candidateA)) {
            break;
        }

        // 3. 用新的刚体模型重新投影全部候选点，得到下一轮 mask。
        std::vector<unsigned char> candidateMask =
            maskByReprojection(src, dst, candidateA, threshold);
        const int candidateInliers = countInliers(candidateMask);
        const double candidateError =
            reprojectionErrorSum(src, dst, candidateMask, candidateA);

        if (logIterations) {
            IR_LOG_TRACE("Rigid SVD refine iter=",
                        iter,
                        ", input_inliers=",
                        inlierSrc.size(),
                        ", candidate_inliers=",
                        candidateInliers,
                        ", candidate_error=",
                        candidateError,
                        ", best_inliers=",
                        bestInliers,
                        ", best_error=",
                        bestError);
        }

        // 4. 保留内点更多的结果；内点相同则选择重投影误差更小的一轮。
        if (candidateInliers > bestInliers ||
            (candidateInliers == bestInliers && candidateError < bestError)) {
            bestInliers = candidateInliers;
            bestError = candidateError;
            bestA = candidateA;
            bestMask = candidateMask;
        }

        if (candidateMask == currentMask) {
            break;
        }
        currentMask = std::move(candidateMask);
    }

    if (bestInliers < 2 || bestA.empty()) {
        return false;
    }

    A = bestA;
    mask = bestMask;
    return true;
}

/// 统一生成“内点数不足”的拒绝信息，便于不同几何估计器复用。
inline std::string rejectMessage(const std::string& kind, int inliers, int minInliers) {
    return "estimated " + kind + " with " + std::to_string(inliers) +
           " inliers, below minInliers=" + std::to_string(minInliers);
}

} // namespace ir::partial_affine_utils

