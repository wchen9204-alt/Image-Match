#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

#include "core/context.h"
#include "data/correspondence_view.h"

namespace ir::partial_affine_utils {

/// 从统一对应点视图中提取两组点，供各类几何估计器复用。
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

/// 将几何估计内点掩码回写到通用几何结果；点特征/学习来源额外同步回 keypoint_match_data。
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
        md.inliers.clear();
        md.inliers.reserve(mask.size());
        for (size_t i = 0; i < view.filtered.size() && i < mask.size(); ++i) {
            if (mask[i]) {
                md.inliers.push_back(view.filtered[i]);
            }
        }
    }
}

/// 按内点掩码收集点对，供 OpenCV RANSAC 后的严格刚体回归复用。
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

/// 基于质心对齐和 SVD 求解严格二维刚体变换，只允许旋转和平移，不引入缩放。
/// OpenCV 当前没有“RANSAC + 无缩放 2D 刚体”一站式函数，因此这里仅承担最终无缩放回归。
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

    // SVD 给出最接近内点点集的正交旋转矩阵；det<0 时修正反射，保证结果是旋转。
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

/// 使用严格刚体矩阵重新评估全量点对内外点，避免沿用含缩放模型的 OpenCV 初始掩码。
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

/// 统计掩码中的内点数量，供点特征法、结构法和直接法共用验收逻辑。
inline int countInliers(const std::vector<unsigned char>& mask) {
    int count = 0;
    for (unsigned char v : mask) {
        if (v) {
            ++count;
        }
    }
    return count;
}

/// 对 OpenCV partial affine 初筛内点做严格刚体回归，并用严格模型重算内点。
/// 这样既保留 OpenCV RANSAC 的鲁棒筛选能力，又避免最终矩阵带入缩放自由度。
inline bool refineRigidFromMask(const std::vector<cv::Point2f>& src,
                                const std::vector<cv::Point2f>& dst,
                                double threshold,
                                std::vector<unsigned char>& mask,
                                cv::Mat& A) {
    std::vector<cv::Point2f> inlierSrc;
    std::vector<cv::Point2f> inlierDst;
    // 1. 根据 mask 收集当前内点
    collectMaskedPoints(src, dst, mask, inlierSrc, inlierDst);
    // 2. 用纯内点重新计算刚性变换
    if (!estimateRigidNoScale2D(inlierSrc, inlierDst, A)) {
        return false;
    }
    // 3. 用新矩阵 重新筛选内点
    mask = maskByReprojection(src, dst, A, threshold);
    // 4. 再次用新内点 再精修一次
    collectMaskedPoints(src, dst, mask, inlierSrc, inlierDst);
    if (inlierSrc.size() >= 2 && estimateRigidNoScale2D(inlierSrc, inlierDst, A)) {
        mask = maskByReprojection(src, dst, A, threshold);
    }
    return countInliers(mask) >= 2;
}

/// 生成统一的模型拒绝信息，便于不同估计器保持日志口径一致。
inline std::string rejectMessage(const std::string& kind, int inliers, int minInliers) {
    return "estimated " + kind + " with " + std::to_string(inliers) +
           " inliers, below minInliers=" + std::to_string(minInliers);
}

} // namespace ir::partial_affine_utils
