#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

#include "core/context.h"

namespace ir::partial_affine_utils {

/// 从过滤后的匹配集合中提取两组对应点，供相似/刚体估计复用。
inline void extractPoints(const RegistrationContext& ctx,
                          std::vector<cv::Point2f>& pts1,
                          std::vector<cv::Point2f>& pts2) {
    const auto& fd = ctx.keypoint_data;
    const auto& md = ctx.keypoint_match_data;
    pts1.clear();
    pts2.clear();
    pts1.reserve(md.filtered.size());
    pts2.reserve(md.filtered.size());
    for (const auto& m : md.filtered) {
        pts1.push_back(fd.first.keypoints[m.queryIdx].pt);
        pts2.push_back(fd.second.keypoints[m.trainIdx].pt);
    }
}

/// 将内点掩码回写到上下文，统一维护 `inlier_mask` 与 `inliers` 两份结果。
inline void promoteInliers(RegistrationContext& ctx, const std::vector<unsigned char>& mask) {
    auto& md = ctx.keypoint_match_data;
    md.inlier_mask = mask;
    md.inliers.clear();
    md.inliers.reserve(mask.size());
    for (size_t i = 0; i < md.filtered.size() && i < mask.size(); ++i) {
        if (mask[i]) {
            md.inliers.push_back(md.filtered[i]);
        }
    }
}

/// 基于质心对齐和 SVD 分解求解二维刚体变换。
inline bool estimateRigid2D(const std::vector<cv::Point2f>& src,
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

    cv::SVD svd(H, cv::SVD::FULL_UV);
    cv::Mat R = svd.vt.t() * svd.u.t();
    if (cv::determinant(R) < 0.0) {
        cv::Mat V = svd.vt.t();
        V.col(1) *= -1.0;
        R = V * svd.u.t();
    }

    cv::Mat c1m = (cv::Mat_<double>(2, 1) << c1.x, c1.y);
    cv::Mat c2m = (cv::Mat_<double>(2, 1) << c2.x, c2.y);
    cv::Mat t = c2m - R * c1m;

    A = cv::Mat::zeros(2, 3, CV_64F);
    R.copyTo(A(cv::Rect(0, 0, 2, 2)));
    A.at<double>(0, 2) = t.at<double>(0, 0);
    A.at<double>(1, 2) = t.at<double>(1, 0);
    return true;
}

/// 按重投影误差阈值重新评估全量匹配的内外点归属。
inline std::vector<unsigned char> maskByReprojection(const std::vector<cv::Point2f>& src,
                                                     const std::vector<cv::Point2f>& dst,
                                                     const cv::Mat& A,
                                                     double threshold) {
    std::vector<unsigned char> mask(src.size(), 0);
    if (src.size() != dst.size() || A.empty()) {
        return mask;
    }

    const double thr2 = threshold * threshold;
    for (size_t i = 0; i < src.size(); ++i) {
        const double x = src[i].x;
        const double y = src[i].y;
        const double px = A.at<double>(0, 0) * x + A.at<double>(0, 1) * y + A.at<double>(0, 2);
        const double py = A.at<double>(1, 0) * x + A.at<double>(1, 1) * y + A.at<double>(1, 2);
        const double dx = px - dst[i].x;
        const double dy = py - dst[i].y;
        if (dx * dx + dy * dy <= thr2) {
            mask[i] = 1;
        }
    }
    return mask;
}

/// 生成统一的模型拒绝信息，便于不同估计器保持日志口径一致。
inline std::string rejectMessage(const std::string& kind, int inliers, int minInliers) {
    return "estimated " + kind + " with " + std::to_string(inliers) +
           " inliers, below minInliers=" + std::to_string(minInliers);
}

} // namespace ir::partial_affine_utils
