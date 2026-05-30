#include "transform/affine_warper.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"

namespace ir {

bool AffineWarper::warp(RegistrationContext& ctx) {
    const auto& fd = ctx.feature_data;
    const auto& gd = ctx.geometry_data;

    if (fd.first.image.empty() || fd.second.image.empty()) {
        IR_LOG_ERROR("AffineWarper: source images empty.");
        return false;
    }

    cv::Mat A;
    if ((gd.type == GeometryType::AFFINE || gd.type == GeometryType::RIGID ||
         gd.type == GeometryType::SIMILARITY) &&
        !gd.A.empty()) {
        gd.A.convertTo(A, CV_64F);
    } else if (gd.type == GeometryType::HOMOGRAPHY && !gd.H.empty()) {
        // 仅在 H 本质上是仿射变换时，取前两行作为 2x3 矩阵。
        cv::Mat H64;
        gd.H.convertTo(H64, CV_64F);
        A = H64(cv::Rect(0, 0, 3, 2)).clone();
        IR_LOG_WARN("AffineWarper: extracted 2x3 from a homography matrix; "
                    "this is exact only if the homography is affine.");
    } else {
        IR_LOG_WARN("AffineWarper: unsupported geometry type ", toString(gd.type));
        ctx.warped_image.release();
        return false;
    }

    cv::warpAffine(fd.first.image,
                   ctx.warped_image,
                   A,
                   fd.second.image.size(),
                   cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT,
                   cv::Scalar::all(0));

    IR_LOG_INFO(
        "AffineWarper produced ", ctx.warped_image.cols, "x", ctx.warped_image.rows, " image.");
    return !ctx.warped_image.empty();
}

} // namespace ir
