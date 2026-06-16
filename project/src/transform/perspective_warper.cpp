#include "transform/perspective_warper.h"

#include <cmath>

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"

namespace ir {

bool PerspectiveWarper::warp(RegistrationContext& ctx) {
    const auto& images = ctx.images;
    const auto& gd = ctx.geometry_data;

    if (images.first.empty() || images.second.empty()) {
        IR_LOG_ERROR("PerspectiveWarper: source images are empty.");
        return false;
    }

    cv::Mat H;
    if (gd.type == GeometryType::HOMOGRAPHY && !gd.H.empty()) {
        gd.H.convertTo(H, CV_64F);
        IR_LOG_INFO("PerspectiveWarper using homography matrix.");
    } else if ((gd.type == GeometryType::AFFINE || gd.type == GeometryType::RIGID ||
                gd.type == GeometryType::SIMILARITY) &&
               !gd.A.empty()) {
        // 将 2x3 仿射族矩阵扩展成 3x3，便于统一使用 warpPerspective。
        H = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat A64;
        gd.A.convertTo(A64, CV_64F);
        A64.copyTo(H(cv::Rect(0, 0, 3, 2)));

        const double a00 = A64.at<double>(0, 0);
        const double a10 = A64.at<double>(1, 0);
        const double scale = std::sqrt(a00 * a00 + a10 * a10);
        const double rotationDeg = std::atan2(a10, a00) * 180.0 / CV_PI;
        IR_LOG_INFO("PerspectiveWarper using ",
                    toString(gd.type),
                    " matrix: scale=",
                    scale,
                    ", rotationDeg=",
                    rotationDeg,
                    ", tx=",
                    A64.at<double>(0, 2),
                    ", ty=",
                    A64.at<double>(1, 2));
    } else {
        IR_LOG_WARN("PerspectiveWarper: only HOMOGRAPHY/AFFINE/RIGID/SIMILARITY are warpable; "
                    "geometry type is ",
                    toString(gd.type),
                    ". Skipping warp.");
        ctx.warped_image.release();
        return false;
    }

    cv::warpPerspective(images.first,
                        ctx.warped_image,
                        H,
                        images.second.size(),
                        cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT,
                        cv::Scalar::all(0));

    IR_LOG_INFO("PerspectiveWarper produced ",
                ctx.warped_image.cols,
                "x",
                ctx.warped_image.rows,
                " image.");
    return !ctx.warped_image.empty();
}

} // namespace ir

