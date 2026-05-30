#include "transform/perspective_warper.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"

namespace ir {

bool PerspectiveWarper::warp(RegistrationContext& ctx) {
    const auto& fd = ctx.feature_data;
    const auto& gd = ctx.geometry_data;

    if (fd.first.image.empty() || fd.second.image.empty()) {
        IR_LOG_ERROR("PerspectiveWarper: source images are empty.");
        return false;
    }

    cv::Mat H;
    if (gd.type == GeometryType::HOMOGRAPHY && !gd.H.empty()) {
        gd.H.convertTo(H, CV_64F);
    } else if ((gd.type == GeometryType::AFFINE || gd.type == GeometryType::RIGID ||
                gd.type == GeometryType::SIMILARITY) &&
               !gd.A.empty()) {
        // 将 2x3 仿射族矩阵扩展成 3x3，便于统一使用 warpPerspective。
        H = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat A64;
        gd.A.convertTo(A64, CV_64F);
        A64.copyTo(H(cv::Rect(0, 0, 3, 2)));
    } else {
        IR_LOG_WARN("PerspectiveWarper: only HOMOGRAPHY/AFFINE/RIGID/SIMILARITY are warpable; "
                    "geometry type is ",
                    toString(gd.type),
                    ". Skipping warp.");
        ctx.warped_image.release();
        return false;
    }

    cv::warpPerspective(fd.first.image,
                        ctx.warped_image,
                        H,
                        fd.second.image.size(),
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
