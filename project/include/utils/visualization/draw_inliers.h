#pragma once

#include <opencv2/core.hpp>

#include "core/context.h"

namespace ir {

// ---------------------------------------------------------------------------
// DrawInliers：只绘制几何估计后的内点匹配。
// ---------------------------------------------------------------------------
class DrawInliers {
public:
    struct Options {
        int        max_inliers      = 200;
        cv::Scalar inlier_color     = cv::Scalar(0, 255, 0);
        cv::Scalar non_inlier_color = cv::Scalar(0, 0, 255);
        bool       draw_outliers    = false;
    };

    static cv::Mat render(const RegistrationContext& ctx, const Options& opt);
};

} // namespace ir
