#pragma once

#include <opencv2/core.hpp>

#include "core/context.h"

namespace ir {

// ---------------------------------------------------------------------------
// DrawMatches：绘制两张图之间的过滤匹配或内点匹配。
// ---------------------------------------------------------------------------
class DrawMatches {
public:
    struct Options {
        bool draw_inliers_only = true;
        int  max_matches       = 100;
        cv::Scalar match_color   = cv::Scalar(0, 255, 0);
        cv::Scalar single_point  = cv::Scalar(0, 0, 255);
    };

    // 返回 BGR 画布，调用者决定保存或显示。
    static cv::Mat render(const RegistrationContext& ctx, const Options& opt);
};

} // namespace ir
