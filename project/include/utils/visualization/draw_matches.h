#pragma once

#include <opencv2/core.hpp>

#include "core/context.h"

namespace ir {

/// 绘制两张图像之间的匹配连线。
class DrawMatches {
public:
    /// 绘制选项。
    struct Options {
        bool       draw_inliers_only = true;
        int        max_matches       = 100;
        cv::Scalar match_color       = cv::Scalar(0, 255, 0);
        cv::Scalar single_point      = cv::Scalar(0, 0, 255);
    };

    /// 渲染匹配可视化图，返回 BGR 图像。
    static cv::Mat render(const RegistrationContext& ctx, const Options& opt);
};

} // namespace ir
