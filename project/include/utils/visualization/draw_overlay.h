#pragma once

#include <opencv2/core.hpp>

#include "core/context.h"

namespace ir {

/// 将变换后的源图与目标图叠加显示。
class DrawOverlay {
public:
    /// 绘制选项。
    struct Options {
        double alpha = 0.5;
    };

    /// 渲染叠加图。
    static cv::Mat render(const RegistrationContext& ctx, const Options& opt);
};

} // namespace ir

