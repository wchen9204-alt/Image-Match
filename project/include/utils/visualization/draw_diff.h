#pragma once

#include <opencv2/core.hpp>

#include "core/context.h"

namespace ir {

/// 显示变换后图像与目标图之间的差异。
class DrawDiff {
public:
    /// 绘制选项。
    struct Options {
        bool heatmap = true;
        double scale = 1.0;
    };

    /// 渲染差异图。
    static cv::Mat render(const RegistrationContext& ctx, const Options& opt);
};

} // namespace ir
