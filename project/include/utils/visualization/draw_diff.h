#pragma once

#include <opencv2/core.hpp>

#include "core/context.h"

namespace ir {

// ---------------------------------------------------------------------------
// DrawDiff：生成变换后源图与目标图的差异图。
// ---------------------------------------------------------------------------
class DrawDiff {
public:
    struct Options {
        bool   heatmap   = true;
        double scale     = 1.0;   // 上色前的缩放系数。
    };

    static cv::Mat render(const RegistrationContext& ctx, const Options& opt);
};

} // namespace ir
