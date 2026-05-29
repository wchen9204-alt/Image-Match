#pragma once

#include <opencv2/core.hpp>

#include "core/context.h"

namespace ir {

// ---------------------------------------------------------------------------
// DrawOverlay：生成变换后源图与目标图的叠加图。
// ---------------------------------------------------------------------------
class DrawOverlay {
public:
    struct Options {
        double alpha = 0.5;       // 变换后源图的权重。
    };

    static cv::Mat render(const RegistrationContext& ctx, const Options& opt);
};

} // namespace ir
