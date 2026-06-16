#include "utils/visualization/draw_overlay.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"

namespace ir {

cv::Mat DrawOverlay::render(const RegistrationContext& ctx, const Options& opt) {
    const cv::Mat& warped = ctx.warped_image;
    const cv::Mat& target = ctx.images.second;

    if (warped.empty() || target.empty()) {
        IR_LOG_WARN("DrawOverlay: warped or target empty.");
        return {};
    }
    if (warped.size() != target.size() || warped.type() != target.type()) {
        IR_LOG_WARN("DrawOverlay: warped and target shape/type differ; cannot blend.");
        return {};
    }

    // 透明度限制在 [0, 1]，保证叠加语义稳定且不依赖调用方做额外校验。
    double a = opt.alpha;
    if (a < 0.0)
        a = 0.0;
    if (a > 1.0)
        a = 1.0;

    // 叠加图主要服务于人工观察结构对齐关系，因此采用对称权重混合。
    cv::Mat blend;
    cv::addWeighted(warped, a, target, 1.0 - a, 0.0, blend);
    return blend;
}

} // namespace ir

