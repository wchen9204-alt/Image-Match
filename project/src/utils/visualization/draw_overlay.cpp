#include "utils/visualization/draw_overlay.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"

namespace ir {

cv::Mat DrawOverlay::render(const RegistrationContext& ctx, const Options& opt) {
    const cv::Mat& warped = ctx.warped_image;
    const cv::Mat& target = ctx.feature_data.second.image;

    if (warped.empty() || target.empty()) {
        IR_LOG_WARN("DrawOverlay: warped or target empty.");
        return {};
    }
    if (warped.size() != target.size() || warped.type() != target.type()) {
        IR_LOG_WARN("DrawOverlay: warped and target shape/type differ; cannot blend.");
        return {};
    }

    double a = opt.alpha;
    if (a < 0.0) a = 0.0;
    if (a > 1.0) a = 1.0;

    cv::Mat blend;
    cv::addWeighted(warped, a, target, 1.0 - a, 0.0, blend);
    return blend;
}

} // namespace ir
