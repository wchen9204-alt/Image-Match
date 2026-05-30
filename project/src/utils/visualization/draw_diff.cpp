#include "utils/visualization/draw_diff.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"

namespace ir {

cv::Mat DrawDiff::render(const RegistrationContext& ctx, const Options& opt) {
    const cv::Mat& warped = ctx.warped_image;
    const cv::Mat& target = ctx.feature_data.second.image;

    if (warped.empty() || target.empty()) {
        IR_LOG_WARN("DrawDiff: warped or target empty.");
        return {};
    }
    if (warped.size() != target.size()) {
        IR_LOG_WARN("DrawDiff: warped and target sizes differ.");
        return {};
    }

    cv::Mat a, b;
    if (warped.channels() == 1)
        a = warped;
    else
        cv::cvtColor(warped, a, cv::COLOR_BGR2GRAY);
    if (target.channels() == 1)
        b = target;
    else
        cv::cvtColor(target, b, cv::COLOR_BGR2GRAY);

    cv::Mat diff;
    cv::absdiff(a, b, diff);
    if (opt.scale != 1.0) {
        diff.convertTo(diff, diff.type(), opt.scale);
    }

    if (!opt.heatmap)
        return diff;

    cv::Mat colored;
    cv::applyColorMap(diff, colored, cv::COLORMAP_JET);
    return colored;
}

} // namespace ir
