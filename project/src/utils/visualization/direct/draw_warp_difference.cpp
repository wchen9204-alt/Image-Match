#include "utils/visualization/direct/draw_warp_difference.h"

#include <opencv2/imgproc.hpp>

namespace ir {

/// 将 warped 与 target 的绝对差渲染成伪彩色热力图，便于肉眼观察局部配准误差。
cv::Mat renderWarpDifference(const cv::Mat& warped, const cv::Mat& target) {
    if (warped.empty() || target.empty() || warped.size() != target.size()) {
        return {};
    }

    cv::Mat warpedGray;
    cv::Mat targetGray;
    if (warped.channels() == 1) {
        warpedGray = warped;
    } else {
        cv::cvtColor(warped, warpedGray, cv::COLOR_BGR2GRAY);
    }
    if (target.channels() == 1) {
        targetGray = target;
    } else {
        cv::cvtColor(target, targetGray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat diff;
    cv::absdiff(warpedGray, targetGray, diff);

    cv::Mat diff8;
    if (diff.depth() == CV_8U) {
        diff8 = diff;
    } else {
        cv::normalize(diff, diff8, 0, 255, cv::NORM_MINMAX, CV_8U);
    }

    cv::Mat colored;
    cv::applyColorMap(diff8, colored, cv::COLORMAP_TURBO);
    return colored;
}

} // namespace ir

