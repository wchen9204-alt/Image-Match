#include "utils/image_utils.h"

#include <opencv2/imgproc.hpp>

namespace ir {
namespace image_utils {

cv::Mat toGrayFloat(const cv::Mat& src) {
    if (src.empty())
        return {};
    cv::Mat gray;
    if (src.channels() == 1) {
        gray = src;
    } else if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    } else {
        cv::extractChannel(src, gray, 0);
    }
    cv::Mat f;
    gray.convertTo(f, CV_32F, 1.0 / 255.0);
    return f;
}

cv::Mat warpedValidMask(const cv::Mat& src, const cv::Mat& H, const cv::Size& dst_size) {
    if (src.empty() || H.empty())
        return {};
    cv::Mat ones(src.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat warped_mask;
    cv::warpPerspective(
        ones, warped_mask, H, dst_size, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
    return warped_mask;
}

cv::Mat nonZeroMask(const cv::Mat& warped) {
    if (warped.empty())
        return {};
    cv::Mat gray;
    if (warped.channels() == 1) {
        gray = warped;
    } else {
        cv::cvtColor(warped, gray, cv::COLOR_BGR2GRAY);
    }
    cv::Mat mask;
    cv::threshold(gray, mask, 0, 255, cv::THRESH_BINARY);
    if (mask.type() != CV_8UC1)
        mask.convertTo(mask, CV_8UC1);
    return mask;
}

void cropToMask(
    const cv::Mat& a, const cv::Mat& b, const cv::Mat& mask, cv::Mat& a_out, cv::Mat& b_out) {
    if (a.empty() || b.empty() || mask.empty()) {
        a_out = a.clone();
        b_out = b.clone();
        return;
    }
    const cv::Rect bbox = cv::boundingRect(mask);
    if (bbox.area() <= 0) {
        a_out = a.clone();
        b_out = b.clone();
        return;
    }
    a_out = a(bbox).clone();
    b_out = b(bbox).clone();
}

} // namespace image_utils
} // namespace ir
