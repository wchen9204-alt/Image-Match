#include "utils/image_utils.h"

#include <opencv2/imgproc.hpp>

namespace ir {
namespace image_utils {

cv::Mat toGrayFloat(const cv::Mat& src) {
    // 统一转为 [0, 1] 范围灰度浮点图，便于图像指标和差异度量复用同一输入规范。
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

int normalizedOddKernelOrZero(int value, int minSize) {
    if (value < minSize) {
        return 0;
    }
    if (value % 2 == 0) {
        ++value;
    }
    return value;
}

int normalizedOddKernel(int value, int fallback, int minimum) {
    if (value < minimum) {
        value = fallback;
    }
    if (value % 2 == 0) {
        ++value;
    }
    return value;
}

int normalizedCannyAperture(int value, int fallback) {
    if (value == 3 || value == 5 || value == 7) {
        return value;
    }
    return (fallback == 3 || fallback == 5 || fallback == 7) ? fallback : 3;
}

bool ensureGray(const cv::Mat& color, cv::Mat& gray) {
    if (!gray.empty()) {
        return gray.channels() == 1;
    }
    if (color.empty()) {
        return false;
    }
    if (color.channels() == 1) {
        gray = color;
        return true;
    }
    if (color.channels() == 3) {
        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
        return true;
    }
    if (color.channels() == 4) {
        cv::cvtColor(color, gray, cv::COLOR_BGRA2GRAY);
        return true;
    }
    return false;
}

void applyOptionalGaussianBlur(const cv::Mat& src, cv::Mat& dst, int blurKernel) {
    const int kernel = normalizedOddKernelOrZero(blurKernel);
    if (kernel > 0) {
        cv::GaussianBlur(src, dst, cv::Size(kernel, kernel), 0.0);
        return;
    }
    dst = src;
}

bool convertGrayToFloat01(const cv::Mat& gray, cv::Mat& out, int blurKernel) {
    out.release();
    if (gray.empty() || gray.channels() != 1) {
        return false;
    }

    gray.convertTo(out, CV_32F, 1.0 / 255.0);
    const int kernel = normalizedOddKernelOrZero(blurKernel);
    if (kernel > 0) {
        cv::GaussianBlur(out, out, cv::Size(kernel, kernel), 0.0);
    }
    return true;
}

cv::Mat warpedValidMask(const cv::Mat& src, const cv::Mat& H, const cv::Size& dst_size) {
    // 有效区域掩码通过对全 1 图做同样的几何变换获得，与实际插值路径保持一致。
    if (src.empty() || H.empty())
        return {};
    cv::Mat ones(src.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat warped_mask;
    cv::warpPerspective(
        ones, warped_mask, H, dst_size, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
    return warped_mask;
}

cv::Mat nonZeroMask(const cv::Mat& warped) {
    // 非零掩码用于近似提取变换后有效区域，适合没有显式几何矩阵的场景。
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
    // 裁剪共同有效区域时保持输入不可变，输出统一返回拷贝结果。
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
