#include "pipeline/base_pipeline_helpers.h"

#include <algorithm>
#include <cctype>
#include <cmath>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/logger.h"

namespace fs = std::filesystem;

namespace ir::base_pipeline_helpers {
namespace {

// TIFF 常见于高位深场景，优先走保留原始位深的读取分支。
bool hasTiffExtension(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext == ".tif" || ext == ".tiff";
}

// 在保留数值位深的前提下完成灰度化，为后续统一归一化预处理提供输入。
bool toGrayPreserveDepth(const cv::Mat& src, cv::Mat& gray) {
    if (src.channels() == 1) {
        gray = src;
        return true;
    }
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        return true;
    }
    if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
        return true;
    }
    return false;
}

// 将单通道图像压缩到 8 位动态范围，兼容多数 OpenCV 算子对输入类型的要求。
bool convertGrayTo8U(const cv::Mat& gray, cv::Mat& gray8) {
    if (gray.empty() || gray.channels() != 1) {
        return false;
    }
    if (gray.depth() == CV_8U) {
        gray8 = gray.clone();
        return true;
    }

    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(gray, &minVal, &maxVal);
    if (!std::isfinite(minVal) || !std::isfinite(maxVal) || maxVal <= minVal) {
        gray8 = cv::Mat::zeros(gray.size(), CV_8U);
        return true;
    }

    const double scale = 255.0 / (maxVal - minVal);
    gray.convertTo(gray8, CV_8U, scale, -minVal * scale);
    return true;
}

// 将任意支持的图像格式转换成 8 位灰度图，供可视化和前景叠加使用。
bool toGray8ForVisualization(const cv::Mat& image, cv::Mat& gray8) {
    cv::Mat gray;
    if (!toGrayPreserveDepth(image, gray)) {
        return false;
    }
    return convertGrayTo8U(gray, gray8);
}

} // namespace

// 读取 pipeline 输入图像，同时输出显示用 BGR 图和算法用 8 位灰度图。
bool loadImageForPipeline(const fs::path& path, cv::Mat& color, cv::Mat& gray) {
    color.release();
    gray.release();

    if (!hasTiffExtension(path)) {
        color = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (!color.empty()) {
            cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
            return true;
        }
    }

    cv::Mat raw = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (raw.empty()) {
        return false;
    }

    if (raw.depth() == CV_8U) {
        if (raw.channels() == 1) {
            gray = raw.clone();
            cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
            return true;
        }
        if (raw.channels() == 3) {
            color = raw.clone();
            cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
            return true;
        }
        if (raw.channels() == 4) {
            cv::cvtColor(raw, color, cv::COLOR_BGRA2BGR);
            cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
            return true;
        }
    }

    cv::Mat nativeGray;
    if (!toGrayPreserveDepth(raw, nativeGray) || !convertGrayTo8U(nativeGray, gray)) {
        return false;
    }

    cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
    if (raw.depth() != CV_8U) {
        IR_LOG_INFO("Loaded and normalized high-depth image: ",
                    path.string(),
                    " (depth=",
                    raw.depth(),
                    ", channels=",
                    raw.channels(),
                    ")");
    }
    return true;
}

// 根据灰度阈值生成前景 mask；黑色背景不会计入后续覆盖率统计。
bool buildForegroundMask(const cv::Mat& image, int thresholdValue, cv::Mat& mask) {
    mask.release();
    if (image.empty()) {
        return false;
    }

    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        return false;
    }

    cv::Mat gray8;
    if (!convertGrayTo8U(gray, gray8)) {
        return false;
    }

    cv::threshold(gray8,
                  mask,
                  static_cast<double>(std::clamp(thresholdValue, 0, 255)),
                  255.0,
                  cv::THRESH_BINARY);
    return true;
}

// 计算 warped 与 target 在 overlapMask 区域内的归一化平均绝对灰度差。
double computePhotometricError(const cv::Mat& warped,
                               const cv::Mat& target,
                               const cv::Mat& overlapMask) {
    if (overlapMask.empty() || cv::countNonZero(overlapMask) == 0) {
        return -1.0;
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

    cv::Mat warpedFloat;
    cv::Mat targetFloat;
    warpedGray.convertTo(warpedFloat, CV_32F, 1.0 / 255.0);
    targetGray.convertTo(targetFloat, CV_32F, 1.0 / 255.0);

    cv::Mat diff;
    cv::absdiff(warpedFloat, targetFloat, diff);

    const cv::Scalar meanDiff = cv::mean(diff, overlapMask);
    return meanDiff[0];
}

// 计算两个前景 mask 的交并比，主要用于结构重叠验证。
double computeMaskIou(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return -1.0;
    }

    cv::Mat intersectionMask;
    cv::Mat unionMask;
    cv::bitwise_and(a, b, intersectionMask);
    cv::bitwise_or(a, b, unionMask);

    const int unionCount = cv::countNonZero(unionMask);
    if (unionCount == 0) {
        return -1.0;
    }

    return static_cast<double>(cv::countNonZero(intersectionMask)) /
           static_cast<double>(unionCount);
}

// 计算 warped source 与 target 的局部包含率，支持一张图是另一张图局部的场景。
double computeMaskLocalContainment(const cv::Mat& sourceMask,
                                   const cv::Mat& warpedSourceMask,
                                   const cv::Mat& targetMask) {
    if (sourceMask.empty() || warpedSourceMask.empty() || targetMask.empty() ||
        warpedSourceMask.size() != targetMask.size()) {
        return -1.0;
    }

    const int sourceCount = cv::countNonZero(sourceMask);
    const int targetCount = cv::countNonZero(targetMask);
    const int denominator = std::min(sourceCount, targetCount);
    if (denominator <= 0) {
        return -1.0;
    }

    cv::Mat intersectionMask;
    cv::bitwise_and(warpedSourceMask, targetMask, intersectionMask);
    return static_cast<double>(cv::countNonZero(intersectionMask)) /
           static_cast<double>(denominator);
}

// 计算前景 mask 经过 warp 后仍落在目标画布内的比例。
double computeMaskCoverage(const cv::Mat& originalMask, const cv::Mat& warpedMask) {
    if (originalMask.empty() || warpedMask.empty()) {
        return -1.0;
    }

    const int originalCount = cv::countNonZero(originalMask);
    if (originalCount <= 0) {
        return -1.0;
    }

    return static_cast<double>(cv::countNonZero(warpedMask)) /
           static_cast<double>(originalCount);
}

// 统计最终内点在 source / target 前景包围盒中的空间覆盖率。
double computeInlierSpatialCoverage(const std::vector<cv::KeyPoint>& sourceKeypoints,
                                    const std::vector<cv::KeyPoint>& targetKeypoints,
                                    const std::vector<cv::DMatch>& inlierMatches,
                                    const cv::Mat& sourceMask,
                                    const cv::Mat& targetMask,
                                    double& sourceCoverage,
                                    double& targetCoverage) {
    sourceCoverage = -1.0;
    targetCoverage = -1.0;
    if (inlierMatches.empty() || sourceMask.empty() || targetMask.empty()) {
        return -1.0;
    }

    std::vector<cv::Point2f> sourcePoints;
    std::vector<cv::Point2f> targetPoints;
    sourcePoints.reserve(inlierMatches.size());
    targetPoints.reserve(inlierMatches.size());
    for (const auto& match : inlierMatches) {
        if (match.queryIdx < 0 || match.trainIdx < 0 ||
            match.queryIdx >= static_cast<int>(sourceKeypoints.size()) ||
            match.trainIdx >= static_cast<int>(targetKeypoints.size())) {
            continue;
        }
        sourcePoints.push_back(sourceKeypoints[match.queryIdx].pt);
        targetPoints.push_back(targetKeypoints[match.trainIdx].pt);
    }
    if (sourcePoints.empty() || targetPoints.empty()) {
        return -1.0;
    }

    cv::Rect sourceForegroundBox = cv::boundingRect(sourceMask);
    cv::Rect targetForegroundBox = cv::boundingRect(targetMask);
    if (sourceForegroundBox.area() <= 0 || targetForegroundBox.area() <= 0) {
        return -1.0;
    }

    const cv::Rect2f sourceInlierBox = cv::boundingRect(sourcePoints);
    const cv::Rect2f targetInlierBox = cv::boundingRect(targetPoints);
    sourceCoverage =
        static_cast<double>(sourceInlierBox.area()) /
        static_cast<double>(sourceForegroundBox.area());
    targetCoverage =
        static_cast<double>(targetInlierBox.area()) /
        static_cast<double>(targetForegroundBox.area());
    sourceCoverage = std::clamp(sourceCoverage, 0.0, 1.0);
    targetCoverage = std::clamp(targetCoverage, 0.0, 1.0);
    return std::max(sourceCoverage, targetCoverage);
}

// 从当前上下文中提取可用于 warp 的 2D 变换矩阵。
bool activeTransformMatrix(const RegistrationContext& ctx, cv::Mat& matrix) {
    matrix.release();
    if (!ctx.geometry_data.A.empty()) {
        ctx.geometry_data.A.convertTo(matrix, CV_64F);
        return matrix.rows >= 2 && matrix.cols >= 3;
    }
    if (!ctx.geometry_data.H.empty()) {
        ctx.geometry_data.H.convertTo(matrix, CV_64F);
        return matrix.rows >= 3 && matrix.cols >= 3;
    }
    if (!ctx.transform_data.M.empty()) {
        ctx.transform_data.M.convertTo(matrix, CV_64F);
        return matrix.rows >= 2 && matrix.cols >= 3;
    }
    return false;
}

// 按需对二值 mask 做形态学膨胀，增强细线或稀疏结构的重叠稳定性。
void dilateMaskIfRequested(cv::Mat& mask, int dilateSize) {
    if (mask.empty() || dilateSize <= 1) {
        return;
    }
    if (dilateSize % 2 == 0) {
        ++dilateSize;
    }
    const cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilateSize, dilateSize));
    cv::dilate(mask, mask, kernel);
}

// 将二值 mask 按当前 2D 变换 warp 到指定画布尺寸。
bool warpMaskToTargetSize(const cv::Mat& sourceMask,
                          const cv::Size& targetSize,
                          const cv::Mat& matrix,
                          cv::Mat& warpedMask) {
    warpedMask.release();
    if (sourceMask.empty() || targetSize.width <= 0 || targetSize.height <= 0 ||
        matrix.empty()) {
        return false;
    }

    if (matrix.rows >= 3 && matrix.cols >= 3) {
        cv::warpPerspective(sourceMask,
                            warpedMask,
                            matrix,
                            targetSize,
                            cv::INTER_NEAREST,
                            cv::BORDER_CONSTANT,
                            cv::Scalar(0));
        return true;
    }

    if (matrix.rows >= 2 && matrix.cols >= 3) {
        const cv::Mat affine = matrix(cv::Rect(0, 0, 3, 2)).clone();
        cv::warpAffine(sourceMask,
                       warpedMask,
                       affine,
                       targetSize,
                       cv::INTER_NEAREST,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(0));
        return true;
    }
    return false;
}

// 计算当前 2D 变换矩阵的逆矩阵，支持 2x3 仿射和 3x3 透视矩阵。
bool invertTransformMatrix(const cv::Mat& matrix, cv::Mat& inverseMatrix) {
    inverseMatrix.release();
    if (matrix.empty()) {
        return false;
    }

    if (matrix.rows >= 3 && matrix.cols >= 3) {
        return cv::invert(matrix, inverseMatrix, cv::DECOMP_SVD);
    }

    if (matrix.rows >= 2 && matrix.cols >= 3) {
        cv::Mat affine3x3 = cv::Mat::eye(3, 3, CV_64F);
        matrix(cv::Rect(0, 0, 3, 2)).copyTo(affine3x3(cv::Rect(0, 0, 3, 2)));
        cv::Mat inverse3x3;
        if (!cv::invert(affine3x3, inverse3x3, cv::DECOMP_SVD)) {
            return false;
        }
        inverseMatrix = inverse3x3(cv::Rect(0, 0, 3, 2)).clone();
        return true;
    }
    return false;
}

// 使用上下文里的生效变换，将结构 mask warp 到目标画布。
bool warpStructureMask(const RegistrationContext& ctx,
                       const cv::Mat& sourceMask,
                       const cv::Size& targetSize,
                       cv::Mat& warpedMask) {
    cv::Mat matrix;
    if (!activeTransformMatrix(ctx, matrix)) {
        return false;
    }
    return warpMaskToTargetSize(sourceMask, targetSize, matrix, warpedMask);
}

// 构建伪彩色重叠图：warped source 为红色通道，target 为绿色通道。
bool buildFalseColorOverlay(const cv::Mat& warped,
                            const cv::Mat& target,
                            int foregroundThreshold,
                            cv::Mat& overlay) {
    overlay.release();
    if (warped.empty() || target.empty() || warped.size() != target.size()) {
        return false;
    }

    cv::Mat warpedGray;
    cv::Mat targetGray;
    if (!toGray8ForVisualization(warped, warpedGray) ||
        !toGray8ForVisualization(target, targetGray)) {
        return false;
    }

    const int thresholdValue = std::clamp(foregroundThreshold, 0, 255);
    cv::Mat warpedMask;
    cv::Mat targetMask;
    cv::threshold(warpedGray, warpedMask, thresholdValue, 255.0, cv::THRESH_BINARY);
    cv::threshold(targetGray, targetMask, thresholdValue, 255.0, cv::THRESH_BINARY);

    cv::Mat warpedRed = cv::Mat::zeros(warpedGray.size(), CV_8U);
    cv::Mat targetGreen = cv::Mat::zeros(targetGray.size(), CV_8U);
    warpedGray.copyTo(warpedRed, warpedMask);
    targetGray.copyTo(targetGreen, targetMask);

    cv::Mat blue = cv::Mat::zeros(warpedGray.size(), CV_8U);
    cv::Mat channels[] = {blue, targetGreen, warpedRed};
    cv::merge(channels, 3, overlay);
    return true;
}

} // namespace ir::base_pipeline_helpers
