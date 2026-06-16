#include "direct/frequency/phase_correlation_aligner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/image_utils.h"
#include "utils/logger.h"
#include "utils/string_utils.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

constexpr double kPhaseEps = 1e-12;

/// 根据梯度幅值生成空间权重图，用于 WPC 强调纹理更可靠的区域。
cv::Mat gradientWeight(const cv::Mat& image,
                       int blurKernel,
                       double power,
                       double floorValue) {
    // 先计算 x/y 方向梯度，为后续构造纹理强度权重图提供基础量。
    cv::Mat gx;
    cv::Mat gy;
    cv::Sobel(image, gx, CV_32F, 1, 0, 3);
    cv::Sobel(image, gy, CV_32F, 0, 1, 3);

    // 梯度幅值越大，表示该区域越有纹理，后续相位相关越值得信任。
    cv::Mat weight;
    cv::magnitude(gx, gy, weight);

    const int kernel = image_utils::normalizedOddKernelOrZero(blurKernel);
    if (kernel > 0) {
        // 对权重图做平滑，减少单个噪声边缘造成的权重尖峰。
        cv::GaussianBlur(weight, weight, cv::Size(kernel, kernel), 0.0);
    }

    double maxVal = 0.0;
    cv::minMaxLoc(weight, nullptr, &maxVal);
    if (maxVal > kPhaseEps) {
        // 归一化到 [0, 1]，让不同图像的梯度权重具有可比性。
        weight.convertTo(weight, CV_32F, 1.0 / maxVal);
    } else {
        weight.setTo(0.0f);
    }

    const double effectivePower = power > 0.0 ? power : 1.0;
    if (std::abs(effectivePower - 1.0) > kPhaseEps) {
        // 幂指数用于调节高纹理区域的强调强度。
        cv::pow(weight, effectivePower, weight);
    }

    const double floorClamped = std::clamp(floorValue, 0.0, 1.0);
    if (floorClamped > 0.0) {
        // 设置权重下限，避免低纹理区域被完全压成 0 而失去全局结构信息。
        weight = weight * static_cast<float>(1.0 - floorClamped) +
                 static_cast<float>(floorClamped);
    }
    return weight;
}

/// 按配置把源图、目标图乘上空间权重；未启用或 mode=NONE 时保持原图。
void applyWeightedInputs(const cv::Mat& src,
                         const cv::Mat& dst,
                         const std::string& mode,
                         int blurKernel,
                         double power,
                         double floorValue,
                         cv::Mat& weightedSrc,
                         cv::Mat& weightedDst) {
    weightedSrc = src;
    weightedDst = dst;

    const std::string key = string_utils::normalizedKey(mode);
    if (key == "NONE") {
        return;
    }

    const bool sourceWeighted = key.empty() || key == "GRADIENT" || key == "BOTH" ||
                                key == "SOURCE" || key == "SOURCEGRADIENT";
    const bool targetWeighted = key.empty() || key == "GRADIENT" || key == "BOTH" ||
                                key == "TARGET" || key == "TARGETGRADIENT";
    if (!sourceWeighted && !targetWeighted) {
        return;
    }

    if (sourceWeighted) {
        // 对源图乘空间权重，让高纹理区域在频域相关中占更高比重。
        const cv::Mat w = gradientWeight(src, blurKernel, power, floorValue);
        cv::multiply(src, w, weightedSrc);
    }
    if (targetWeighted) {
        // 对目标图乘空间权重，保证两侧输入在同一权重语义下比较。
        const cv::Mat w = gradientWeight(dst, blurKernel, power, floorValue);
        cv::multiply(dst, w, weightedDst);
    }
}

/// 重建相位相关响应面，用于主峰/次峰和局部尖锐度诊断。
bool computePhaseResponseSurface(const cv::Mat& src,
                                 const cv::Mat& dst,
                                 const cv::Mat& window,
                                 cv::Mat& response) {
    cv::Mat srcWindowed = src;
    cv::Mat dstWindowed = dst;
    if (!window.empty()) {
        // 先乘 Hann 窗，减轻图像边界突变造成的频域泄漏。
        cv::multiply(src, window, srcWindowed);
        cv::multiply(dst, window, dstWindowed);
    }

    // 对两幅图做 DFT，并构造共轭互谱。
    cv::Mat srcDft;
    cv::Mat dstDft;
    cv::dft(srcWindowed, srcDft, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(dstWindowed, dstDft, cv::DFT_COMPLEX_OUTPUT);

    cv::Mat crossPower;
    cv::mulSpectrums(srcDft, dstDft, crossPower, 0, true);

    std::vector<cv::Mat> planes;
    cv::split(crossPower, planes);
    if (planes.size() != 2) {
        return false;
    }

    // 将互谱归一化成相位相关的单位模形式，只保留相位信息。
    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);
    cv::max(magnitude, cv::Scalar(kPhaseEps), magnitude);
    cv::divide(planes[0], magnitude, planes[0]);
    cv::divide(planes[1], magnitude, planes[1]);
    cv::merge(planes, crossPower);

    // 逆 DFT 得到响应面，后续可在空间域分析主峰和次峰结构。
    cv::dft(crossPower, response, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
    return !response.empty() && response.type() == CV_32F && cv::checkRange(response, true);
}

/// 以环绕方式读取响应图像素，匹配 DFT 周期边界语义。
double responseAtWrapped(const cv::Mat& response, int y, int x) {
    const int yy = (y % response.rows + response.rows) % response.rows;
    const int xx = (x % response.cols + response.cols) % response.cols;
    return static_cast<double>(response.at<float>(yy, xx));
}

/// 计算周期边界下的一维最短距离，用于在主峰周围排除次峰搜索区域。
int wrappedDistance(int a, int b, int period) {
    const int d = std::abs(a - b);
    return std::min(d, period - d);
}

/// 相位相关峰值诊断结果。
struct PeakDiagnostics {
    bool valid = false;
    double peakValue = 0.0;
    double secondPeakValue = 0.0;
    double peakRatio = 0.0;
    double peakSharpness = 0.0;
    double subpixelConfidence = 0.0;
};

/// 分析响应面峰值结构，得到主峰/次峰比、局部尖锐度和亚像素拟合置信度。
PeakDiagnostics analyzePeakDiagnostics(const cv::Mat& response, int exclusionRadius) {
    PeakDiagnostics diag;
    if (response.empty() || response.rows < 3 || response.cols < 3 || response.type() != CV_32F) {
        return diag;
    }

    // 先找到响应面主峰，作为平移估计最可能的位置。
    double maxVal = 0.0;
    cv::Point maxLoc;
    cv::minMaxLoc(response, nullptr, &maxVal, nullptr, &maxLoc);
    if (!std::isfinite(maxVal) || maxVal <= kPhaseEps) {
        return diag;
    }

    // 在主峰邻域外搜索次峰，用于判断主峰是否足够突出。
    const int radius = std::max(0, exclusionRadius);
    bool foundSecond = false;
    double secondPeak = -std::numeric_limits<double>::infinity();
    for (int y = 0; y < response.rows; ++y) {
        for (int x = 0; x < response.cols; ++x) {
            const int dx = wrappedDistance(x, maxLoc.x, response.cols);
            const int dy = wrappedDistance(y, maxLoc.y, response.rows);
            if (dx <= radius && dy <= radius) {
                continue;
            }

            const double v = responseAtWrapped(response, y, x);
            if (std::isfinite(v) && (!foundSecond || v > secondPeak)) {
                secondPeak = v;
                foundSecond = true;
            }
        }
    }

    // 统计 8 邻域均值，用于刻画主峰相对周边背景的尖锐程度。
    double neighborAbsSum = 0.0;
    int neighborCount = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            neighborAbsSum += std::abs(responseAtWrapped(response, maxLoc.y + dy, maxLoc.x + dx));
            ++neighborCount;
        }
    }

    const double center = maxVal;
    const double left = responseAtWrapped(response, maxLoc.y, maxLoc.x - 1);
    const double right = responseAtWrapped(response, maxLoc.y, maxLoc.x + 1);
    const double up = responseAtWrapped(response, maxLoc.y - 1, maxLoc.x);
    const double down = responseAtWrapped(response, maxLoc.y + 1, maxLoc.x);
    const double denomX = left - 2.0 * center + right;
    const double denomY = up - 2.0 * center + down;

    // 通过一维二次曲线近似检查主峰是否支持稳定的亚像素拟合。
    bool subpixelFitOk = denomX < -kPhaseEps && denomY < -kPhaseEps;
    if (subpixelFitOk) {
        const double offsetX = 0.5 * (left - right) / denomX;
        const double offsetY = 0.5 * (up - down) / denomY;
        subpixelFitOk = std::isfinite(offsetX) && std::isfinite(offsetY) &&
                        std::abs(offsetX) <= 1.0 && std::abs(offsetY) <= 1.0;
    }

    // 汇总主峰/次峰比、局部尖锐度和亚像素拟合置信度。
    const double meanNeighborAbs =
        neighborCount > 0 ? neighborAbsSum / static_cast<double>(neighborCount) : 0.0;
    diag.valid = true;
    diag.peakValue = maxVal;
    diag.secondPeakValue = foundSecond ? secondPeak : 0.0;
    diag.peakRatio = diag.secondPeakValue > kPhaseEps
                         ? diag.peakValue / diag.secondPeakValue
                         : std::numeric_limits<double>::infinity();
    diag.peakSharpness = diag.peakValue / (meanNeighborAbs + kPhaseEps);
    diag.subpixelConfidence = subpixelFitOk ? diag.peakSharpness : 0.0;
    return diag;
}

} // namespace

DirectPhaseCorrelationAligner::DirectPhaseCorrelationAligner(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    // 读取基础响应阈值、WPC 权重和峰值诊断相关配置。
    _responseThreshold = yaml_utils::getDouble(params, "response_threshold", 0.01);
    _blurKernel = yaml_utils::getInt(params, "blur_kernel", 5);
    _useHannWindow = yaml_utils::getBool(params, "use_hann_window", true);
    _weighted = yaml_utils::getBool(params, "weighted", false);
    _weightMode = yaml_utils::getString(params, "weight_mode", "GRADIENT");
    _weightBlurKernel = yaml_utils::getInt(params, "weight_blur_kernel", 0);
    _weightPower = yaml_utils::getDouble(params, "weight_power", 1.0);
    _weightFloor = yaml_utils::getDouble(params, "weight_floor", 0.05);
    _confidenceCheck = yaml_utils::getBool(params, "confidence_check", false);
    _peakRatioThreshold = yaml_utils::getDouble(params, "peak_ratio_threshold", 0.0);
    _subpixelConfidenceThreshold =
        yaml_utils::getDouble(params, "subpixel_confidence_threshold", 0.0);
    _peakExclusionRadius = std::max(
        0, yaml_utils::getInt(params, "peak_exclusion_radius", 5));
    if (_peakRatioThreshold > 0.0 || _subpixelConfidenceThreshold > 0.0) {
        _confidenceCheck = true;
    }
}

bool DirectPhaseCorrelationAligner::align(RegistrationContext& ctx) {
    auto& dd = ctx.direct_data;
    auto& gd = ctx.geometry_data;
    dd.clear();
    gd.clear();
    dd.method = name();

    // 1. 预处理输入灰度图；当前相位相关直接法要求两幅图尺寸一致。
    cv::Mat src;
    cv::Mat dst;
    if (!image_utils::convertGrayToFloat01(ctx.images.first_gray, src, _blurKernel) ||
        !image_utils::convertGrayToFloat01(ctx.images.second_gray, dst, _blurKernel)) {
        dd.message = "phase correlation requires non-empty grayscale images";
        gd.message = dd.message;
        return false;
    }
    if (src.size() != dst.size()) {
        dd.message = "phase correlation requires images with the same size";
        gd.message = dd.message;
        return false;
    }

    // 2. 可选执行加权相位相关 WPC，用纹理权重抑制低信息区域对峰值的干扰。
    cv::Mat phaseSrc = src;
    cv::Mat phaseDst = dst;
    if (_weighted) {
        applyWeightedInputs(src,
                            dst,
                            _weightMode,
                            _weightBlurKernel,
                            _weightPower,
                            _weightFloor,
                            phaseSrc,
                            phaseDst);
    }

    cv::Mat window;
    if (_useHannWindow) {
        cv::createHanningWindow(window, src.size(), CV_32F);
    }

    // 3. 在频域估计平移位移，并先用 OpenCV 返回的相关响应做第一层筛选。
    double score = 0.0;
    const cv::Point2d shift = cv::phaseCorrelate(phaseSrc, phaseDst, window, &score);
    if (!std::isfinite(shift.x) || !std::isfinite(shift.y) || !std::isfinite(score)) {
        dd.message = "phase correlation returned non-finite result";
        gd.message = dd.message;
        return false;
    }
    if (score < _responseThreshold) {
        dd.message = "phase correlation response below threshold: " + std::to_string(score);
        gd.message = dd.message;
        IR_LOG_WARN("Direct phase correlation rejected: ", dd.message);
        return false;
    }

    PeakDiagnostics diag;
    if (_confidenceCheck) {
        // 额外重建响应面，分析主峰/次峰结构和亚像素峰值是否可靠。
        cv::Mat response;
        if (!computePhaseResponseSurface(phaseSrc, phaseDst, window, response)) {
            dd.message = "phase correlation response surface is invalid";
            gd.message = dd.message;
            IR_LOG_WARN("Direct phase correlation rejected: ", dd.message);
            return false;
        }

        diag = analyzePeakDiagnostics(response, _peakExclusionRadius);
        if (!diag.valid) {
            dd.message = "phase correlation peak diagnostics are invalid";
            gd.message = dd.message;
            IR_LOG_WARN("Direct phase correlation rejected: ", dd.message);
            return false;
        }
        if (_peakRatioThreshold > 0.0 && diag.peakRatio < _peakRatioThreshold) {
            dd.message = "phase correlation peak ratio below threshold: " +
                         std::to_string(diag.peakRatio);
            gd.message = dd.message;
            IR_LOG_WARN("Direct phase correlation rejected: ", dd.message);
            return false;
        }
        if (_subpixelConfidenceThreshold > 0.0 &&
            diag.subpixelConfidence < _subpixelConfidenceThreshold) {
            dd.message = "phase correlation subpixel confidence below threshold: " +
                         std::to_string(diag.subpixelConfidence);
            gd.message = dd.message;
            IR_LOG_WARN("Direct phase correlation rejected: ", dd.message);
            return false;
        }
    }

    // 4. 将平移结果写成仿射矩阵，并把峰值诊断写回统一直接法输出结构。
    gd.type = GeometryType::AFFINE;
    gd.A = (cv::Mat_<double>(2, 3) << 1.0, 0.0, shift.x, 0.0, 1.0, shift.y);
    gd.valid = true;
    gd.num_inliers = 1;
    gd.inlier_ratio = score;

    dd.A = gd.A.clone();
    dd.valid = true;
    dd.score = score;
    if (diag.valid) {
        dd.addDiagnostic("peak_ratio", "peak ratio", diag.peakRatio);
        dd.addDiagnostic("peak_sharpness", "peak sharpness", diag.peakSharpness);
        dd.addDiagnostic("subpixel_confidence", "subpixel conf", diag.subpixelConfidence);
    }

    IR_LOG_INFO("Direct phase correlation dx=",
                shift.x,
                ", dy=",
                shift.y,
                ", score=",
                score,
                ", weighted=",
                _weighted ? "true" : "false",
                ", peak_ratio=",
                diag.peakRatio,
                ", subpixel_confidence=",
                diag.subpixelConfidence);
    return true;
}

} // namespace ir

