#include "direct/frequency/fourier_mellin_aligner.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "core/types.h"
#include "utils/image_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

constexpr double kEps = 1e-12;

/// 将角度压到 [-180, 180) 区间，便于候选去重和日志阅读。
double normalizeAngleDeg(double angleDeg) {
    while (angleDeg >= 180.0) {
        angleDeg -= 360.0;
    }
    while (angleDeg < -180.0) {
        angleDeg += 360.0;
    }
    return angleDeg;
}

/// 交换频谱象限，把低频移动到图像中心，便于后续 log-polar 采样。
void fftShift(cv::Mat& mag) {
    const int cx = mag.cols / 2;
    const int cy = mag.rows / 2;
    if (cx <= 0 || cy <= 0) {
        return;
    }

    cv::Mat q0(mag, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(mag, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(mag, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(mag, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

/// 计算平移不变的频谱幅值图，供 Fourier-Mellin 估计旋转 & 缩放，完全剔除平移影响。
bool computeMagnitudeSpectrum(const cv::Mat& image,
                              const cv::Mat& window,
                              int dcSuppressRadius,
                              int blurKernel,
                              cv::Mat& spectrum) {
    if (image.empty() || image.type() != CV_32F) {
        return false;
    }

    cv::Mat input;
    if (!window.empty()) {
        // 先乘 Hann 窗，图像边界突变会在频域产生大量噪声，乘窗函数让图像边缘平滑衰减到 0。
        cv::multiply(image, window, input);
    } else {
        image.copyTo(input);
    }

    // 对输入图做二维 DFT，把图像从空间域 → 频率域，输出是复数矩阵，图像的旋转、缩放会体现在频谱结构上，图像的平移会被隐藏在相位中，不影响幅度。
    cv::Mat complexSpectrum;
    cv::dft(input, complexSpectrum, cv::DFT_COMPLEX_OUTPUT);

    std::vector<cv::Mat> planes;
    cv::split(complexSpectrum, planes);
    if (planes.size() != 2) {
        return false;
    }

    // 计算复数幅度，并用 log 压缩动态范围。
    cv::magnitude(planes[0], planes[1], spectrum);
    cv::log(spectrum + cv::Scalar::all(1.0), spectrum);

    // 交换频谱象限，把低频中心移到图像中央，便于后续极坐标采样。
    fftShift(spectrum);

    if (dcSuppressRadius > 0) {
        // 抑制中心 DC 分量，避免整体亮度主导旋转/尺度峰值。
        cv::circle(spectrum,
                   cv::Point(spectrum.cols / 2, spectrum.rows / 2),
                   dcSuppressRadius,
                   cv::Scalar::all(0),
                   cv::FILLED);
    }

    const int kernel = image_utils::normalizedOddKernelOrZero(blurKernel);
    if (kernel > 0) {
        // 对频谱幅值图再做平滑，弱化局部噪声尖峰。
        cv::GaussianBlur(spectrum, spectrum, cv::Size(kernel, kernel), 0.0);
    }

    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(spectrum, &minVal, &maxVal);
    if (maxVal - minVal > kEps) {
        // 归一化到 [0, 1]，让不同层和不同图像的响应幅值可比。
        cv::normalize(spectrum, spectrum, 0.0, 1.0, cv::NORM_MINMAX);
    }
    return cv::checkRange(spectrum, true);
}

/// 将频谱幅值图转成 log-polar 图；旋转和尺度在该空间中表现为平移。
bool buildLogPolarSpectrum(const cv::Mat& spectrum,
                           cv::Size logPolarSize,
                           cv::Mat& logPolar,
                           double& logScaleFactor) {
    if (spectrum.empty() || spectrum.type() != CV_32F) {
        return false;
    }

    if (logPolarSize.width <= 0) {
        logPolarSize.width = spectrum.cols;
    }
    if (logPolarSize.height <= 0) {
        logPolarSize.height = spectrum.rows;
    }
    if (logPolarSize.width < 8 || logPolarSize.height < 8) {
        return false;
    }

    const cv::Point2f center((spectrum.cols - 1) * 0.5f, (spectrum.rows - 1) * 0.5f);
    const double maxRadius = std::min(center.x, center.y);
    if (maxRadius <= 1.0) {
        return false;
    }

    // 将频谱幅值映射到 log-polar 空间，把旋转和尺度变化转成平移问题。
    cv::warpPolar(spectrum,
                  logPolar,
                  logPolarSize,
                  center,
                  maxRadius,
                  cv::INTER_LINEAR | cv::WARP_POLAR_LOG | cv::WARP_FILL_OUTLIERS);

    // 记录横向位移到 log(scale) 的换算系数，后续用于恢复真实尺度。
    logScaleFactor = static_cast<double>(logPolarSize.width) / std::log(maxRadius);
    return !logPolar.empty() && logPolar.type() == CV_32F && cv::checkRange(logPolar, true);
}

/// 为 phaseCorrelate 构造 Hann 窗；关闭 windowed 时返回空矩阵。
cv::Mat makeHannWindow(cv::Size size, bool enabled) {
    cv::Mat window;
    if (enabled && size.width > 1 && size.height > 1) {
        cv::createHanningWindow(window, size, CV_32F);
    }
    return window;
}

struct RotScaleCandidate {
    double angleDeg = 0.0;
    double scale = 1.0;
    double response = 0.0;
    int level = 0;
};

struct EvaluatedCandidate {
    RotScaleCandidate candidate;
    cv::Point2d translation;
    double translationResponse = 0.0;
    cv::Mat A;
};

/// 记录旋转/尺度候选搜索阶段的诊断信息，便于分析为何未产出候选。
struct RotScaleSearchStats {
    int levelsAttempted = 0;
    int spectrumLevelsOk = 0;
    int logPolarLevelsOk = 0;
    int finiteResponseLevels = 0;
    int responseAcceptedLevels = 0;
    int candidateLevelsAdded = 0;
    int totalCandidatesAdded = 0;
    double bestResponse = -std::numeric_limits<double>::infinity();
    double bestShiftX = std::numeric_limits<double>::quiet_NaN();
    double bestShiftY = std::numeric_limits<double>::quiet_NaN();
    double bestRawAngleDeg = std::numeric_limits<double>::quiet_NaN();
    double bestRawLogScale = std::numeric_limits<double>::quiet_NaN();
    double bestScaleFromPositiveLog = std::numeric_limits<double>::quiet_NaN();
    double bestScaleFromNegativeLog = std::numeric_limits<double>::quiet_NaN();
};

/// 添加旋转/尺度候选，并去掉数值上重复的候选。
void addCandidate(std::vector<RotScaleCandidate>& candidates,
                  double angleDeg,
                  double scale,
                  double response,
                  int level,
                  double minScale,
                  double maxScale) {
    if (!std::isfinite(angleDeg) || !std::isfinite(scale) || scale < minScale ||
        scale > maxScale) {
        return;
    }

    // 先把角度归一化到统一范围，避免 180/-180 一类表示重复。
    angleDeg = normalizeAngleDeg(angleDeg);
    for (const auto& existing : candidates) {
        const double angleDiff = std::abs(normalizeAngleDeg(existing.angleDeg - angleDeg));
        const double scaleDiff = std::abs(existing.scale - scale) / std::max(existing.scale, kEps);
        if (angleDiff < 1e-3 && scaleDiff < 1e-3) {
            return;
        }
    }

    // 只保留数值上不重复的候选，减少后续逐个 warp 评估的开销。
    candidates.push_back({angleDeg, scale, response, level});
}

/// 根据 log-polar 相位相关位移生成候选；正负符号和 180 度歧义都交给后续评分筛选。
void expandShiftToCandidates(const cv::Point2d& shift,
                             cv::Size logPolarSize,
                             double logScaleFactor,
                             double response,
                             int level,
                             bool tryHalfTurnAmbiguity,
                             double minScale,
                             double maxScale,
                             std::vector<RotScaleCandidate>& candidates) {
    if (logPolarSize.width <= 0 || logPolarSize.height <= 0 || logScaleFactor <= kEps) {
        return;
    }

    const double rawAngle = shift.y * 360.0 / static_cast<double>(logPolarSize.height);
    const double rawLogScale = shift.x / logScaleFactor;
    const double angleSigns[] = {1.0, -1.0};
    const double scaleSigns[] = {1.0, -1.0};

    // 对同一个 log-polar 位移同时尝试角度符号和 log-scale 符号两种解释。
    for (const double angleSign : angleSigns) {
        for (const double scaleSign : scaleSigns) {
            const double angle = angleSign * rawAngle;
            const double scale = std::exp(scaleSign * rawLogScale);
            addCandidate(candidates, angle, scale, response, level, minScale, maxScale);
            if (tryHalfTurnAmbiguity) {
                // 频谱幅值存在中心对称性，额外补一个 180 度歧义候选。
                addCandidate(candidates,
                             angle + 180.0,
                             scale,
                             response,
                             level,
                             minScale,
                             maxScale);
            }
        }
    }
}

/// 构建从原图到目标图的相似变换候选，并用平移 phase correlation 评估该候选。
bool evaluateCandidate(const cv::Mat& src,
                       const cv::Mat& dst,
                       const cv::Mat& translationWindow,
                       const RotScaleCandidate& candidate,
                       EvaluatedCandidate& out) {
    const cv::Point2f center((src.cols - 1) * 0.5f, (src.rows - 1) * 0.5f);

    // 先根据候选角度和尺度构造从源图到目标图的相似变换矩阵。
    cv::Mat A = cv::getRotationMatrix2D(center, candidate.angleDeg, candidate.scale);

    cv::Mat warped;
    // 只应用旋转和尺度，把剩余平移交给 phase correlation 继续估计。
    cv::warpAffine(src,
                   warped,
                   A,
                   dst.size(),
                   cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT,
                   cv::Scalar::all(0));
    if (warped.empty()) {
        return false;
    }

    double response = 0.0;
    // 在已对齐角度和尺度的前提下，用 phase correlation 估计最终平移。
    const cv::Point2d shift = cv::phaseCorrelate(warped, dst, translationWindow, &response);
    if (!std::isfinite(shift.x) || !std::isfinite(shift.y) || !std::isfinite(response)) {
        return false;
    }

    // 将平移直接补进 2x3 矩阵，得到完整的相似变换结果。
    A.convertTo(A, CV_64F);
    A.at<double>(0, 2) += shift.x;
    A.at<double>(1, 2) += shift.y;

    out.candidate = candidate;
    out.translation = shift;
    out.translationResponse = response;
    out.A = A;
    return true;
}

/// 按配置构建图像金字塔；不开启 pyramid 时只返回原图。
std::vector<cv::Mat> buildPyramidImages(const cv::Mat& image,
                                        bool usePyramid,
                                        int levels,
                                        double scaleFactor) {
    std::vector<cv::Mat> pyr;
    pyr.push_back(image);
    if (!usePyramid) {
        return pyr;
    }

    levels = std::max(1, levels);
    // 缩放系数限制在 [0.25, 0.95] 之间
    scaleFactor = std::clamp(scaleFactor, 0.25, 0.95);
    for (int i = 1; i < levels; ++i) {
        cv::Mat next;
        // 对上一层图像进行区域插值缩放
        cv::resize(pyr.back(), next, cv::Size(), scaleFactor, scaleFactor, cv::INTER_AREA);
        if (next.cols < 64 || next.rows < 64) {
            break;
        }
        pyr.push_back(next);
    }
    return pyr;
}

} // namespace

DirectFourierMellinAligner::DirectFourierMellinAligner(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    // 读取图像预处理、金字塔、log-polar 和候选筛选相关配置。
    _blurKernel = yaml_utils::getInt(params, "blur_kernel", 3);
    _windowed = yaml_utils::getBool(params, "windowed", true);
    _usePyramid = yaml_utils::getBool(params, "use_pyramid", false);
    _pyramidLevels = std::max(1, yaml_utils::getInt(params, "pyramid_levels", 3));
    _pyramidScale = std::clamp(yaml_utils::getDouble(params, "pyramid_scale", 0.5), 0.25, 0.95);
    _logPolarCols = yaml_utils::getInt(params, "log_polar_cols", 0);
    _logPolarRows = yaml_utils::getInt(params, "log_polar_rows", 0);
    _magnitudeBlurKernel = yaml_utils::getInt(params, "magnitude_blur_kernel", 3);
    _dcSuppressRadius = yaml_utils::getInt(params, "dc_suppress_radius", 3);
    _minScale = std::max(1e-3, yaml_utils::getDouble(params, "min_scale", 0.25));
    _maxScale = std::max(_minScale, yaml_utils::getDouble(params, "max_scale", 4.0));
    _rotationScaleResponseThreshold =
        yaml_utils::getDouble(params, "rotation_scale_response_threshold", 0.0);
    _translationResponseThreshold =
        yaml_utils::getDouble(params, "translation_response_threshold", 0.01);
    _tryHalfTurnAmbiguity = yaml_utils::getBool(params, "try_half_turn_ambiguity", true);
}

bool DirectFourierMellinAligner::align(RegistrationContext& ctx) {
    auto& dd = ctx.direct_data;
    auto& gd = ctx.geometry_data;
    dd.clear();
    gd.clear();
    dd.method = name();

    // 1. 转为32位灰度图 + 高斯模糊降噪；Fourier-Mellin 当前要求两幅图尺寸一致。
    cv::Mat src;
    cv::Mat dst;
    if (!image_utils::convertGrayToFloat01(ctx.images.first_gray, src, _blurKernel) ||
        !image_utils::convertGrayToFloat01(ctx.images.second_gray, dst, _blurKernel)) {
        dd.message = "fourier mellin requires non-empty grayscale images";
        gd.message = dd.message;
        return false;
    }
    if (src.size() != dst.size()) {
        dd.message = "fourier mellin requires images with the same size";
        gd.message = dd.message;
        return false;
    }

    // 2. 构建图像金字塔，低分辨率层快速粗匹配，高分辨率层精匹配
    const std::vector<cv::Mat> srcPyr =
        buildPyramidImages(src, _usePyramid, _pyramidLevels, _pyramidScale);
    const std::vector<cv::Mat> dstPyr =
        buildPyramidImages(dst, _usePyramid, _pyramidLevels, _pyramidScale);
    const int pyramidCount = static_cast<int>(std::min(srcPyr.size(), dstPyr.size()));

    // 3. 在每个金字塔层的频谱幅值图上估计旋转/尺度位移，并展开为候选集合。
    std::vector<RotScaleCandidate> candidates;
    RotScaleSearchStats searchStats;
    searchStats.levelsAttempted = pyramidCount;
    for (int level = 0; level < pyramidCount; ++level) {
        const cv::Mat levelWindow = makeHannWindow(srcPyr[level].size(), _windowed);

        // 3.1 计算幅度谱（傅里叶变换）
        cv::Mat srcSpectrum;
        cv::Mat dstSpectrum;
        if (!computeMagnitudeSpectrum(srcPyr[level],
                                      levelWindow,
                                      _dcSuppressRadius,
                                      _magnitudeBlurKernel,
                                      srcSpectrum) ||
            !computeMagnitudeSpectrum(dstPyr[level],
                                      levelWindow,
                                      _dcSuppressRadius,
                                      _magnitudeBlurKernel,
                                      dstSpectrum)) {
            continue;
        }
        ++searchStats.spectrumLevelsOk;

        // 3.2 对数极坐标变换（梅林变换）
        cv::Mat srcLogPolar;
        cv::Mat dstLogPolar;
        double logScaleFactor = 0.0;
        const cv::Size logPolarSize(_logPolarCols > 0 ? _logPolarCols : srcSpectrum.cols,
                                    _logPolarRows > 0 ? _logPolarRows : srcSpectrum.rows);
        if (!buildLogPolarSpectrum(srcSpectrum, logPolarSize, srcLogPolar, logScaleFactor) ||
            !buildLogPolarSpectrum(dstSpectrum, logPolarSize, dstLogPolar, logScaleFactor)) {
            continue;
        }
        ++searchStats.logPolarLevelsOk;

        const cv::Mat logPolarWindow = makeHannWindow(srcLogPolar.size(), _windowed);
        double response = 0.0;
        // 3.3 相位相关（计算平移量）
        const cv::Point2d shift =
            cv::phaseCorrelate(srcLogPolar, dstLogPolar, logPolarWindow, &response);
        if (!std::isfinite(shift.x) || !std::isfinite(shift.y) || !std::isfinite(response)) {
            continue;
        }
        ++searchStats.finiteResponseLevels;

        const double rawAngleDeg =
            shift.y * 360.0 / static_cast<double>(std::max(srcLogPolar.rows, 1));
        const double rawLogScale = shift.x / std::max(logScaleFactor, kEps);
        if (response > searchStats.bestResponse) {
            searchStats.bestResponse = response;
            searchStats.bestShiftX = shift.x;
            searchStats.bestShiftY = shift.y;
            searchStats.bestRawAngleDeg = rawAngleDeg;
            searchStats.bestRawLogScale = rawLogScale;
            searchStats.bestScaleFromPositiveLog = std::exp(rawLogScale);
            searchStats.bestScaleFromNegativeLog = std::exp(-rawLogScale);
        }
        if (response < _rotationScaleResponseThreshold) {
            continue;
        }
        ++searchStats.responseAcceptedLevels;

        const size_t candidateCountBefore = candidates.size();
        
        // 3.4 生成旋缩候选
        expandShiftToCandidates(shift,
                                srcLogPolar.size(),
                                logScaleFactor,
                                response,
                                level,
                                _tryHalfTurnAmbiguity,
                                _minScale,
                                _maxScale,
                                candidates);
        const int addedCount = static_cast<int>(candidates.size() - candidateCountBefore);
        searchStats.totalCandidatesAdded += addedCount;
        if (addedCount > 0) {
            ++searchStats.candidateLevelsAdded;
        }
    }

    // 4. 若候选阶段完全失败，则把关键搜索诊断写回，便于判断是响应太弱还是候选被过滤掉。
    if (candidates.empty()) {
        dd.addDiagnostic("levels_attempted",
                         "levels attempted",
                         static_cast<double>(searchStats.levelsAttempted));
        dd.addDiagnostic("spectrum_levels_ok",
                         "spectrum levels",
                         static_cast<double>(searchStats.spectrumLevelsOk));
        dd.addDiagnostic("logpolar_levels_ok",
                         "log-polar levels",
                         static_cast<double>(searchStats.logPolarLevelsOk));
        dd.addDiagnostic("finite_response_levels",
                         "finite response levels",
                         static_cast<double>(searchStats.finiteResponseLevels));
        dd.addDiagnostic("response_accepted_levels",
                         "accepted response levels",
                         static_cast<double>(searchStats.responseAcceptedLevels));
        dd.addDiagnostic("candidate_levels_added",
                         "candidate levels",
                         static_cast<double>(searchStats.candidateLevelsAdded));
        dd.addDiagnostic("candidate_count", "candidate count", 0.0);
        if (std::isfinite(searchStats.bestResponse)) {
            dd.addDiagnostic("best_logpolar_response",
                             "best log-polar response",
                             searchStats.bestResponse);
            dd.addDiagnostic("best_logpolar_shift_x",
                             "best log-polar shift x",
                             searchStats.bestShiftX);
            dd.addDiagnostic("best_logpolar_shift_y",
                             "best log-polar shift y",
                             searchStats.bestShiftY);
            dd.addDiagnostic("best_raw_angle_deg",
                             "best raw angle deg",
                             searchStats.bestRawAngleDeg);
            dd.addDiagnostic("best_raw_log_scale",
                             "best raw log-scale",
                             searchStats.bestRawLogScale);
            dd.addDiagnostic("best_scale_pos",
                             "best scale +",
                             searchStats.bestScaleFromPositiveLog);
            dd.addDiagnostic("best_scale_neg",
                             "best scale -",
                             searchStats.bestScaleFromNegativeLog);
        }
        dd.message = "fourier mellin failed to produce rotation/scale candidates";
        gd.message = dd.message;
        IR_LOG_WARN("Direct Fourier-Mellin rejected: ", dd.message);
        return false;
    }

    // 5. 回到原图尺度逐个评估候选：先应用旋转/尺度，再用平移 phase correlation 选出最佳解。
    const cv::Mat translationWindow = makeHannWindow(src.size(), _windowed);
    bool found = false;
    EvaluatedCandidate best;
    // 遍历所有旋缩候选，估计最佳平移
    for (const auto& candidate : candidates) {
        EvaluatedCandidate evaluated;
        if (!evaluateCandidate(src, dst, translationWindow, candidate, evaluated)) {
            continue;
        }
        if (!found || evaluated.translationResponse > best.translationResponse ||
            (std::abs(evaluated.translationResponse - best.translationResponse) < 1e-9 &&
             evaluated.candidate.response > best.candidate.response)) {
            best = evaluated;
            found = true;
        }
    }

    if (!found || best.translationResponse < _translationResponseThreshold) {
        dd.message = "fourier mellin translation response below threshold";
        gd.message = dd.message;
        IR_LOG_WARN("Direct Fourier-Mellin rejected: ",
                    dd.message,
                    ", response=",
                    found ? best.translationResponse : -1.0);
        return false;
    }

    // 6. 将最佳候选写回统一的直接法/几何结果结构，并补充诊断项供摘要和 JSON 复用。
    gd.type = GeometryType::SIMILARITY;
    gd.A = best.A.clone();
    gd.valid = true;
    gd.num_inliers = 1;
    gd.inlier_ratio = best.translationResponse;
    gd.correspondence_source = "DIRECT";
    gd.num_correspondences = 1;

    dd.A = gd.A.clone();
    dd.valid = true;
    dd.score = best.translationResponse;
    dd.addDiagnostic("rotation_deg", "rotation deg", best.candidate.angleDeg);
    dd.addDiagnostic("scale", "scale", best.candidate.scale);
    dd.addDiagnostic("rotation_scale_response",
                     "rot-scale response",
                     best.candidate.response);
    dd.addDiagnostic("translation_response", "translation response", best.translationResponse);
    dd.addDiagnostic("candidate_count", "candidate count", static_cast<double>(candidates.size()));
    dd.addDiagnostic("pyramid_levels_used", "pyramid levels", static_cast<double>(pyramidCount));

    IR_LOG_INFO("Direct Fourier-Mellin angle=",
                best.candidate.angleDeg,
                " deg, scale=",
                best.candidate.scale,
                ", tx=",
                best.translation.x,
                ", ty=",
                best.translation.y,
                ", rot_scale_response=",
                best.candidate.response,
                ", translation_response=",
                best.translationResponse,
                ", candidates=",
                candidates.size());
    return true;
}

} // namespace ir
