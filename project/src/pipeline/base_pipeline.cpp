#include "pipeline/base_pipeline.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/config.h"
#include "transform/affine_warper.h"
#include "transform/perspective_warper.h"
#include "utils/logger.h"
#include "utils/timer.h"

namespace fs = std::filesystem;

namespace ir {

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

// 为配准方法准备一对同步图像：显示用 BGR 图与计算用 8 位灰度图。
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

// 将图像转换为前景 mask，用于判断 warped source 与 target 是否真正重合。
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

/// 计算 warped 与 target 在 overlapMask 区域内的归一化平均绝对差 (NMAD)。
/// 返回值归一化到 [0, 1]；0 = 完全相同，1 = 完全不同。
/// 若 overlapMask 无前景像素则返回 -1.0。
double computePhotometricError(const cv::Mat& warped,
                                const cv::Mat& target,
                                const cv::Mat& overlapMask) {
    if (overlapMask.empty() || cv::countNonZero(overlapMask) == 0) {
        return -1.0;
    }

    cv::Mat warpedGray, targetGray;
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

    cv::Mat warpedFloat, targetFloat;
    warpedGray.convertTo(warpedFloat, CV_32F, 1.0 / 255.0);
    targetGray.convertTo(targetFloat, CV_32F, 1.0 / 255.0);

    cv::Mat diff;
    cv::absdiff(warpedFloat, targetFloat, diff);

    const cv::Scalar meanDiff = cv::mean(diff, overlapMask);
    return meanDiff[0];
}

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
        return (matrix.rows >= 2 && matrix.cols >= 3);
    }
    return false;
}

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

bool warpStructureMask(const RegistrationContext& ctx,
                       const cv::Mat& sourceMask,
                       const cv::Size& targetSize,
                       cv::Mat& warpedMask) {
    warpedMask.release();
    cv::Mat matrix;
    if (!activeTransformMatrix(ctx, matrix)) {
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
        cv::Mat affine = matrix(cv::Rect(0, 0, 3, 2)).clone();
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

bool toGray8ForVisualization(const cv::Mat& image, cv::Mat& gray8) {
    cv::Mat gray;
    if (!toGrayPreserveDepth(image, gray)) {
        return false;
    }
    return convertGrayTo8U(gray, gray8);
}

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

} // namespace

bool BasePipeline::configure(const PipelineConfig& cfg) {
    // 1. 保存配置，并清空上一轮创建的阶段组件。
    _config = cfg;
    _warper.reset();
    resetStages();

    // 2. 委托子类创建提取、关联和估计阶段组件。
    try {
        if (!configureStages(cfg)) {
            IR_LOG_ERROR(name(), "::configureStages returned false.");
            return false;
        }
    } catch (const std::exception& e) {
        IR_LOG_ERROR(name(), "::configure failed: ", e.what());
        return false;
    }

    // 3. 创建通用 warp 组件；默认使用 2x3 仿射 warper，透视 warper 作为扩展保留。
    _warper = std::make_shared<AffineWarper>();

    // 4. 加载评测指标（可选）。
    _evaluator.clear();
    if (!cfg.evaluator_path.empty()) {
        _evaluator.loadFromYaml(cfg.evaluator_path);
    }

    IR_LOG_INFO(name(), " configured.");
    return true;
}

bool BasePipeline::loadImages(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_load_ms);

    // 1. 检查输入路径是否完整且文件存在。
    if (ctx.image1_path.empty() || ctx.image2_path.empty()) {
        IR_LOG_ERROR("loadImages: one of the image paths is empty.");
        return false;
    }
    if (!fs::exists(ctx.image1_path) || !fs::exists(ctx.image2_path)) {
        IR_LOG_ERROR("loadImages: image not found. img1=",
                     ctx.image1_path.string(),
                     ", img2=",
                     ctx.image2_path.string());
        return false;
    }

    // 2. 读取图像，同时准备显示用 BGR 图和算法用 8 位灰度图。
    if (!loadImageForPipeline(ctx.image1_path, ctx.images.first, ctx.images.first_gray) ||
        !loadImageForPipeline(ctx.image2_path, ctx.images.second, ctx.images.second_gray)) {
        IR_LOG_ERROR("loadImages: cv::imread failed or image format is unsupported.");
        return false;
    }

    // 3. 记录输入尺寸，便于排查 warp 和 blend 的画布大小。
    IR_LOG_INFO("Loaded images: ",
                ctx.images.first.cols,
                "x",
                ctx.images.first.rows,
                " and ",
                ctx.images.second.cols,
                "x",
                ctx.images.second.rows);
    return true;
}

bool BasePipeline::runWarp(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_warp_ms);

    // 1. 若配置关闭 warp 或没有 warper，直接跳过。
    if (!_config.warp || !_warper) {
        return true;
    }

    // 2. 检查 runEstimation 是否写入了可用于图像重采样的几何模型。
    const auto t = ctx.geometry_data.type;
    if (t != GeometryType::HOMOGRAPHY && t != GeometryType::AFFINE && t != GeometryType::RIGID &&
        t != GeometryType::SIMILARITY) {
        IR_LOG_INFO("Warp skipped (", toString(t), " is not warpable).");
        return true;
    }
    if (!ctx.geometry_data.valid) {
        IR_LOG_WARN("Warp skipped: geometry estimation invalid.");
        return false;
    }

    // 3. HOMOGRAPHY 必须保留完整 3x3 透视项；仿射族继续走默认 2x3 warper。
    if (t == GeometryType::HOMOGRAPHY) {
        PerspectiveWarper perspectiveWarper;
        return perspectiveWarper.warp(ctx);
    }
    return _warper->warp(ctx);
}

bool BasePipeline::validateRegistrationQuality(RegistrationContext& ctx) {
    // 先重置可选验证结果。
    ctx.result.structure_overlap_iou = -1.0;

    // 顺序执行各项验证。
    if (!validateMatchQuality(ctx)) {
        return false;
    }
    if (!validateStructureOverlap(ctx)) {
        return false;
    }
    if (!validateWarpQuality(ctx)) {
        return false;
    }
    return validateMetricQuality(ctx);
}

bool BasePipeline::validateMatchQuality(RegistrationContext& ctx) {
    if (!_config.validate_match_quality) {
        return true;
    }

    // 统一读取结果对象里的匹配统计。
    const auto& r = ctx.result;
    IR_LOG_INFO("Match quality: inliers=",
                r.num_inliers,
                ", ratio=",
                r.inlier_ratio,
                ", reproj=",
                r.mean_reproj_error);

    // 条件1：最少内点数。
    if (_config.min_match_inliers > 0 && r.num_inliers < _config.min_match_inliers) {
        ctx.result.message = "match quality validation failed: inliers " +
                             std::to_string(r.num_inliers) + " < " +
                             std::to_string(_config.min_match_inliers);
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    // 条件2：最小内点率。
    if (_config.min_match_inlier_ratio >= 0.0 &&
        r.inlier_ratio < _config.min_match_inlier_ratio) {
        ctx.result.message = "match quality validation failed: inlier ratio " +
                             std::to_string(r.inlier_ratio) + " < " +
                             std::to_string(_config.min_match_inlier_ratio);
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    // 条件3：最大重投影误差。
    if (_config.max_match_reproj_error >= 0.0 &&
        r.mean_reproj_error > _config.max_match_reproj_error) {
        ctx.result.message = "match quality validation failed: reprojection error " +
                             std::to_string(r.mean_reproj_error) + " > " +
                             std::to_string(_config.max_match_reproj_error);
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    return true;
}

bool BasePipeline::validateStructureOverlap(RegistrationContext& ctx) {
    if (!_config.validate_structure_overlap) {
        return true;
    }

    // 条件1：必须有结构响应图。
    if (ctx.structure_data.first.response.empty() || ctx.structure_data.second.response.empty()) {
        ctx.result.message = "structure overlap validation failed: missing structure response";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    cv::Mat sourceMask;
    cv::Mat targetMask;
    const int thresholdValue = _config.structure_overlap_foreground_threshold;

    // 条件2：结构响应图要能转成前景 mask。
    if (!buildForegroundMask(ctx.structure_data.first.response, thresholdValue, sourceMask) ||
        !buildForegroundMask(ctx.structure_data.second.response, thresholdValue, targetMask)) {
        ctx.result.message = "structure overlap validation failed: cannot build masks";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    dilateMaskIfRequested(sourceMask, _config.structure_overlap_dilate_size);
    dilateMaskIfRequested(targetMask, _config.structure_overlap_dilate_size);

    cv::Mat warpedSourceMask;

    // 条件3：source mask 要能 warp 到 target 坐标系。
    if (!warpStructureMask(ctx, sourceMask, targetMask.size(), warpedSourceMask)) {
        ctx.result.message = "structure overlap validation failed: cannot warp source mask";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    const double iou = computeMaskIou(warpedSourceMask, targetMask);
    ctx.result.structure_overlap_iou = iou;
    if (iou < 0.0) {
        ctx.result.message = "structure overlap IoU failed: empty structure union";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    // 条件4：结构重叠 IoU 要达到阈值。
    IR_LOG_INFO("Structure overlap IoU=", iou, ", min=", _config.min_structure_overlap_iou);
    if (iou < _config.min_structure_overlap_iou) {
        ctx.result.message = "structure overlap IoU below threshold: " +
                             std::to_string(iou) + " < " +
                             std::to_string(_config.min_structure_overlap_iou);
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    return true;
}

bool BasePipeline::validateMetricQuality(RegistrationContext& ctx) {
    if (!_config.validate_metric_quality) {
        return true;
    }

    // 统一读取 evaluator 产出的指标。
    auto findRequiredMetric = [&](const std::string& name) -> const MetricResult* {
        const MetricResult* metric = ctx.evaluation.find(name);
        if (!metric || !metric->valid) {
            ctx.result.message = "metric quality validation failed: missing metric " + name;
            IR_LOG_WARN(ctx.result.message);
            return nullptr;
        }
        return metric;
    };

    // 条件1：PSNR 不低于阈值。
    if (_config.min_metric_psnr >= 0.0) {
        const MetricResult* metric = findRequiredMetric("PSNR");
        if (!metric) {
            return false;
        }
        IR_LOG_INFO("Metric quality PSNR=", metric->value, ", min=", _config.min_metric_psnr);
        if (metric->value < _config.min_metric_psnr) {
            ctx.result.message = "metric quality validation failed: PSNR " +
                                 std::to_string(metric->value) + " < " +
                                 std::to_string(_config.min_metric_psnr);
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
    }

    // 条件2：SSIM 不低于阈值。
    if (_config.min_metric_ssim >= 0.0) {
        const MetricResult* metric = findRequiredMetric("SSIM");
        if (!metric) {
            return false;
        }
        IR_LOG_INFO("Metric quality SSIM=", metric->value, ", min=", _config.min_metric_ssim);
        if (metric->value < _config.min_metric_ssim) {
            ctx.result.message = "metric quality validation failed: SSIM " +
                                 std::to_string(metric->value) + " < " +
                                 std::to_string(_config.min_metric_ssim);
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
    }

    // 条件3：RMSE 不高于阈值。
    if (_config.max_metric_rmse >= 0.0) {
        const MetricResult* metric = findRequiredMetric("RMSE");
        if (!metric) {
            return false;
        }
        IR_LOG_INFO("Metric quality RMSE=", metric->value, ", max=", _config.max_metric_rmse);
        if (metric->value > _config.max_metric_rmse) {
            ctx.result.message = "metric quality validation failed: RMSE " +
                                 std::to_string(metric->value) + " > " +
                                 std::to_string(_config.max_metric_rmse);
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
    }

    return true;
}

bool BasePipeline::validateWarpQuality(RegistrationContext& ctx) {
    ctx.result.warp_overlap_iou = -1.0;
    ctx.result.warp_photometric_error = -1.0;

    if (!_config.validate_warp_overlap && !_config.validate_warp_photometric) {
        return true;
    }

    // 条件1：warp 结果必须存在。
    if (!_config.warp || ctx.warped_image.empty()) {
        ctx.result.message = "warp validation failed: warped image is empty";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    // 条件2：warp 和 target 尺寸必须一致。
    if (ctx.images.second.empty() || ctx.warped_image.size() != ctx.images.second.size()) {
        ctx.result.message =
            "warp validation failed: warped image and target have different sizes";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    // 条件3：两张图都要能提取前景 mask。
    cv::Mat warpedMask;
    cv::Mat targetMask;
    const int thresholdValue = _config.warp_overlap_foreground_threshold;
    if (!buildForegroundMask(ctx.warped_image, thresholdValue, warpedMask) ||
        !buildForegroundMask(ctx.images.second, thresholdValue, targetMask)) {
        ctx.result.message = "warp validation failed: cannot build foreground masks";
        IR_LOG_WARN(ctx.result.message);
        return false;
    }

    // 条件4：前景 IoU 要达到阈值。
    if (_config.validate_warp_overlap) {
        const double iou = computeMaskIou(warpedMask, targetMask);
        ctx.result.warp_overlap_iou = iou;
        if (iou < 0.0) {
            ctx.result.message = "warp overlap IoU failed: empty foreground union";
            IR_LOG_WARN(ctx.result.message);
            return false;
        }

        IR_LOG_INFO("Warp overlap IoU=", iou, ", min=", _config.min_warp_overlap_iou);
        if (iou < _config.min_warp_overlap_iou) {
            ctx.result.message = "warp overlap IoU below threshold: " + std::to_string(iou) +
                                 " < " + std::to_string(_config.min_warp_overlap_iou);
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
    }

    // 条件5：重叠区域光度误差不能过大。
    if (_config.validate_warp_photometric) {
        cv::Mat overlapMask;
        cv::bitwise_and(warpedMask, targetMask, overlapMask);
        const double nmad =
            computePhotometricError(ctx.warped_image, ctx.images.second, overlapMask);
        ctx.result.warp_photometric_error = nmad;
        if (nmad < 0.0) {
            ctx.result.message = "warp photometric validation failed: empty overlap";
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
        IR_LOG_INFO("Warp photometric NMAD=",
                    nmad,
                    ", max=",
                    _config.max_warp_photometric_error);
        if (nmad > _config.max_warp_photometric_error) {
            ctx.result.message = "warp photometric error above threshold: " +
                                 std::to_string(nmad) +
                                 " > " + std::to_string(_config.max_warp_photometric_error);
            IR_LOG_WARN(ctx.result.message);
            return false;
        }
    }

    return true;
}

std::string BasePipeline::buildOutputStem(const RegistrationContext& ctx) const {
    return ctx.image1_path.stem().string() + "_" + ctx.image2_path.stem().string() + "_" + name();
}

bool BasePipeline::saveOutputs(RegistrationContext& ctx) {
    if (_config.output_dir.empty()) {
        return true;
    }

    // 1. 创建通用输出目录。
    const fs::path originals_dir = _config.output_dir / "originals";
    const fs::path warped_dir = _config.output_dir / "warped";
    const fs::path blend_dir = _config.output_dir / "blend";
    const fs::path false_color_overlay_dir = _config.output_dir / "false_color_overlay";
    std::error_code ec;
    fs::create_directories(originals_dir, ec);
    fs::create_directories(warped_dir, ec);
    fs::create_directories(blend_dir, ec);
    fs::create_directories(false_color_overlay_dir, ec);

    const std::string stem = buildOutputStem(ctx);
    const std::string sampleStem = ctx.image1_path.stem().string() + "_" +
                                   ctx.image2_path.stem().string();

    // 2. 保存原始 source / target，便于和 warped / blend 对照。
    if (!ctx.images.first.empty()) {
        const fs::path out = originals_dir / (sampleStem + "_source_original.png");
        if (cv::imwrite(out.string(), ctx.images.first)) {
            IR_LOG_INFO("Wrote source original image: ", out.string());
        } else {
            IR_LOG_WARN("Failed to write source original image: ", out.string());
        }
    }
    if (!ctx.images.second.empty()) {
        const fs::path out = originals_dir / (sampleStem + "_target_original.png");
        if (cv::imwrite(out.string(), ctx.images.second)) {
            IR_LOG_INFO("Wrote target original image: ", out.string());
        } else {
            IR_LOG_WARN("Failed to write target original image: ", out.string());
        }
    }

    // 3. 保存 warped source，并和 target 按同尺寸画布生成 blend。
    if (_config.warp && !ctx.warped_image.empty()) {
        const fs::path out = warped_dir / (stem + "_warped.png");
        cv::imwrite(out.string(), ctx.warped_image);
        IR_LOG_INFO("Wrote warped image: ", out.string());

        if (ctx.warped_image.size() == ctx.images.second.size() &&
            ctx.warped_image.type() == ctx.images.second.type()) {
            cv::Mat blend;
            cv::addWeighted(ctx.warped_image, 0.5, ctx.images.second, 0.5, 0.0, blend);
            const fs::path blend_out = blend_dir / (stem + "_blend.png");
            cv::imwrite(blend_out.string(), blend);
            IR_LOG_INFO("Wrote blend image: ", blend_out.string());
        }

        if (ctx.warped_image.size() == ctx.images.second.size()) {
            cv::Mat falseColorOverlay;
            if (buildFalseColorOverlay(ctx.warped_image,
                                       ctx.images.second,
                                       _config.warp_overlap_foreground_threshold,
                                       falseColorOverlay)) {
                const fs::path false_color_out =
                    false_color_overlay_dir / (stem + "_false_color_overlay.png");
                cv::imwrite(false_color_out.string(), falseColorOverlay);
                IR_LOG_INFO("Wrote false-color overlay image: ", false_color_out.string());
            } else {
                IR_LOG_WARN("Failed to build false-color overlay image for: ", stem);
            }
        }
    }

    return true;
}

bool BasePipeline::showWindows(RegistrationContext& ctx) {
    bool shown = false;

    // 1. 按配置显示 source 窗口。
    if (_config.show_source_window && !ctx.images.first.empty()) {
        cv::imshow("Source Image", ctx.images.first);
        shown = true;
    }
    // 2. 按配置显示 target 窗口。
    if (_config.show_target_window && !ctx.images.second.empty()) {
        cv::imshow("Target Image", ctx.images.second);
        shown = true;
    }
    // 3. 按配置显示 warped source 窗口。
    if (_config.show_warped_window) {
        if (!ctx.warped_image.empty()) {
            cv::imshow("Warped Image", ctx.warped_image);
            shown = true;
        } else {
            IR_LOG_WARN("show_warped_window is enabled, but warped_image is empty.");
        }
    }

    if (shown) {
        const int wait = (_config.wait_key < 0) ? 0 : _config.wait_key;
        IR_LOG_INFO("Displaying visualization windows; waitKey=", wait);
        cv::waitKey(wait);
    }

    return true;
}

bool BasePipeline::run(RegistrationContext& ctx) {
    Timer total;

    // 1. 初始化本次运行上下文，并写入输入输出路径。
    ctx.reset();
    ctx.image1_path = _config.image1_path;
    ctx.image2_path = _config.image2_path;
    ctx.output_dir = _config.output_dir;

    auto fail = [&](const std::string& msg) {
        ctx.result.success = false;
        ctx.result.message = msg;
        ctx.result.t_total_ms = total.elapsedMs();
        IR_LOG_ERROR("Pipeline failed: ", msg);
        saveOutputs(ctx);
        return false;
    };

    // 2. 依次执行公共流程：读图、提取、关联、估计、warp 和输出。
    if (!loadImages(ctx)) {
        return fail("load failed");
    }

    if (!runExtraction(ctx)) {
        return fail("extract failed");
    }

    if (!runAssociation(ctx)) {
        std::string detail = "associate failed";
        if (!ctx.structure_match_data.message.empty()) {
            detail += ": " + ctx.structure_match_data.message;
        }
        return fail(detail);
    }

    if (!runEstimation(ctx)) {
        const std::string detail =
            ctx.geometry_data.message.empty()
                ? std::string("estimation failed")
                : std::string("estimation failed: ") + ctx.geometry_data.message;
        return fail(detail);
    }

    if (!runWarp(ctx)) {
        return fail(ctx.result.message.empty() ? "warp failed" : ctx.result.message);
    }
    if (_config.warp && ctx.warped_image.empty()) {
        return fail("warp failed: warped image is empty");
    }
    // 运行评测指标（仅成功时计算）
    if (!_evaluator.metrics().empty()) {
        Sample dummySample;
        _evaluator.evaluate(ctx, dummySample);
    }
    if (!validateRegistrationQuality(ctx)) {
        return fail(ctx.result.message.empty() ? "registration validation failed"
                                              : ctx.result.message);
    }

    saveOutputs(ctx);
    // 3. 所有阶段完成后记录总耗时和成功状态。
    ctx.result.success = true;
    ctx.result.t_total_ms = total.elapsedMs();
    ctx.result.message = "OK";
    return true;
}

} // namespace ir
