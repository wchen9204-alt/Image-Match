#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "core/context.h"
#include "core/types.h"
#include "utils/image_utils.h"
#include "utils/logger.h"

namespace ir {
namespace rigid_direct_common {

constexpr int kMinSamples = 12;
constexpr double kPi = 3.14159265358979323846;

/// 功能：表示刚体直接法优化的 3 个自由度参数。
struct RigidParams {
    double theta = 0.0;
    double tx = 0.0;
    double ty = 0.0;
};

/// 功能：保存源图上的一个采样点及其光度值。
struct SamplePoint {
    float x = 0.0f;
    float y = 0.0f;
    float value = 0.0f;
};

/// 功能：统计当前参数下的有效采样数和均方残差。
struct ResidualStats {
    int valid_samples = 0;
    double mse = -1.0;
};

/// 功能：记录单层金字塔优化的迭代结果。
struct LevelStats {
    int iterations = 0;
    bool converged = false;
    ResidualStats residual;
};

/// 功能：汇总刚体直接法共用优化配置。
struct RigidDirectOptions {
    int max_iterations = 50;
    double epsilon = 1e-4;
    int pyramid_levels = 4;
    int blur_kernel = 5;
    double gradient_threshold = 1e-3;
    int sample_step = 2;
};

/// 功能：抽象单层优化器接口，允许 ESM/LK 等方法复用同一套金字塔调度逻辑。
using LevelOptimizer = bool (*)(const cv::Mat& source,
                                const cv::Mat& target,
                                const std::vector<SamplePoint>& samples,
                                int maxIterations,
                                double epsilon,
                                RigidParams& params,
                                LevelStats& stats,
                                std::string& message);

/// 功能：从原图构建由粗到细的金字塔。
/// 作用：为刚体直接法提供多尺度初值传播，扩大初始位姿的可收敛范围。
inline std::vector<cv::Mat> buildPyramid(const cv::Mat& base, int requestedLevels) {
    std::vector<cv::Mat> pyramid;
    if (base.empty()) {
        return pyramid;
    }

    pyramid.push_back(base);
    const int levels = std::max(1, requestedLevels);
    for (int i = 1; i < levels; ++i) {
        const cv::Mat& prev = pyramid.back();
        if (prev.cols < 16 || prev.rows < 16) {
            break;
        }

        cv::Mat next;
        // 每层使用 pyrDown 逐步降采样，为粗到细刚体优化提供稳定初值。
        cv::pyrDown(prev, next);
        if (next.empty() || next.size() == prev.size()) {
            break;
        }
        pyramid.push_back(next);
    }
    return pyramid;
}

/// 功能：在单通道浮点图上做双线性插值采样。
/// 作用：把连续坐标上的 warp 结果映射成可比较的目标灰度值。
inline bool sampleBilinear(const cv::Mat& image, double x, double y, float& value) {
    if (image.empty() || image.type() != CV_32F || x < 0.0 || y < 0.0 ||
        x >= static_cast<double>(image.cols - 1) || y >= static_cast<double>(image.rows - 1)) {
        return false;
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const double ax = x - x0;
    const double ay = y - y0;

    const float* row0 = image.ptr<float>(y0);
    const float* row1 = image.ptr<float>(y0 + 1);
    const double v00 = row0[x0];
    const double v01 = row0[x0 + 1];
    const double v10 = row1[x0];
    const double v11 = row1[x0 + 1];

    value = static_cast<float>((1.0 - ax) * (1.0 - ay) * v00 + ax * (1.0 - ay) * v01 +
                               (1.0 - ax) * ay * v10 + ax * ay * v11);
    return true;
}

/// 功能：计算图像在 x/y 方向上的一阶梯度。
/// 作用：为采样点筛选和直接法雅可比构建提供局部光度变化信息。
inline void computeImageGradient(const cv::Mat& image, cv::Mat& gradX, cv::Mat& gradY) {
    cv::Sobel(image, gradX, CV_32F, 1, 0, 3, 1.0 / 8.0);
    cv::Sobel(image, gradY, CV_32F, 0, 1, 3, 1.0 / 8.0);
}

/// 功能：从源图中挑选可用于直接法优化的采样点。
/// 作用：跳过低梯度平坦区域，优先保留对位姿更新更敏感的像素。
inline std::vector<SamplePoint>
buildSamples(const cv::Mat& source, int sampleStep, double gradientThreshold) {
    std::vector<SamplePoint> samples;
    if (source.empty() || source.type() != CV_32F || source.cols < 3 || source.rows < 3) {
        return samples;
    }

    // 先计算源图梯度，后续只在信息量足够的像素上建立残差约束。
    cv::Mat gradX;
    cv::Mat gradY;
    computeImageGradient(source, gradX, gradY);

    const int step = std::max(1, sampleStep);
    const double threshold2 = std::max(0.0, gradientThreshold) *
                              std::max(0.0, gradientThreshold);
    const bool useThreshold = threshold2 > 0.0;
    samples.reserve(static_cast<size_t>((source.rows / step + 1) * (source.cols / step + 1)));

    for (int y = 1; y < source.rows - 1; y += step) {
        const float* srcRow = source.ptr<float>(y);
        const float* gxRow = gradX.ptr<float>(y);
        const float* gyRow = gradY.ptr<float>(y);
        for (int x = 1; x < source.cols - 1; x += step) {
            const double gx = gxRow[x];
            const double gy = gyRow[x];
            // 梯度阈值用于跳过平坦区域，避免无信息样本拖慢优化并削弱稳定性。
            if (useThreshold && gx * gx + gy * gy < threshold2) {
                continue;
            }
            samples.push_back(
                SamplePoint{static_cast<float>(x), static_cast<float>(y), srcRow[x]});
        }
    }
    return samples;
}

/// 功能：在给定刚体参数下统计目标图上的光度残差。
/// 作用：为最终结果验收和各层优化状态记录提供统一误差度量。
inline bool computeResidualStats(const cv::Mat& target,
                                 const std::vector<SamplePoint>& samples,
                                 const RigidParams& params,
                                 ResidualStats& stats) {
    stats = {};
    if (target.empty() || samples.empty()) {
        return false;
    }

    const double c = std::cos(params.theta);
    const double s = std::sin(params.theta);
    double sse = 0.0;
    int valid = 0;

    for (const SamplePoint& sample : samples) {
        // 先用当前刚体参数把源图采样点 warp 到目标图坐标系。
        const double wx = c * sample.x - s * sample.y + params.tx;
        const double wy = s * sample.x + c * sample.y + params.ty;

        float targetValue = 0.0f;
        if (!sampleBilinear(target, wx, wy, targetValue)) {
            continue;
        }

        // 以目标灰度减源灰度作为残差，累计最终光度误差统计。
        const double residual = static_cast<double>(targetValue) - sample.value;
        sse += residual * residual;
        ++valid;
    }

    if (valid < kMinSamples) {
        return false;
    }

    stats.valid_samples = valid;
    stats.mse = sse / static_cast<double>(valid);
    return true;
}

/// 功能：求解 3x3 法方程的位姿增量。
/// 作用：优先用 Cholesky 保持效率，退化时回退到 SVD 提高可解性。
inline bool solveNormalEquation(const cv::Matx33d& h,
                                const cv::Vec3d& g,
                                cv::Vec3d& delta) {
    const double diagSum = std::abs(h(0, 0)) + std::abs(h(1, 1)) + std::abs(h(2, 2));
    if (diagSum < 1e-12) {
        return false;
    }

    cv::Mat hMat = (cv::Mat_<double>(3, 3) << h(0, 0),
                    h(0, 1),
                    h(0, 2),
                    h(1, 0),
                    h(1, 1),
                    h(1, 2),
                    h(2, 0),
                    h(2, 1),
                    h(2, 2));
    cv::Mat rhs = (cv::Mat_<double>(3, 1) << -g[0], -g[1], -g[2]);
    cv::Mat deltaMat;
    // 优先尝试 Cholesky；若矩阵条件较差，则回退到 SVD 保证可解性。
    if (!cv::solve(hMat, rhs, deltaMat, cv::DECOMP_CHOLESKY) &&
        !cv::solve(hMat, rhs, deltaMat, cv::DECOMP_SVD)) {
        return false;
    }

    delta = cv::Vec3d(deltaMat.at<double>(0), deltaMat.at<double>(1), deltaMat.at<double>(2));
    return std::isfinite(delta[0]) && std::isfinite(delta[1]) && std::isfinite(delta[2]);
}

/// 功能：把刚体参数打包成 2x3 仿射矩阵。
/// 作用：统一下游 warp、验证和输出阶段读取的几何表示。
inline cv::Mat rigidMatrix(const RigidParams& params) {
    const double c = std::cos(params.theta);
    const double s = std::sin(params.theta);
    return (cv::Mat_<double>(2, 3) << c, -s, params.tx, s, c, params.ty);
}

/// 功能：从仿射初值中提取刚体参数。
/// 作用：把点特征粗估给出的 similarity/rigid 结果转换成直接法可继续优化的初始位姿。
inline bool rigidParamsFromAffine(const cv::Mat& affine, RigidParams& params) {
    if (affine.empty() || affine.rows < 2 || affine.cols < 3) {
        return false;
    }

    // 初值可能来自 similarity；这里提取旋转和平移，避免把缩放带入刚体优化。
    cv::Mat a64;
    affine.convertTo(a64, CV_64F);
    const double a00 = a64.at<double>(0, 0);
    const double a10 = a64.at<double>(1, 0);
    const double scale = std::sqrt(a00 * a00 + a10 * a10);
    if (!std::isfinite(scale) || scale <= 1e-9) {
        return false;
    }

    params.theta = std::atan2(a10 / scale, a00 / scale);
    params.tx = a64.at<double>(0, 2);
    params.ty = a64.at<double>(1, 2);
    return std::isfinite(params.theta) && std::isfinite(params.tx) && std::isfinite(params.ty);
}

/// 功能：从上下文中的点特征初始化结果读取刚体初值。
/// 作用：仅在粗估已通过采用条件时，为直接法提供更稳定的起点。
inline bool rigidParamsFromContextInitializer(const RegistrationContext& ctx,
                                              RigidParams& params) {
    const auto& init = ctx.feature_initializer_data;
    if (!init.accepted || init.A.empty()) {
        return false;
    }
    return rigidParamsFromAffine(init.A, params);
}

/// 功能：执行刚体直接法公共主流程。
/// 作用：统一完成灰度预处理、金字塔调度、初值注入、逐层优化和结果写回。
inline bool runRigidAlignment(RegistrationContext& ctx,
                              const std::string& methodName,
                              const std::string& displayName,
                              const RigidDirectOptions& options,
                              LevelOptimizer optimizeLevel) {
    auto& dd = ctx.direct_data;
    auto& gd = ctx.geometry_data;

    dd.clear();
    gd.clear();
    dd.method = methodName;

    // 1. 读取并预处理灰度图，作为共享刚体直接法的优化输入。
    cv::Mat source;
    cv::Mat target;
    if (!image_utils::convertGrayToFloat01(ctx.images.first_gray, source, options.blur_kernel) ||
        !image_utils::convertGrayToFloat01(ctx.images.second_gray, target, options.blur_kernel)) {
        dd.message = displayName + " requires non-empty grayscale images";
        gd.message = dd.message;
        return false;
    }

    if (source.size() != target.size()) {
        dd.message = displayName + " requires images with the same size";
        gd.message = dd.message;
        return false;
    }

    // 2. 构建从粗到细的金字塔，为共享刚体优化器提供多尺度初值传播。
    std::vector<cv::Mat> sourcePyr = buildPyramid(source, options.pyramid_levels);
    std::vector<cv::Mat> targetPyr = buildPyramid(target, options.pyramid_levels);
    const int levels = static_cast<int>(std::min(sourcePyr.size(), targetPyr.size()));
    if (levels <= 0) {
        dd.message = displayName + " failed to build image pyramid";
        gd.message = dd.message;
        return false;
    }

    // 可选使用点特征安全门通过后的初始位姿；失败时保持零旋转、零平移起步。
    RigidParams fullParams;
    if (rigidParamsFromContextInitializer(ctx, fullParams)) {
        dd.addDiagnostic("feature_initializer_rigid_seeded",
                         "rigid seeded by feature init",
                         1.0);
        IR_LOG_INFO(displayName,
                    " initialized from feature initializer: theta=",
                    fullParams.theta * 180.0 / kPi,
                    " deg, tx=",
                    fullParams.tx,
                    ", ty=",
                    fullParams.ty);
    }

    // 3. 逐层采样高梯度像素并调用具体优化器，在金字塔层之间传播刚体参数。
    LevelStats lastStats;
    for (int level = levels - 1; level >= 0; --level) {
        const cv::Mat& levelSource = sourcePyr[level];
        const cv::Mat& levelTarget = targetPyr[level];
        const double sx =
            static_cast<double>(levelSource.cols) / static_cast<double>(sourcePyr[0].cols);
        const double sy =
            static_cast<double>(levelSource.rows) / static_cast<double>(sourcePyr[0].rows);

        RigidParams levelParams;
        // 把全分辨率参数投影到当前层，作为本层优化初值。
        levelParams.theta = fullParams.theta;
        levelParams.tx = fullParams.tx * sx;
        levelParams.ty = fullParams.ty * sy;

        const std::vector<SamplePoint> samples =
            buildSamples(levelSource, options.sample_step, options.gradient_threshold);
        std::string levelMessage;
        if (!optimizeLevel(levelSource,
                           levelTarget,
                           samples,
                           options.max_iterations,
                           options.epsilon,
                           levelParams,
                           lastStats,
                           levelMessage)) {
            dd.message = displayName + " failed at pyramid level " + std::to_string(level) + ": " +
                         levelMessage;
            gd.message = dd.message;
            IR_LOG_WARN(displayName, " rejected: ", dd.message);
            return false;
        }

        // 将本层优化结果映射回原图尺度，供下一层继续细化。
        fullParams.theta = levelParams.theta;
        fullParams.tx = levelParams.tx / sx;
        fullParams.ty = levelParams.ty / sy;
    }

    // 4. 回到原图层评估最终光度残差，并据此生成统一的刚体矩阵结果。
    const std::vector<SamplePoint> finalSamples =
        buildSamples(sourcePyr[0], options.sample_step, options.gradient_threshold);
    ResidualStats finalStats;
    if (!computeResidualStats(targetPyr[0], finalSamples, fullParams, finalStats)) {
        dd.message = displayName + " failed to evaluate final photometric error";
        gd.message = dd.message;
        return false;
    }

    const cv::Mat A = rigidMatrix(fullParams);
    gd.type = GeometryType::RIGID;
    gd.A = A.clone();
    gd.valid = true;
    gd.num_inliers = finalStats.valid_samples;
    gd.inlier_ratio = 1.0 / (1.0 + std::max(0.0, finalStats.mse));
    gd.correspondence_source = "DIRECT";
    gd.num_correspondences = finalStats.valid_samples;

    dd.A = gd.A.clone();
    dd.valid = true;
    dd.score = gd.inlier_ratio;
    dd.photometric_error = finalStats.mse;

    IR_LOG_INFO(displayName,
                " estimated RIGID theta=",
                fullParams.theta * 180.0 / kPi,
                " deg, tx=",
                fullParams.tx,
                ", ty=",
                fullParams.ty,
                ", mse=",
                finalStats.mse,
                ", samples=",
                finalStats.valid_samples);
    return true;
}

} // namespace rigid_direct_common
} // namespace ir
