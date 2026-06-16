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

struct RigidParams {
    double theta = 0.0;
    double tx = 0.0;
    double ty = 0.0;
};

struct SamplePoint {
    float x = 0.0f;
    float y = 0.0f;
    float value = 0.0f;
};

struct ResidualStats {
    int valid_samples = 0;
    double mse = -1.0;
};

struct LevelStats {
    int iterations = 0;
    bool converged = false;
    ResidualStats residual;
};

struct RigidDirectOptions {
    int max_iterations = 50;
    double epsilon = 1e-4;
    int pyramid_levels = 4;
    int blur_kernel = 5;
    double gradient_threshold = 1e-3;
    int sample_step = 2;
};

using LevelOptimizer = bool (*)(const cv::Mat& source,
                                const cv::Mat& target,
                                const std::vector<SamplePoint>& samples,
                                int maxIterations,
                                double epsilon,
                                RigidParams& params,
                                LevelStats& stats,
                                std::string& message);

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

inline void computeImageGradient(const cv::Mat& image, cv::Mat& gradX, cv::Mat& gradY) {
    cv::Sobel(image, gradX, CV_32F, 1, 0, 3, 1.0 / 8.0);
    cv::Sobel(image, gradY, CV_32F, 0, 1, 3, 1.0 / 8.0);
}

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

inline cv::Mat rigidMatrix(const RigidParams& params) {
    const double c = std::cos(params.theta);
    const double s = std::sin(params.theta);
    return (cv::Mat_<double>(2, 3) << c, -s, params.tx, s, c, params.ty);
}

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

    // 初始化位姿优化暂时关闭；后续统一设计前，所有刚体直接法都从零旋转/零平移开始。
    RigidParams fullParams;

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

