#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "direct/common/rigid_direct_common.h"

namespace ir::esm_rigid_common {

using rigid_direct_common::LevelStats;
using rigid_direct_common::RigidParams;
using rigid_direct_common::SamplePoint;

/// 功能：执行 ESM 刚体直接法的单层优化。
/// 作用：在共享金字塔调度之外，保留 ESM 专属的对称梯度 Jacobian 构造和参数更新逻辑。
inline bool optimizeRigidLevel(const cv::Mat& source,
                               const cv::Mat& target,
                               const std::vector<SamplePoint>& samples,
                               int maxIterations,
                               double epsilon,
                               RigidParams& params,
                               LevelStats& stats,
                               std::string& message) {
    stats = {};
    message.clear();
    if (samples.size() < rigid_direct_common::kMinSamples) {
        message = "not enough textured samples";
        return false;
    }

    cv::Mat srcGradX;
    cv::Mat srcGradY;
    cv::Mat dstGradX;
    cv::Mat dstGradY;
    rigid_direct_common::computeImageGradient(source, srcGradX, srcGradY);
    rigid_direct_common::computeImageGradient(target, dstGradX, dstGradY);

    const int iterations = std::max(1, maxIterations);
    const double eps = std::max(0.0, epsilon);

    for (int iter = 0; iter < iterations; ++iter) {
        const double c = std::cos(params.theta);
        const double s = std::sin(params.theta);

        cv::Matx33d h = cv::Matx33d::zeros();
        cv::Vec3d g(0.0, 0.0, 0.0);
        double sse = 0.0;
        int valid = 0;

        // 对每个采样点同时读取源图梯度和 warp 后目标梯度，再取对称平均构造 ESM Jacobian。
        for (const SamplePoint& sample : samples) {
            const double x = sample.x;
            const double y = sample.y;
            const double wx = c * x - s * y + params.tx;
            const double wy = s * x + c * y + params.ty;

            float targetValue = 0.0f;
            float srcIx = 0.0f;
            float srcIy = 0.0f;
            float dstIx = 0.0f;
            float dstIy = 0.0f;
            if (!rigid_direct_common::sampleBilinear(target, wx, wy, targetValue) ||
                !rigid_direct_common::sampleBilinear(srcGradX, x, y, srcIx) ||
                !rigid_direct_common::sampleBilinear(srcGradY, x, y, srcIy) ||
                !rigid_direct_common::sampleBilinear(dstGradX, wx, wy, dstIx) ||
                !rigid_direct_common::sampleBilinear(dstGradY, wx, wy, dstIy)) {
                continue;
            }

            const double residual = static_cast<double>(targetValue) - sample.value;
            const double avgIx = 0.5 * (static_cast<double>(srcIx) + static_cast<double>(dstIx));
            const double avgIy = 0.5 * (static_cast<double>(srcIy) + static_cast<double>(dstIy));
            const double dxdTheta = -s * x - c * y;
            const double dydTheta = c * x - s * y;
            const cv::Vec3d j(avgIx * dxdTheta + avgIy * dydTheta, avgIx, avgIy);

            for (int r = 0; r < 3; ++r) {
                g[r] += j[r] * residual;
                for (int col = 0; col < 3; ++col) {
                    h(r, col) += j[r] * j[col];
                }
            }

            sse += residual * residual;
            ++valid;
        }

        if (valid < rigid_direct_common::kMinSamples) {
            message = "not enough overlapping samples";
            return false;
        }

        cv::Vec3d delta;
        if (!rigid_direct_common::solveNormalEquation(h, g, delta)) {
            message = "normal equation is singular";
            return false;
        }

        // ESM 与 LK 一样在参数空间做加法更新；theta 归一化到 [-pi, pi]。
        params.theta += delta[0];
        params.tx += delta[1];
        params.ty += delta[2];
        if (params.theta > rigid_direct_common::kPi || params.theta < -rigid_direct_common::kPi) {
            params.theta = std::atan2(std::sin(params.theta), std::cos(params.theta));
        }

        stats.iterations = iter + 1;
        stats.residual.valid_samples = valid;
        stats.residual.mse = sse / static_cast<double>(valid);

        const double maxDelta =
            std::max({std::abs(delta[0]), std::abs(delta[1]), std::abs(delta[2])});
        if (maxDelta <= eps) {
            stats.converged = true;
            break;
        }
    }

    rigid_direct_common::computeResidualStats(target, samples, params, stats.residual);
    return true;
}

} // namespace ir::esm_rigid_common
