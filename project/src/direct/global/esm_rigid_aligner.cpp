#include "direct/global/esm_rigid_aligner.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "direct/global/rigid_direct_common.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

using rigid_direct_common::LevelStats;
using rigid_direct_common::RigidDirectOptions;
using rigid_direct_common::RigidParams;
using rigid_direct_common::SamplePoint;

/// ESM 在每层使用源图与目标图梯度的对称平均构造 Jacobian，
/// 以减轻纯前向 LK 在较大残差阶段对单侧梯度近似的偏置。
bool optimizeRigidLevelEsm(const cv::Mat& source,
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

} // namespace

EsmRigidAligner::EsmRigidAligner(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    // 兼容算法配置文件中直接写参数或写在 params 节点下两种形式。
    _maxIterations = yaml_utils::getInt(params, "max_iterations", 50);
    _epsilon = yaml_utils::getDouble(params, "epsilon", 1e-4);
    _pyramidLevels = yaml_utils::getInt(params, "pyramid_levels", 4);
    _blurKernel = yaml_utils::getInt(params, "blur_kernel", 5);
    _gradientThreshold = yaml_utils::getDouble(params, "gradient_threshold", 1e-3);
    _sampleStep = yaml_utils::getInt(params, "sample_step", 2);
}

bool EsmRigidAligner::align(RegistrationContext& ctx) {
    // 1. 组装 ESM 刚体直接法的共享优化参数，具体多层迭代流程复用 rigid_direct_common。
    RigidDirectOptions options;
    options.max_iterations = _maxIterations;
    options.epsilon = _epsilon;
    options.pyramid_levels = _pyramidLevels;
    options.blur_kernel = _blurKernel;
    options.gradient_threshold = _gradientThreshold;
    options.sample_step = _sampleStep;
    // 2. 共享流程负责预处理、金字塔逐层优化、残差评估和结果写回。
    return rigid_direct_common::runRigidAlignment(
        ctx, name(), "ESM rigid", options, optimizeRigidLevelEsm);
}

} // namespace ir
