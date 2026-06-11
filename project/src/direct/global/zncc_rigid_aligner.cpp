#include "direct/global/zncc_rigid_aligner.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "direct/global/rigid_direct_common.h"
#include "utils/image_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

using rigid_direct_common::LevelStats;
using rigid_direct_common::RigidDirectOptions;
using rigid_direct_common::RigidParams;
using rigid_direct_common::SamplePoint;

constexpr double kZnccStdEps = 1e-6;
constexpr double kObjectiveAcceptEps = 1e-8;
constexpr double kLmLambdaInit = 1e-3;
constexpr double kLmLambdaFloor = 1e-6;
constexpr double kLmLambdaGrow = 8.0;
constexpr double kLmLambdaShrink = 0.5;
constexpr double kLmLambdaMax = 1e8;
constexpr double kGradientConvergedEps = 1e-6;
constexpr int kLmMaxTrials = 12;
constexpr double kBacktrackingScales[] = {
    1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125};

// 保存一次 ZNCC 线性化后的近似二阶项、梯度项和当前目标值，
// 便于主优化循环直接做阻尼 Gauss-Newton / LM 更新。
struct ZnccLinearization {
    cv::Matx33d h = cv::Matx33d::zeros();
    cv::Vec3d g = cv::Vec3d(0.0, 0.0, 0.0);
    double objective = std::numeric_limits<double>::infinity();
    int valid = 0;
};

// 功能：从 2x3 刚体矩阵中提取旋转角和平移量。
// 作用：把公共流程输出的矩阵表示转换成优化阶段更方便使用的参数表示。
RigidParams paramsFromRigidMatrix(const cv::Mat& A) {
    RigidParams params;
    if (A.empty() || A.rows < 2 || A.cols < 3) {
        return params;
    }

    const double a00 = A.at<double>(0, 0);
    const double a10 = A.at<double>(1, 0);
    params.theta = std::atan2(a10, a00);
    params.tx = A.at<double>(0, 2);
    params.ty = A.at<double>(1, 2);
    return params;
}

// 功能：把最小化得到的归一化残差目标值映射成 ZNCC 相似度分数。
double znccScoreFromObjective(double objective) {
    return std::clamp(1.0 - 0.5 * objective, -1.0, 1.0);
}

// 功能：评估当前刚体参数下的 ZNCC 目标值。
// 步骤：
// 1. 将 source 采样点 warp 到 target；
// 2. 收集重叠区域的 source/target 灰度对；
// 3. 计算均值、方差并做零均值归一化；
// 4. 输出归一化残差的平均平方和。
bool evaluateZnccObjective(const cv::Mat& target,
                           const std::vector<SamplePoint>& samples,
                           const RigidParams& params,
                           double& objective,
                           int& valid,
                           std::string& message) {
    objective = std::numeric_limits<double>::infinity();
    valid = 0;
    message.clear();

    std::vector<float> sourceValues;
    std::vector<float> targetValues;
    sourceValues.reserve(samples.size());
    targetValues.reserve(samples.size());

    const double c = std::cos(params.theta);
    const double s = std::sin(params.theta);

    // 只保留能够成功落入 target 内部的重叠样本。
    for (const SamplePoint& sample : samples) {
        const double wx = c * sample.x - s * sample.y + params.tx;
        const double wy = s * sample.x + c * sample.y + params.ty;

        float targetValue = 0.0f;
        if (!rigid_direct_common::sampleBilinear(target, wx, wy, targetValue)) {
            continue;
        }

        sourceValues.push_back(sample.value);
        targetValues.push_back(targetValue);
    }

    valid = static_cast<int>(targetValues.size());
    if (valid < rigid_direct_common::kMinSamples) {
        message = "not enough overlapping samples";
        return false;
    }

    double meanSource = 0.0;
    double meanTarget = 0.0;
    for (int i = 0; i < valid; ++i) {
        meanSource += sourceValues[static_cast<size_t>(i)];
        meanTarget += targetValues[static_cast<size_t>(i)];
    }
    meanSource /= static_cast<double>(valid);
    meanTarget /= static_cast<double>(valid);

    double varSource = 0.0;
    double varTarget = 0.0;
    for (int i = 0; i < valid; ++i) {
        const double ds = static_cast<double>(sourceValues[static_cast<size_t>(i)]) - meanSource;
        const double dt = static_cast<double>(targetValues[static_cast<size_t>(i)]) - meanTarget;
        varSource += ds * ds;
        varTarget += dt * dt;
    }

    const double sigmaSource =
        std::sqrt(varSource / static_cast<double>(valid) + kZnccStdEps);
    const double sigmaTarget =
        std::sqrt(varTarget / static_cast<double>(valid) + kZnccStdEps);
    if (sigmaSource <= kZnccStdEps || sigmaTarget <= kZnccStdEps) {
        message = "low-variance normalized intensities";
        return false;
    }

    double sse = 0.0;
    for (int i = 0; i < valid; ++i) {
        const double normSource =
            (static_cast<double>(sourceValues[static_cast<size_t>(i)]) - meanSource) / sigmaSource;
        const double normTarget =
            (static_cast<double>(targetValues[static_cast<size_t>(i)]) - meanTarget) / sigmaTarget;
        const double residual = normTarget - normSource;
        sse += residual * residual;
    }

    objective = sse / static_cast<double>(valid);
    return true;
}

// 功能：构造 ZNCC 目标的一阶线性化结果。
// 步骤：
// 1. 在当前参数下采样 target 灰度和梯度；
// 2. 计算 target 灰度关于 theta / tx / ty 的原始 Jacobian；
// 3. 把原始 Jacobian 转成归一化残差 Jacobian；
// 4. 累加近似 Hessian、梯度和当前目标值。
bool buildZnccLinearization(const cv::Mat& target,
                            const cv::Mat& gradX,
                            const cv::Mat& gradY,
                            const std::vector<SamplePoint>& samples,
                            const RigidParams& params,
                            ZnccLinearization& linearization,
                            std::string& message) {
    linearization = {};
    message.clear();

    std::vector<float> sourceValues;
    std::vector<float> targetValues;
    std::vector<cv::Vec3d> rawJacobians;
    sourceValues.reserve(samples.size());
    targetValues.reserve(samples.size());
    rawJacobians.reserve(samples.size());

    const double c = std::cos(params.theta);
    const double s = std::sin(params.theta);

    // 收集有效重叠样本，并记录 target 灰度对刚体参数的原始导数。
    for (const SamplePoint& sample : samples) {
        const double x = sample.x;
        const double y = sample.y;
        const double wx = c * x - s * y + params.tx;
        const double wy = s * x + c * y + params.ty;

        float targetValue = 0.0f;
        float ix = 0.0f;
        float iy = 0.0f;
        if (!rigid_direct_common::sampleBilinear(target, wx, wy, targetValue) ||
            !rigid_direct_common::sampleBilinear(gradX, wx, wy, ix) ||
            !rigid_direct_common::sampleBilinear(gradY, wx, wy, iy)) {
            continue;
        }

        const double dxdTheta = -s * x - c * y;
        const double dydTheta = c * x - s * y;
        rawJacobians.emplace_back(static_cast<double>(ix) * dxdTheta +
                                      static_cast<double>(iy) * dydTheta,
                                  static_cast<double>(ix),
                                  static_cast<double>(iy));
        sourceValues.push_back(sample.value);
        targetValues.push_back(targetValue);
    }

    linearization.valid = static_cast<int>(targetValues.size());
    if (linearization.valid < rigid_direct_common::kMinSamples) {
        message = "not enough overlapping samples";
        return false;
    }

    double meanSource = 0.0;
    double meanTarget = 0.0;
    cv::Vec3d meanRawJacobian(0.0, 0.0, 0.0);
    for (int i = 0; i < linearization.valid; ++i) {
        meanSource += sourceValues[static_cast<size_t>(i)];
        meanTarget += targetValues[static_cast<size_t>(i)];
        meanRawJacobian += rawJacobians[static_cast<size_t>(i)];
    }
    const double invCount = 1.0 / static_cast<double>(linearization.valid);
    meanSource *= invCount;
    meanTarget *= invCount;
    meanRawJacobian *= invCount;

    double varSource = 0.0;
    double varTarget = 0.0;
    cv::Vec3d centeredTargetWeightedJacobian(0.0, 0.0, 0.0);
    for (int i = 0; i < linearization.valid; ++i) {
        const double ds = static_cast<double>(sourceValues[static_cast<size_t>(i)]) - meanSource;
        const double dt = static_cast<double>(targetValues[static_cast<size_t>(i)]) - meanTarget;
        varSource += ds * ds;
        varTarget += dt * dt;
        centeredTargetWeightedJacobian += dt * rawJacobians[static_cast<size_t>(i)];
    }

    const double sigmaSource = std::sqrt(varSource * invCount + kZnccStdEps);
    const double sigmaTarget = std::sqrt(varTarget * invCount + kZnccStdEps);
    if (sigmaSource <= kZnccStdEps || sigmaTarget <= kZnccStdEps) {
        message = "low-variance normalized intensities";
        return false;
    }

    const double invSigmaSource = 1.0 / sigmaSource;
    const double invSigmaTarget = 1.0 / sigmaTarget;
    const double denom = static_cast<double>(linearization.valid) * sigmaTarget * sigmaTarget *
                         sigmaTarget;

    double sse = 0.0;
    for (int i = 0; i < linearization.valid; ++i) {
        const double centeredSource =
            static_cast<double>(sourceValues[static_cast<size_t>(i)]) - meanSource;
        const double centeredTarget =
            static_cast<double>(targetValues[static_cast<size_t>(i)]) - meanTarget;
        const double residual =
            centeredTarget * invSigmaTarget - centeredSource * invSigmaSource;
        const cv::Vec3d jacobian =
            (rawJacobians[static_cast<size_t>(i)] - meanRawJacobian) * invSigmaTarget -
            centeredTarget * centeredTargetWeightedJacobian / denom;

        for (int r = 0; r < 3; ++r) {
            linearization.g[r] += jacobian[r] * residual;
            for (int col = 0; col < 3; ++col) {
                linearization.h(r, col) += jacobian[r] * jacobian[col];
            }
        }

        sse += residual * residual;
    }

    linearization.objective = sse * invCount;
    return true;
}

// 功能：在单层图像上优化 ZNCC 刚体参数。
// 步骤：
// 1. 用解析 Jacobian 构造 ZNCC 线性化；
// 2. 对近似 Hessian 加 LM 阻尼，求解增量；
// 3. 用真实目标值检查该增量是否下降；
// 4. 成功时接受并减小阻尼，失败时增大阻尼重试；
// 5. 增量或梯度足够小时判定收敛。
bool optimizeRigidLevelZncc(const cv::Mat& source,
                            const cv::Mat& target,
                            const std::vector<SamplePoint>& samples,
                            int maxIterations,
                            double epsilon,
                            RigidParams& params,
                            LevelStats& stats,
                            std::string& message) {
    (void)source;

    stats = {};
    message.clear();
    if (samples.size() < rigid_direct_common::kMinSamples) {
        message = "not enough textured samples";
        return false;
    }

    cv::Mat gradX;
    cv::Mat gradY;
    rigid_direct_common::computeImageGradient(target, gradX, gradY);

    const int iterations = std::max(1, maxIterations);
    const double eps = std::max(0.0, epsilon);
    double lambda = kLmLambdaInit;

    for (int iter = 0; iter < iterations; ++iter) {
        ZnccLinearization linearization;
        if (!buildZnccLinearization(target, gradX, gradY, samples, params, linearization, message)) {
            return false;
        }

        bool accepted = false;
        double acceptedObjective = linearization.objective;
        int acceptedValid = linearization.valid;
        RigidParams acceptedParams = params;
        cv::Vec3d acceptedDelta(0.0, 0.0, 0.0);
        cv::Vec3d lastDelta(0.0, 0.0, 0.0);
        double trialLambda = lambda;

        // 同一轮迭代内逐步增大阻尼，直到找到真正降低目标值的更新。
        for (int trial = 0; trial < kLmMaxTrials; ++trial) {
            cv::Matx33d dampedH = linearization.h;
            for (int i = 0; i < 3; ++i) {
                dampedH(i, i) +=
                    trialLambda * std::max(1e-6, std::abs(linearization.h(i, i)));
            }

            cv::Vec3d delta;
            if (!rigid_direct_common::solveNormalEquation(dampedH, linearization.g, delta)) {
                trialLambda *= kLmLambdaGrow;
                if (trialLambda > kLmLambdaMax) {
                    break;
                }
                continue;
            }
            lastDelta = delta;

            // LM 给出一个候选增量后，再做一层小范围回溯缩步，提升非线性目标下的接受率。
            for (double scale : kBacktrackingScales) {
                RigidParams candidate = params;
                candidate.theta += scale * delta[0];
                candidate.tx += scale * delta[1];
                candidate.ty += scale * delta[2];
                if (candidate.theta > rigid_direct_common::kPi ||
                    candidate.theta < -rigid_direct_common::kPi) {
                    candidate.theta =
                        std::atan2(std::sin(candidate.theta), std::cos(candidate.theta));
                }

                double candidateObjective = std::numeric_limits<double>::infinity();
                int candidateValid = 0;
                std::string candidateMessage;
                if (!evaluateZnccObjective(target,
                                           samples,
                                           candidate,
                                           candidateObjective,
                                           candidateValid,
                                           candidateMessage)) {
                    continue;
                }

                if (candidateObjective + kObjectiveAcceptEps < linearization.objective) {
                    accepted = true;
                    acceptedObjective = candidateObjective;
                    acceptedValid = candidateValid;
                    acceptedParams = candidate;
                    acceptedDelta = scale * delta;
                    lambda = std::clamp(
                        trialLambda * kLmLambdaShrink, kLmLambdaFloor, kLmLambdaMax);
                    break;
                }
            }

            if (accepted) {
                break;
            }

            trialLambda *= kLmLambdaGrow;
            if (trialLambda > kLmLambdaMax) {
                break;
            }
        }

        if (!accepted) {
            const double gradInfNorm = std::max({std::abs(linearization.g[0]),
                                                 std::abs(linearization.g[1]),
                                                 std::abs(linearization.g[2])});
            const double maxDelta =
                std::max({std::abs(lastDelta[0]), std::abs(lastDelta[1]), std::abs(lastDelta[2])});

            stats.iterations = iter + 1;
            stats.residual.valid_samples = linearization.valid;
            stats.residual.mse = linearization.objective;
            if (gradInfNorm <= kGradientConvergedEps || maxDelta <= eps) {
                stats.converged = true;
                break;
            }

            message = "no descent step found";
            return false;
        }

        params = acceptedParams;
        stats.iterations = iter + 1;
        stats.residual.valid_samples = acceptedValid;
        stats.residual.mse = acceptedObjective;

        const double maxAcceptedDelta = std::max({std::abs(acceptedDelta[0]),
                                                  std::abs(acceptedDelta[1]),
                                                  std::abs(acceptedDelta[2])});
        if (maxAcceptedDelta <= eps) {
            stats.converged = true;
            break;
        }
    }

    // 退出迭代后再评估一次最终目标，保证输出统计与最终参数一致。
    double finalObjective = std::numeric_limits<double>::infinity();
    int finalValid = 0;
    std::string finalMessage;
    if (evaluateZnccObjective(target, samples, params, finalObjective, finalValid, finalMessage)) {
        stats.residual.valid_samples = finalValid;
        stats.residual.mse = finalObjective;
    }
    return true;
}

} // namespace

// 功能：从 YAML 读取 ZNCC 刚体配准所需参数。
// 说明：兼容参数直接写在根节点，或者统一写在 params 节点下两种形式。
ZnccRigidAligner::ZnccRigidAligner(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    _maxIterations = yaml_utils::getInt(params, "max_iterations", 50);
    _epsilon = yaml_utils::getDouble(params, "epsilon", 1e-4);
    _pyramidLevels = yaml_utils::getInt(params, "pyramid_levels", 4);
    _blurKernel = yaml_utils::getInt(params, "blur_kernel", 5);
    _gradientThreshold = yaml_utils::getDouble(params, "gradient_threshold", 1e-3);
    _sampleStep = yaml_utils::getInt(params, "sample_step", 2);
}

// 功能：执行完整的 ZNCC 刚体配准流程，并把最终分数写回上下文。
// 步骤：
// 1. 复用 rigid_direct_common 的多层刚体对齐主流程；
// 2. 若对齐成功，则在原图尺度重新评估最终 ZNCC；
// 3. 把最终有效样本数和 ZNCC 分数写回输出上下文。
bool ZnccRigidAligner::align(RegistrationContext& ctx) {
    RigidDirectOptions options;
    options.max_iterations = _maxIterations;
    options.epsilon = _epsilon;
    options.pyramid_levels = _pyramidLevels;
    options.blur_kernel = _blurKernel;
    options.gradient_threshold = _gradientThreshold;
    options.sample_step = _sampleStep;

    const bool ok = rigid_direct_common::runRigidAlignment(
        ctx, name(), "ZNCC rigid", options, optimizeRigidLevelZncc);
    if (!ok || !ctx.geometry_data.valid || ctx.geometry_data.A.empty()) {
        return ok;
    }

    cv::Mat source;
    cv::Mat target;
    if (!image_utils::convertGrayToFloat01(ctx.images.first_gray, source, _blurKernel) ||
        !image_utils::convertGrayToFloat01(ctx.images.second_gray, target, _blurKernel) ||
        source.size() != target.size()) {
        return true;
    }

    const std::vector<SamplePoint> samples =
        rigid_direct_common::buildSamples(source, _sampleStep, _gradientThreshold);
    const RigidParams params = paramsFromRigidMatrix(ctx.geometry_data.A);
    double objective = std::numeric_limits<double>::infinity();
    int valid = 0;
    std::string message;
    if (!evaluateZnccObjective(target, samples, params, objective, valid, message)) {
        IR_LOG_WARN("ZNCC rigid final score evaluation skipped: ", message);
        return true;
    }

    const double zncc = znccScoreFromObjective(objective);
    ctx.geometry_data.num_inliers = valid;
    ctx.geometry_data.inlier_ratio = zncc;
    ctx.direct_data.score = zncc;
    IR_LOG_INFO("ZNCC rigid final correlation=", zncc, ", normalized_objective=", objective);
    return true;
}

} // namespace ir
