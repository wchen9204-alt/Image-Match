#include "direct/dense/tvl1_flow_aligner.h"

#include <algorithm>

#ifdef IR_HAS_OPENCV_OPTFLOW
#include <opencv2/optflow.hpp>
#endif

#include "utils/yaml_utils.h"

namespace ir {

Tvl1FlowAligner::Tvl1FlowAligner(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    // 兼容算法配置文件中直接写参数或写在 params 节点下两种形式。
    _tau = yaml_utils::getDouble(params, "tau", 0.25);
    _lambda = yaml_utils::getDouble(params, "lambda", 0.15);
    _theta = yaml_utils::getDouble(params, "theta", 0.3);
    _gamma = yaml_utils::getDouble(params, "gamma", 0.0);
    _scalesNumber = yaml_utils::getInt(params, "scales_number", 5);
    _warpingsNumber = yaml_utils::getInt(params, "warpings_number", 5);
    _epsilon = yaml_utils::getDouble(params, "epsilon", 0.01);
    _innerIterations = yaml_utils::getInt(params, "inner_iterations", 30);
    _outerIterations = yaml_utils::getInt(params, "outer_iterations", 10);
    _scaleStep = yaml_utils::getDouble(params, "scale_step", 0.8);
    _medianFiltering = yaml_utils::getInt(params, "median_filtering", 5);
    _useInitialFlow = yaml_utils::getBool(params, "use_initial_flow", false);
    _blurKernel = yaml_utils::getInt(params, "blur_kernel", 0);
    _postprocess = dense_flow_common::readPostprocessOptions(params);
}

bool Tvl1FlowAligner::align(RegistrationContext& ctx) {
    auto& dd = ctx.direct_data;
    auto& gd = ctx.geometry_data;
    dd.clear();
    gd.clear();
    dd.method = name();

#ifndef IR_HAS_OPENCV_OPTFLOW
    // 1. 若当前 OpenCV 未编译 optflow 模块，则直接返回能力缺失错误。
    dd.message = "TV-L1 requires OpenCV optflow module, but this build was configured without it";
    gd.message = dd.message;
    return false;
#else
    // 1. 读取并预处理灰度图，作为 TV-L1 稠密光流输入。
    if (ctx.images.first_gray.empty() || ctx.images.second_gray.empty()) {
        dd.message = "TV-L1 requires non-empty grayscale images";
        gd.message = dd.message;
        return false;
    }

    const cv::Mat firstGray =
        dense_flow_common::prepareGray(ctx.images.first_gray, _blurKernel);
    const cv::Mat secondGray =
        dense_flow_common::prepareGray(ctx.images.second_gray, _blurKernel);
    if (firstGray.empty() || secondGray.empty()) {
        dd.message = "TV-L1 failed to prepare grayscale images";
        gd.message = dd.message;
        return false;
    }

    auto createSolver = [this]() {
        cv::Ptr<cv::optflow::DualTVL1OpticalFlow> solver =
            cv::optflow::DualTVL1OpticalFlow::create();
        // 显式写入 YAML 参数，保证 TV-L1 实验可复现，不依赖 OpenCV 版本内部默认值。
        solver->setTau(std::max(1e-9, _tau));
        solver->setLambda(std::max(1e-9, _lambda));
        solver->setTheta(std::max(1e-9, _theta));
        solver->setGamma(std::max(0.0, _gamma));
        solver->setScalesNumber(std::max(1, _scalesNumber));
        solver->setWarpingsNumber(std::max(1, _warpingsNumber));
        solver->setEpsilon(std::max(1e-12, _epsilon));
        solver->setInnerIterations(std::max(1, _innerIterations));
        solver->setOuterIterations(std::max(1, _outerIterations));
        solver->setScaleStep(std::clamp(_scaleStep, 0.01, 0.99));
        solver->setMedianFiltering(std::max(1, _medianFiltering));
        solver->setUseInitialFlow(_useInitialFlow);
        return solver;
    };

    // 2. 用 TV-L1 估计源图到目标图的前向稠密光流。
    // TV-L1 输出每个源图像素到目标图的二维位移，后续共用稠密光流采样和全局几何拟合。
    cv::Ptr<cv::optflow::DualTVL1OpticalFlow> forwardSolver = createSolver();
    if (_useInitialFlow) {
        // 当前平台没有外部初始光流输入；开启时以零光流作为显式初始值，便于复现实验条件。
        dd.flow = cv::Mat::zeros(firstGray.size(), CV_32FC2);
    }
    forwardSolver->calc(firstGray, secondGray, dd.flow);

    // 3. 若开启前后向一致性检查，则额外估计 target->source 光流。
    cv::Mat backwardFlow;
    if (_postprocess.forward_backward_check) {
        // 前后向一致性检查需要额外估计 target->source 光流，TV-L1 下会明显增加耗时。
        cv::Ptr<cv::optflow::DualTVL1OpticalFlow> backwardSolver = createSolver();
        if (_useInitialFlow) {
            backwardFlow = cv::Mat::zeros(secondGray.size(), CV_32FC2);
        }
        backwardSolver->calc(secondGray, firstGray, backwardFlow);
    }

    // 4. 复用共享后处理：采样、过滤、全局几何拟合，并把结果写回统一输出结构。
    return dense_flow_common::finalizeFlowAlignment(
        ctx, firstGray, backwardFlow, _postprocess, name());
#endif
}

} // namespace ir

