#include "direct/dense/dis_flow_aligner.h"

#include <algorithm>
#include <string>

#include <opencv2/video/tracking.hpp>

#include "utils/string_utils.h"
#include "utils/yaml_utils.h"

namespace ir {
namespace {

int disPresetFromString(const std::string& preset) {
    const std::string key = string_utils::normalizedKey(preset);
    if (key == "ULTRAFAST" || key == "PRESETULTRAFAST") {
        return cv::DISOpticalFlow::PRESET_ULTRAFAST;
    }
    if (key == "MEDIUM" || key == "PRESETMEDIUM") {
        return cv::DISOpticalFlow::PRESET_MEDIUM;
    }
    return cv::DISOpticalFlow::PRESET_FAST;
}

} // namespace

DisFlowAligner::DisFlowAligner(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    // 兼容算法配置文件中直接写参数或写在 params 节点下两种形式。
    _preset = disPresetFromString(yaml_utils::getString(params, "preset", "FAST"));
    _finestScale = yaml_utils::getInt(params, "finest_scale", -1);
    _patchSize = yaml_utils::getInt(params, "patch_size", -1);
    _patchStride = yaml_utils::getInt(params, "patch_stride", -1);
    _gradientDescentIterations =
        yaml_utils::getInt(params, "gradient_descent_iterations", -1);
    _variationalRefinementIterations =
        yaml_utils::getInt(params, "variational_refinement_iterations", -1);
    _variationalRefinementAlpha =
        yaml_utils::getFloat(params, "variational_refinement_alpha", -1.0f);
    _variationalRefinementDelta =
        yaml_utils::getFloat(params, "variational_refinement_delta", -1.0f);
    _variationalRefinementGamma =
        yaml_utils::getFloat(params, "variational_refinement_gamma", -1.0f);
    _variationalRefinementEpsilon =
        yaml_utils::getFloat(params, "variational_refinement_epsilon", -1.0f);
    _useMeanNormalization = yaml_utils::getBool(params, "use_mean_normalization", true);
    _useSpatialPropagation = yaml_utils::getBool(params, "use_spatial_propagation", true);
    _blurKernel = yaml_utils::getInt(params, "blur_kernel", 0);
    _postprocess = dense_flow_common::readPostprocessOptions(params);
}

bool DisFlowAligner::align(RegistrationContext& ctx) {
    auto& dd = ctx.direct_data;
    auto& gd = ctx.geometry_data;
    dd.clear();
    gd.clear();
    dd.method = name();

    // 1. 读取并预处理灰度图，作为 DIS 稠密光流输入。
    if (ctx.images.first_gray.empty() || ctx.images.second_gray.empty()) {
        dd.message = "DIS requires non-empty grayscale images";
        gd.message = dd.message;
        return false;
    }

    const cv::Mat firstGray =
        dense_flow_common::prepareGray(ctx.images.first_gray, _blurKernel);
    const cv::Mat secondGray =
        dense_flow_common::prepareGray(ctx.images.second_gray, _blurKernel);
    if (firstGray.empty() || secondGray.empty()) {
        dd.message = "DIS failed to prepare grayscale images";
        gd.message = dd.message;
        return false;
    }

    auto createSolver = [this]() {
        cv::Ptr<cv::DISOpticalFlow> solver = cv::DISOpticalFlow::create(_preset);
        // 负值表示沿用 preset 默认值，避免 YAML 未填写时破坏 OpenCV 的 preset 语义。
        if (_finestScale >= 0) {
            solver->setFinestScale(_finestScale);
        }
        if (_patchSize > 0) {
            solver->setPatchSize(_patchSize);
        }
        if (_patchStride > 0) {
            solver->setPatchStride(_patchStride);
        }
        if (_gradientDescentIterations >= 0) {
            solver->setGradientDescentIterations(_gradientDescentIterations);
        }
        if (_variationalRefinementIterations >= 0) {
            solver->setVariationalRefinementIterations(_variationalRefinementIterations);
        }
        if (_variationalRefinementAlpha >= 0.0f) {
            solver->setVariationalRefinementAlpha(_variationalRefinementAlpha);
        }
        if (_variationalRefinementDelta >= 0.0f) {
            solver->setVariationalRefinementDelta(_variationalRefinementDelta);
        }
        if (_variationalRefinementGamma >= 0.0f) {
            solver->setVariationalRefinementGamma(_variationalRefinementGamma);
        }
        if (_variationalRefinementEpsilon >= 0.0f) {
            solver->setVariationalRefinementEpsilon(_variationalRefinementEpsilon);
        }
        solver->setUseMeanNormalization(_useMeanNormalization);
        solver->setUseSpatialPropagation(_useSpatialPropagation);
        return solver;
    };

    // 2. 用 DIS 直接估计整幅图的前向稠密光流。
    // DIS 输出每个源图像素到目标图的二维位移，后续与 Farneback 共用采样和全局几何拟合。
    cv::Ptr<cv::DISOpticalFlow> forwardSolver = createSolver();
    forwardSolver->calc(firstGray, secondGray, dd.flow);

    // 3. 若开启前后向一致性检查，则补算一份 target->source 光流。
    cv::Mat backwardFlow;
    if (_postprocess.forward_backward_check) {
        // 前后向一致性检查需要额外估计 target->source 光流，用于剔除不自洽采样点。
        cv::Ptr<cv::DISOpticalFlow> backwardSolver = createSolver();
        backwardSolver->calc(secondGray, firstGray, backwardFlow);
    }

    // 4. 复用共享后处理：采样、过滤、全局几何拟合，并把结果写回统一输出结构。
    return dense_flow_common::finalizeFlowAlignment(
        ctx, firstGray, backwardFlow, _postprocess, name());
}

} // namespace ir

