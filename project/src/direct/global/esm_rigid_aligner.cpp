#include "direct/global/esm_rigid_aligner.h"

#include "direct/common/esm_rigid_common.h"
#include "direct/common/rigid_direct_common.h"
#include "utils/yaml_utils.h"

namespace ir {

using rigid_direct_common::RigidDirectOptions;

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
        ctx, name(), "ESM rigid", options, esm_rigid_common::optimizeRigidLevel);
}

} // namespace ir
