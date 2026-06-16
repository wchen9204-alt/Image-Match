#include "utils/descriptor_norm_utils.h"

#include "utils/yaml_utils.h"

namespace ir {
namespace descriptor_norm_utils {

NormType readConfiguredNorm(const YAML::Node& cfg, NormType fallback) {
    /* 1. 优先读取当前配置根节点，兼容现有 keypoint/*.yaml 的顶层 descriptor_norm。 */
    std::string norm_string = yaml_utils::getString(cfg, "descriptor_norm");
    if (norm_string.empty()) {
        /* 2. 同时兼容未来将 descriptor_norm 放入 params 的配置写法。 */
        const YAML::Node params = cfg["params"];
        norm_string = yaml_utils::getString(params, "descriptor_norm");
    }
    if (norm_string.empty()) {
        return fallback;
    }

    const NormType parsed = normTypeFromString(norm_string);
    /* 3. AUTO 会解析成 UNKNOWN，此时回到算法默认值，避免把 UNKNOWN 写入提取结果。 */
    return parsed == NormType::UNKNOWN ? fallback : parsed;
}

NormType inferFromDescriptors(const cv::Mat& descriptors) {
    if (descriptors.empty()) {
        return NormType::UNKNOWN;
    }
    /* OpenCV 二进制描述子通常使用 CV_8U，其它浮点描述子默认使用 L2。 */
    return descriptors.depth() == CV_8U ? NormType::HAMMING : NormType::L2;
}

NormType resolve(NormType configuredNorm, NormType providerNorm, const cv::Mat& descriptors) {
    /* 1. matcher 显式 norm_type 优先级最高，用于手动覆盖描述子侧建议。 */
    if (configuredNorm != NormType::UNKNOWN) {
        return configuredNorm;
    }
    /* 2. matcher 使用 AUTO 时，继承描述子提供方写入的 norm_type。 */
    if (providerNorm != NormType::UNKNOWN) {
        return providerNorm;
    }

    /* 3. 两侧都未指定时，按描述子矩阵类型做最后兜底。 */
    const NormType inferred = inferFromDescriptors(descriptors);
    return inferred == NormType::UNKNOWN ? NormType::L2 : inferred;
}

} // namespace descriptor_norm_utils
} // namespace ir
