#include "filter/gms_filter.h"

#include <opencv2/xfeatures2d.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

// 读取网格运动统计所需的参数开关，用于几何一致性过滤。
GmsFilter::GmsFilter(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    _withRotation = yaml_utils::getBool(params, "withRotation", false);
    _withScale = yaml_utils::getBool(params, "withScale", false);
    _thresholdFactor = yaml_utils::getDouble(params, "thresholdFactor", 6.0);

    IR_LOG_INFO("GmsFilter: withRotation=",
                _withRotation,
                ", withScale=",
                _withScale,
                ", thresholdFactor=",
                _thresholdFactor);
}

bool GmsFilter::apply(RegistrationContext& ctx) {
    // 结构法路径：网格运动统计依赖关键点空间分布，当前结构匹配不支持。
    if (!ctx.structure_match_data.raw_matches_knn.empty() ||
        !ctx.structure_match_data.filtered_matches.empty()) {
        IR_LOG_WARN("GmsFilter [structure]: not supported for line matches, pass-through.");
        return true;
    }

    const auto& fd = ctx.keypoint_data;
    const auto& images = ctx.images;
    auto& md = ctx.keypoint_match_data;

    if (images.first.empty() || images.second.empty()) {
        IR_LOG_ERROR("GMS: source images empty.");
        return false;
    }

    std::vector<cv::DMatch> input = md.filtered;
    if (input.empty()) {
        IR_LOG_WARN("GMS: no input matches.");
        return false;
    }

    // 步骤一：调用网格运动统计，保留局部运动一致的匹配。
    std::vector<cv::DMatch> kept;
    cv::xfeatures2d::matchGMS(images.first.size(),
                              images.second.size(),
                              fd.first.keypoints,
                              fd.second.keypoints,
                              input,
                              kept,
                              _withRotation,
                              _withScale,
                              _thresholdFactor);

    IR_LOG_INFO("GMS kept ", kept.size(), " / ", input.size(), " matches");

    // 步骤二：用网格运动统计精炼后的结果覆盖当前筛选结果。
    md.filtered = std::move(kept);
    return true;
}

}
