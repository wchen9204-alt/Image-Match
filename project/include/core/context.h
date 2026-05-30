#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "core/result.h"
#include "core/types.h"
#include "data/evaluation_data.h"
#include "data/feature_data.h"
#include "data/geometry_data.h"
#include "data/match_data.h"
#include "data/transform_data.h"

namespace ir {

/// 配准流程贯穿各阶段的共享上下文。
///
/// 各个模块通过这个对象读写特征、匹配、几何估计、变换结果和评测数据，
/// 避免在模块之间频繁传递零散参数。
class RegistrationContext {
public:
    /// 构造一个空上下文。
    RegistrationContext() = default;

    /// 原始与中间结果数据：特征、匹配、几何、变换以及评测信息。
    FeatureData feature_data;
    MatchData match_data;
    GeometryData geometry_data;
    TransformData transform_data;
    EvaluationData evaluation;

    /// 本次配准流程的汇总结果。
    RegistrationResult result;

    /// 输入图像 1 和图像 2 的路径，以及结果输出目录。
    std::filesystem::path image1_path;
    std::filesystem::path image2_path;
    std::filesystem::path output_dir;

    /// 可选的变换后图像。
    cv::Mat warped_image;

    /// 重置运行时数据，但保留已配置的输入输出路径。
    void reset() {
        feature_data.clear();
        match_data.clear();
        geometry_data.clear();
        transform_data.clear();
        evaluation.clear();
        warped_image.release();
        result = RegistrationResult{};
    }
};

} // namespace ir
