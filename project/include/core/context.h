#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "core/result.h"
#include "core/types.h"
#include "data/evaluation_data.h"
#include "data/direct_data.h"
#include "data/feature_initializer_data.h"
#include "data/geometry_data.h"
#include "data/image_data.h"
#include "data/keypoint_data.h"
#include "data/keypoint_match_data.h"
#include "data/structure_match_data.h"
#include "data/structure_data.h"
#include "data/transform_data.h"

namespace ir {

/// 配准流程贯穿各阶段的共享上下文。
class RegistrationContext {
public:
    RegistrationContext() = default;

    ImagePairData images;
    KeypointData keypoint_data;
    StructureData structure_data;
    StructureMatchData structure_match_data;
    KeypointMatchData keypoint_match_data;

    /// 当前阶段显式使用的对应点来源类型；由 pipeline 或 direct aligner 在进入几何/可视化阶段前写入。
    std::string correspondence_source;

    /// 直接法阶段的专属输出；DirectPipeline 会从这里同步通用几何结果和可视化点对。
    DirectData direct_data;

    /// 直接法前置点特征初始化结果；仅表示是否可作为直接法初始值，不代表最终配准结果。
    FeatureInitializerData feature_initializer_data;

    GeometryData geometry_data;
    TransformData transform_data;
    EvaluationData evaluation;

    RegistrationResult result;

    std::filesystem::path image1_path;
    std::filesystem::path image2_path;
    std::filesystem::path output_dir;

    cv::Mat warped_image;

    /// 清空所有阶段缓存，保证批处理或多次运行时上下文互不污染。
    void reset() {
        images.clear();
        keypoint_data.clear();
        structure_data.clear();
        structure_match_data.clear();
        keypoint_match_data.clear();
        correspondence_source.clear();
        direct_data.clear();
        feature_initializer_data.clear();
        geometry_data.clear();
        transform_data.clear();
        evaluation.clear();
        warped_image.release();
        result = RegistrationResult{};
    }
};

} // namespace ir

