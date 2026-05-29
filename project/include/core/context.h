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

// ---------------------------------------------------------------------------
// RegistrationContext：贯穿整个配准流程的共享上下文。
//
// 各组件应通过上下文读写数据，不持有其内部字段的悬空引用。
// ---------------------------------------------------------------------------
class RegistrationContext {
public:
    RegistrationContext() = default;

    // ---- 主要数据 ------------------------------------------------------
    FeatureData    feature_data;
    MatchData      match_data;
    GeometryData   geometry_data;
    TransformData  transform_data;
    EvaluationData evaluation;

    // ---- 摘要结果 ------------------------------------------------------
    RegistrationResult result;

    // ---- 输入输出路径 --------------------------------------------------
    std::filesystem::path image1_path;
    std::filesystem::path image2_path;
    std::filesystem::path output_dir;

    // 可选的变换后图像。
    cv::Mat warped_image;

    // ---- 工具方法 ------------------------------------------------------
    // 重置运行数据，但保留已配置路径。
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
