#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "core/result.h"
#include "core/types.h"
#include "data/evaluation_data.h"
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
    GeometryData geometry_data;
    TransformData transform_data;
    EvaluationData evaluation;

    RegistrationResult result;

    std::filesystem::path image1_path;
    std::filesystem::path image2_path;
    std::filesystem::path output_dir;

    cv::Mat warped_image;

    void reset() {
        images.clear();
        keypoint_data.clear();
        structure_data.clear();
        structure_match_data.clear();
        keypoint_match_data.clear();
        geometry_data.clear();
        transform_data.clear();
        evaluation.clear();
        warped_image.release();
        result = RegistrationResult{};
    }
};

} // namespace ir
