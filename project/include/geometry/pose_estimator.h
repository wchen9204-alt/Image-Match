#pragma once

#include <opencv2/core.hpp>

#include "core/context.h"

namespace ir {

// ---------------------------------------------------------------------------
// PoseEstimator：基于已估计的本质矩阵恢复相机姿态。
//
// 该类不注册到 Factory，由 EssentialEstimator 直接调用。
// ---------------------------------------------------------------------------
class PoseEstimator {
public:
    PoseEstimator() = default;

    // 从 E 和 K 恢复 R、t，并返回通过手性检查的内点数量。
    int recoverPose(RegistrationContext& ctx);
};

} // namespace ir
