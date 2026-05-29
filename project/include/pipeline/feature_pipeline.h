#pragma once

#include "pipeline/base_pipeline.h"

namespace ir {

// ---------------------------------------------------------------------------
// FeaturePipeline：当前应用使用的稀疏特征配准流程。
// ---------------------------------------------------------------------------
class FeaturePipeline : public BasePipeline {
public:
    FeaturePipeline() = default;

    std::string name() const override { return "FeaturePipeline"; }
};

} // namespace ir
