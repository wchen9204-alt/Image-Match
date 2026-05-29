#pragma once

#include "pipeline/base_pipeline.h"

namespace ir {

/// 当前项目默认使用的特征配准流水线。
class FeaturePipeline : public BasePipeline {
public:
    FeaturePipeline() = default;

    /// 返回流水线名称。
    std::string name() const override { return "FeaturePipeline"; }
};

} // namespace ir
