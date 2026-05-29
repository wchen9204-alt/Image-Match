#pragma once

#include <string>

#include "core/config.h"
#include "core/context.h"

namespace ir {

// ---------------------------------------------------------------------------
// IPipeline：组织完整的图像配准流程。
// ---------------------------------------------------------------------------
class IPipeline {
public:
    virtual ~IPipeline() = default;

    virtual std::string name() const = 0;

    // 根据 PipelineConfig 配置各子组件。
    virtual bool configure(const PipelineConfig& cfg) = 0;

    // 对 ctx 执行完整配准流程。
    virtual bool run(RegistrationContext& ctx) = 0;
};

} // namespace ir
