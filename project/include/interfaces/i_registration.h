#pragma once

#include <memory>
#include <string>

#include "core/config.h"
#include "core/context.h"
#include "interfaces/i_pipeline.h"

namespace ir {

// ---------------------------------------------------------------------------
// IRegistration：封装已配置 pipeline 的高层配准接口。
// ---------------------------------------------------------------------------
class IRegistration {
public:
    virtual ~IRegistration() = default;

    virtual std::string name() const = 0;

    // 配置一次，可多次运行。
    virtual bool configure(const PipelineConfig& cfg) = 0;

    virtual bool run(RegistrationContext& ctx) = 0;
};

} // namespace ir
