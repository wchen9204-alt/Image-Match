#pragma once

#include <opencv2/core.hpp>

#include "core/context.h"

namespace ir {

// ---------------------------------------------------------------------------
// IWarper：应用几何变换并写入 ctx.warped_image。
// ---------------------------------------------------------------------------
class IWarper {
public:
    virtual ~IWarper() = default;

    virtual std::string name() const = 0;

    virtual bool warp(RegistrationContext& ctx) = 0;
};

} // namespace ir
