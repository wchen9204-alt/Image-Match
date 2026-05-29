#pragma once

#include <string>

#include "core/context.h"

namespace ir {

// ---------------------------------------------------------------------------
// IFilter：在 ctx.match_data 中原地过滤匹配结果。
// ---------------------------------------------------------------------------
class IFilter {
public:
    virtual ~IFilter() = default;

    virtual std::string name() const = 0;

    virtual bool apply(RegistrationContext& ctx) = 0;
};

} // namespace ir
