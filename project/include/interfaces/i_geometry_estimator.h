#pragma once

#include <string>

#include "core/context.h"
#include "core/types.h"

namespace ir {

// ---------------------------------------------------------------------------
// IGeometryEstimator：根据过滤后的匹配估计几何模型，并写入内点信息。
// ---------------------------------------------------------------------------
class IGeometryEstimator {
public:
    virtual ~IGeometryEstimator() = default;

    virtual std::string  name() const = 0;
    virtual GeometryType type() const = 0;

    virtual bool estimate(RegistrationContext& ctx) = 0;
};

} // namespace ir
