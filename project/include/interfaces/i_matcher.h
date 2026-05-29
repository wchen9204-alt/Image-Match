#pragma once

#include <string>

#include "core/context.h"

namespace ir {

// ---------------------------------------------------------------------------
// IMatcher：生成两张图描述子之间的初始匹配。
// ---------------------------------------------------------------------------
class IMatcher {
public:
    virtual ~IMatcher() = default;

    virtual std::string name() const = 0;

    // 执行匹配；AUTO 距离类型应参考 FeatureData::norm_type。
    virtual bool match(RegistrationContext& ctx) = 0;
};

} // namespace ir
