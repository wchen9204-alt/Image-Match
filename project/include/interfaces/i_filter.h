#pragma once

#include <string>

#include "core/context.h"

namespace ir {

/// 匹配过滤器接口。
///
/// 用于对原始匹配结果进行二次筛选，例如 ratio test、cross-check 或 GMS。
class IFilter {
public:
    virtual ~IFilter() = default;

    /// 返回过滤器名称。
    virtual std::string name() const = 0;

    /// 对上下文中的匹配结果执行过滤。
    virtual bool apply(RegistrationContext& ctx) = 0;
};

} // namespace ir

