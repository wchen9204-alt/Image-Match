#pragma once

#include <string>

#include "core/context.h"

namespace ir {

/// 描述子匹配器接口。
///
/// 接收特征提取阶段产生的描述子，并生成原始匹配结果写入上下文。
class IMatcher {
public:
    virtual ~IMatcher() = default;

    /// 返回匹配器名称。
    virtual std::string name() const = 0;

    /// 执行描述子匹配，结果写入 `RegistrationContext::match_data`。
    virtual bool match(RegistrationContext& ctx) = 0;
};

} // namespace ir

