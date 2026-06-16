#pragma once

#include <string>

#include "core/context.h"

namespace ir {

/// 深度学习匹配器接口；第一版主要用于桥接 Python 推理并产出统一点对。
class ILearningMatcher {
public:
    virtual ~ILearningMatcher() = default;

    /// 返回方法名称，用于日志、摘要和输出命名。
    virtual std::string name() const = 0;

    /// 执行深度匹配并写入 keypoint_data / keypoint_match_data。
    virtual bool match(RegistrationContext& ctx) = 0;
};

} // namespace ir

