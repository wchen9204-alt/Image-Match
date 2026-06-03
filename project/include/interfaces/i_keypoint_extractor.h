#pragma once

#include <string>

#include "core/context.h"
#include "core/types.h"

namespace ir {

/// 特征提取器接口。
///
/// 负责从图像中检测关键点并计算描述子，结果写入 `RegistrationContext`。
class IKeypointExtractor {
public:
    virtual ~IKeypointExtractor() = default;

    /// 返回提取器名称，用于日志和调试输出。
    virtual std::string name() const = 0;

    /// 返回当前提取器对应的特征类型。
    virtual KeypointType type() const = 0;

    /// 返回描述子匹配所使用的距离类型。
    virtual NormType normType() const = 0;

    /// 在上下文中执行特征检测与描述子计算。
    virtual bool extract(RegistrationContext& ctx) = 0;
};

} // namespace ir
