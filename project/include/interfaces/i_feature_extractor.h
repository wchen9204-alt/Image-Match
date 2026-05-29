#pragma once

#include <string>

#include "core/context.h"
#include "core/types.h"

namespace ir {

// ---------------------------------------------------------------------------
// IFeatureExtractor：检测关键点并计算两张图的描述子。
// ---------------------------------------------------------------------------
class IFeatureExtractor {
public:
    virtual ~IFeatureExtractor() = default;

    // 具体提取器名称。
    virtual std::string name() const = 0;

    // 特征类型。
    virtual FeatureType type() const = 0;

    // 描述子匹配所需的距离类型。
    virtual NormType    normType() const = 0;

    // 对两张图执行检测和描述，成功返回 true。
    virtual bool extract(RegistrationContext& ctx) = 0;
};

} // namespace ir
