#pragma once

#include <string>

#include "core/context.h"
#include "core/types.h"

namespace ir {

/// 几何模型估计器接口。
///
/// 根据过滤后的匹配点估计单应矩阵、仿射矩阵、基础矩阵或本质矩阵。
class IGeometryEstimator {
public:
    virtual ~IGeometryEstimator() = default;

    /// 返回估计器名称。
    virtual std::string name() const = 0;

    /// 返回当前估计器对应的几何模型类型。
    virtual GeometryType type() const = 0;

    /// 根据上下文中的匹配结果估计几何模型。
    virtual bool estimate(RegistrationContext& ctx) = 0;
};

} // namespace ir

