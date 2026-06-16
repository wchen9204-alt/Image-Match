#pragma once

#include <opencv2/core.hpp>
#include <string>

#include "core/context.h"

namespace ir {

/// 图像变换器接口。
///
/// 根据几何估计结果对输入图像进行仿射或透视变换，并写入 `warped_image`。
class IWarper {
public:
    virtual ~IWarper() = default;

    /// 返回变换器名称。
    virtual std::string name() const = 0;

    /// 执行图像变换。
    virtual bool warp(RegistrationContext& ctx) = 0;
};

} // namespace ir

