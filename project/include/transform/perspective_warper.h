#pragma once

#include "transform/warper.h"

namespace ir {

/// 基于 3x3 单应矩阵的透视变换器。
class PerspectiveWarper : public IWarper {
public:
    /// 构造一个透视变换器实例。
    PerspectiveWarper() = default;

    /// 返回变换器名称。
    std::string name() const override { return "PerspectiveWarper"; }

    /// 对上下文中的图像执行透视变换。
    bool warp(RegistrationContext& ctx) override;
};

} // namespace ir

