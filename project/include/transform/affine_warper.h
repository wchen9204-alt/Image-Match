#pragma once

#include "transform/warper.h"

namespace ir {

/// 基于 2x3 仿射矩阵的图像变换器。
class AffineWarper : public IWarper {
public:
    /// 构造一个仿射变换器实例。
    AffineWarper() = default;

    /// 返回变换器名称。
    std::string name() const override { return "AffineWarper"; }

    /// 对上下文中的图像执行仿射变换。
    bool warp(RegistrationContext& ctx) override;
};

} // namespace ir
