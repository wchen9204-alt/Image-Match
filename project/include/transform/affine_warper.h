#pragma once

#include "transform/warper.h"

namespace ir {

// ---------------------------------------------------------------------------
// AffineWarper：使用 2x3 仿射矩阵生成变换后图像。
// ---------------------------------------------------------------------------
class AffineWarper : public IWarper {
public:
    AffineWarper() = default;

    std::string name() const override { return "AffineWarper"; }

    bool warp(RegistrationContext& ctx) override;
};

} // namespace ir
