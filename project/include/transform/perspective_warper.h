#pragma once

#include "transform/warper.h"

namespace ir {

// ---------------------------------------------------------------------------
// PerspectiveWarper：将第一张图变换到第二张图的坐标系。
// ---------------------------------------------------------------------------
class PerspectiveWarper : public IWarper {
public:
    PerspectiveWarper() = default;

    std::string name() const override { return "PerspectiveWarper"; }

    bool warp(RegistrationContext& ctx) override;
};

} // namespace ir
