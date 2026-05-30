#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_filter.h"

namespace ir {

// ---------------------------------------------------------------------------
// RatioTestFilter：对 k-NN 匹配执行 Lowe 比值检验。
// ---------------------------------------------------------------------------
class RatioTestFilter : public IFilter {
public:
    explicit RatioTestFilter(const YAML::Node& cfg);

    std::string name() const override { return "RatioTest"; }

    bool apply(RegistrationContext& ctx) override;

private:
    float _ratio = 0.75f;
};

} // namespace ir
