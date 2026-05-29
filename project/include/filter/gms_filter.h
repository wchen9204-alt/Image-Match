#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_filter.h"

namespace ir {

// ---------------------------------------------------------------------------
// GmsFilter：基于 GMS 的空间一致性匹配过滤器。
// ---------------------------------------------------------------------------
class GmsFilter : public IFilter {
public:
    explicit GmsFilter(const YAML::Node& cfg);

    std::string name() const override { return "GMS"; }

    bool apply(RegistrationContext& ctx) override;

private:
    bool   withRotation_   = false;
    bool   withScale_      = false;
    double thresholdFactor_ = 6.0;
};

} // namespace ir
