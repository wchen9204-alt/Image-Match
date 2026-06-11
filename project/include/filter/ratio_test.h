#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_filter.h"

namespace ir {

// ---------------------------------------------------------------------------
// 比值检验过滤器：对近邻候选匹配执行距离比值检验。
// ---------------------------------------------------------------------------
class RatioTestFilter : public IFilter {
public:
    explicit RatioTestFilter(const YAML::Node& cfg);

    std::string name() const override { return "RatioTest"; }

    bool apply(RegistrationContext& ctx) override;

private:
    float _ratio = 0.75f;
};

}
