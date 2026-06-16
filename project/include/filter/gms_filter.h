#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_filter.h"

namespace ir {

// ---------------------------------------------------------------------------
// 网格运动统计过滤器：按空间一致性筛选点特征匹配。
// ---------------------------------------------------------------------------
class GmsFilter : public IFilter {
public:
    explicit GmsFilter(const YAML::Node& cfg);

    std::string name() const override { return "GMS"; }

    bool apply(RegistrationContext& ctx) override;

private:
    bool _withRotation = false;
    bool _withScale = false;
    double _thresholdFactor = 6.0;
    bool _fallbackToInputIfEmpty = true;
};

}
