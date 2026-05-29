#pragma once

#include <yaml-cpp/yaml.h>
#include <opencv2/features2d.hpp>

#include "core/types.h"
#include "interfaces/i_matcher.h"

namespace ir {

class BfMatcher : public IMatcher {
public:
    explicit BfMatcher(const YAML::Node& cfg);

    std::string name() const override { return "BFMatcher"; }

    bool match(RegistrationContext& ctx) override;

private:
    NormType norm_type_  = NormType::UNKNOWN;   // UNKNOWN 表示 AUTO。
    bool     crossCheck_ = false;
    int      knn_k_      = 2;
};

} // namespace ir
