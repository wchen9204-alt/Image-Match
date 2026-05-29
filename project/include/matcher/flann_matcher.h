#pragma once

#include <yaml-cpp/yaml.h>
#include <opencv2/features2d.hpp>

#include "core/types.h"
#include "interfaces/i_matcher.h"

namespace ir {

// ---------------------------------------------------------------------------
// FlannMatcher：封装 FlannBasedMatcher，并按描述子类型选择 KDTree 或 LSH。
// ---------------------------------------------------------------------------
class FlannMatcher : public IMatcher {
public:
    explicit FlannMatcher(const YAML::Node& cfg);

    std::string name() const override { return "FlannMatcher"; }

    bool match(RegistrationContext& ctx) override;

private:
    NormType norm_type_ = NormType::UNKNOWN;
    int      knn_k_     = 2;

    // KDTree 参数。
    int kd_trees_       = 5;

    // LSH 参数。
    int lsh_table_number_      = 12;
    int lsh_key_size_          = 20;
    int lsh_multi_probe_level_ = 2;

    // 搜索参数。
    int    search_checks_  = 50;
    float  search_eps_     = 0.0f;
    bool   search_sorted_  = true;
};

} // namespace ir
