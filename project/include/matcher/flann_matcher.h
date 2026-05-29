#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "core/types.h"
#include "interfaces/i_matcher.h"

namespace ir {

/// 基于 FLANN 的描述子匹配器。
class FlannMatcher : public IMatcher {
public:
    /// 根据 YAML 配置初始化匹配器参数。
    explicit FlannMatcher(const YAML::Node& cfg);

    /// 返回匹配器名称。
    std::string name() const override { return "FlannMatcher"; }

    /// 在上下文中执行近似最近邻匹配。
    bool match(RegistrationContext& ctx) override;

private:
    NormType norm_type_ = NormType::UNKNOWN;
    int      knn_k_     = 2;

    /// KDTree 相关参数。
    int kd_trees_ = 5;

    /// LSH 相关参数。
    int lsh_table_number_      = 12;
    int lsh_key_size_          = 20;
    int lsh_multi_probe_level_ = 2;

    /// 搜索参数。
    int   search_checks_ = 50;
    float search_eps_    = 0.0f;
    bool  search_sorted_ = true;
};

} // namespace ir
