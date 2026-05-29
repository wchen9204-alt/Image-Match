#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "core/types.h"
#include "interfaces/i_matcher.h"

namespace ir {

/// 基于暴力搜索的描述子匹配器。
class BfMatcher : public IMatcher {
public:
    /// 根据 YAML 配置初始化匹配器参数。
    explicit BfMatcher(const YAML::Node& cfg);

    /// 返回匹配器名称。
    std::string name() const override { return "BFMatcher"; }

    /// 在上下文中执行最近邻匹配。
    bool match(RegistrationContext& ctx) override;

private:
    NormType norm_type_  = NormType::UNKNOWN;
    bool     _crossCheck  = false;
    int      knn_k_       = 2;
};

} // namespace ir

