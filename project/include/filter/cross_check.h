#pragma once

#include <yaml-cpp/yaml.h>

#include "core/types.h"
#include "interfaces/i_filter.h"

namespace ir {

/// 通过双向一致性检查过滤匹配点，只保留互相匹配的最近邻对。
///
/// 这种过滤通常用于暴力匹配之后，用来剔除单向成立、双向不一致的匹配。
class CrossCheckFilter : public IFilter {
public:
    /// 根据 YAML 配置创建过滤器。
    explicit CrossCheckFilter(const YAML::Node& cfg);

    /// 返回日志和诊断中使用的显示名称。
    std::string name() const override { return "CrossCheck"; }

    /// 对上下文中的匹配结果执行双向一致性过滤。
    bool apply(RegistrationContext& ctx) override;

private:
    /// 是否启用该过滤器。
    bool _enabled = true;
};

} // namespace ir

