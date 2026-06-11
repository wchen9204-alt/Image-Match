#pragma once

#include <yaml-cpp/yaml.h>

#include "core/types.h"
#include "interfaces/i_filter.h"

namespace ir {

/// 点特征法专用的双向一致性过滤器。
///
/// 该过滤器通常用于暴力匹配之后，用来剔除只在单向成立、
/// 但不满足双向一致性的匹配。结构法和直接法当前不使用该过滤器。
class CrossCheckFilter : public IFilter {
public:
    /// 根据配置构造过滤器。
    explicit CrossCheckFilter(const YAML::Node& cfg);

    /// 返回日志和诊断中使用的显示名称。
    std::string name() const override { return "CrossCheck"; }

    /// 对点特征匹配结果执行双向一致性过滤。
    bool apply(RegistrationContext& ctx) override;

private:
    /// 是否启用该过滤器。
    bool _enabled = true;
};

}
