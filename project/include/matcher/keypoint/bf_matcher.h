#pragma once

#include <yaml-cpp/yaml.h>

#include "core/types.h"
#include "interfaces/i_matcher.h"

namespace ir {

/// 暴力点特征匹配器，可通过 YAML 选择具体的 OpenCV 匹配接口。
class BfMatcher : public IMatcher {
public:
    /// 根据 YAML 配置初始化匹配器参数。
    explicit BfMatcher(const YAML::Node& cfg);

    /// 返回匹配器名称。
    std::string name() const override { return "BFMatcher"; }

    /// 执行暴力匹配，并将结果写入配准上下文。
    bool match(RegistrationContext& ctx) override;

private:
    NormType _normType = NormType::UNKNOWN;
    MatchMethod _method = MatchMethod::MATCH;
    int _knnK = 2;
    float _radius = 100.0f;
    bool _crossCheck = false;
};

} // namespace ir

