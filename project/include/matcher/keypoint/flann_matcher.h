#pragma once

#include <yaml-cpp/yaml.h>

#include "core/types.h"
#include "interfaces/i_matcher.h"

namespace ir {

/// 基于 FLANN 的点特征匹配器，可通过 YAML 选择具体的 OpenCV 匹配接口。
class FlannMatcher : public IMatcher {
public:
    /// 根据 YAML 配置初始化匹配器参数。
    explicit FlannMatcher(const YAML::Node& cfg);

    /// 返回匹配器名称。
    std::string name() const override { return "FlannMatcher"; }

    /// 执行 FLANN 匹配，并将结果写入配准上下文。
    bool match(RegistrationContext& ctx) override;

private:
    NormType _normType = NormType::UNKNOWN;
    MatchMethod _method = MatchMethod::KNN;
    int _knnK = 2;
    float _radius = 100.0f;

    int _kdTrees = 5;

    int _lshTableNumber = 12;
    int _lshKeySize = 20;
    int _lshMultiProbeLevel = 2;

    int _searchChecks = 50;
    float _searchEps = 0.0f;
    bool _searchSorted = true;
};

} // namespace ir
