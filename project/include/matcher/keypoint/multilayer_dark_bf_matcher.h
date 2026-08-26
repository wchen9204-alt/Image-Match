#pragma once

#include <yaml-cpp/yaml.h>

#include "core/types.h"
#include "interfaces/i_matcher.h"

namespace ir {

/// 多层暗部专用 BF KNN 匹配器，保留参考流程的扩展候选匹配。
class MultilayerDarkBfMatcher final : public IMatcher {
public:
    /// 从 bf.yaml 的 multilayer_dark 节点读取 KNN 和扩展参数。
    explicit MultilayerDarkBfMatcher(const YAML::Node& cfg);

    std::string name() const override { return "BFMatcher_DARK_MULTILAYER"; }
    bool match(RegistrationContext& ctx) override;

private:
    NormType _norm_type = NormType::UNKNOWN;
    int _knn_k = 64;
    float _expansion_multiplier = 2.0f;
};

} // namespace ir
