#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/context.h"
#include "interfaces/i_filter.h"
#include "interfaces/i_geometry_estimator.h"
#include "interfaces/i_keypoint_extractor.h"
#include "interfaces/i_matcher.h"

namespace ir {

/// 直接法前置点特征初始化器。
///
/// 它只负责给 ECC/ESM 这类需要初始值的直接法提供 source -> target 初始矩阵；
/// 是否作为最终正确结果，仍由直接法输出和 DirectPipeline 的验证逻辑决定。
class DirectFeatureInitializer {
public:
    DirectFeatureInitializer() = default;

    /// 清空已创建的点特征候选、匹配器、过滤器和几何估计器。
    void reset();

    /// 按 pipeline 配置读取单个点特征候选，并创建共享的 matcher/filter/geometry 链。
    bool configure(const PipelineConfig& cfg);

    /// 返回当前配置是否启用了直接法点特征初始化。
    bool enabled() const { return _enabled; }

    /// 尝试生成可用初始矩阵；通过安全门时写入 ctx.feature_initializer_data。
    bool run(RegistrationContext& ctx);

private:
    struct CandidateStage {
        std::string name;
        std::shared_ptr<IKeypointExtractor> extractor;
    };

    PipelineConfig _config;
    bool _enabled = false;
    bool _has_candidate = false;
    CandidateStage _candidate;
    std::shared_ptr<IMatcher> _matcher;
    std::vector<std::shared_ptr<IFilter>> _filters;
    std::shared_ptr<IGeometryEstimator> _geometry;
};

} // namespace ir
