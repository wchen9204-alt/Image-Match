#pragma once

#include <memory>
#include <string>

#include "interfaces/i_geometry_estimator.h"
#include "interfaces/i_learning_matcher.h"
#include "pipeline/base_pipeline.h"

namespace ir {

/// 深度学习配准流水线；第一版通过 Python bridge 产出匹配点并复用几何估计链路。
class LearningPipeline : public BasePipeline {
public:
    LearningPipeline() = default;

    /// 返回流水线名称，用于日志、摘要和工厂默认标识。
    std::string name() const override { return "LearningPipeline"; }

protected:
    /// 清空已创建的学习匹配器和几何估计器。
    void resetStages() override;
    /// 从 PipelineConfig 创建 Python learning matcher 和几何估计器。
    bool configureStages(const PipelineConfig& cfg) override;
    /// 学习方法没有独立 C++ 提取阶段，此处仅维护通用阶段计数。
    bool runExtraction(RegistrationContext& ctx) override;
    /// 执行深度学习匹配，并把模型输出转换为平台统一点对数据。
    bool runAssociation(RegistrationContext& ctx) override;
    /// 使用 LEARNING 对应点来源执行几何估计。
    bool runEstimation(RegistrationContext& ctx) override;
    /// 保存学习方法专属匹配图，并继续保存通用输出。
    bool saveOutputs(RegistrationContext& ctx) override;
    /// 生成包含样本名、学习方法名和几何模型名的输出文件 stem。
    std::string buildOutputStem(const RegistrationContext& ctx) const override;

private:
    /// 深度学习匹配器，第一版通过 Python bridge 调用外部模型。
    std::shared_ptr<ILearningMatcher> _matcher;
    /// 后续几何估计器，复用现有 Homography/Affine/Rigid/Similarity 链路。
    std::shared_ptr<IGeometryEstimator> _geometry;
};

} // namespace ir

