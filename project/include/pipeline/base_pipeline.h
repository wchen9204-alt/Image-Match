#pragma once

#include <string>

#include "core/config.h"
#include "evaluator/evaluator.h"
#include "interfaces/i_pipeline.h"

namespace ir {

/// 通用配准流水线骨架，负责组织读图、提取、关联、估计、warp、验证和输出流程。
class BasePipeline : public IPipeline {
public:
    BasePipeline() = default;
    ~BasePipeline() override = default;

    /// 加载 pipeline 配置，重置阶段对象和评测器，并委托子类配置专属阶段组件。
    bool configure(const PipelineConfig& cfg) override;

    /// 执行一次完整配准流程；省略运行时路径时回退到 YAML 配置。
    bool run(RegistrationContext& ctx,
             const PipelineRunOptions& options = PipelineRunOptions{}) override;

    /// 按 visualization 配置显示源图、目标图或 warp 结果窗口。
    bool showWindows(RegistrationContext& ctx) override;

protected:
    /// 子类在重新 configure 前清理已创建的提取器、匹配器、几何估计器等阶段对象。
    virtual void resetStages() {}

    /// 1.子类根据方法族配置创建自身需要的阶段组件。
    virtual bool configureStages(const PipelineConfig& cfg) = 0;

    /// 2.读取输入图像，并准备显示用 BGR 图和算法用灰度图。
    virtual bool loadImages(RegistrationContext& ctx);

    /// 3.子类执行特征、结构或直接法专属的提取阶段。
    virtual bool runExtraction(RegistrationContext& ctx) = 0;

    /// 4.子类执行匹配、关联或直接法占位阶段。
    virtual bool runAssociation(RegistrationContext& ctx) = 0;

    /// 5.子类执行几何估计或直接法配准估计阶段。
    virtual bool runEstimation(RegistrationContext& ctx) = 0;

    /// 6.根据几何估计结果选择 affine/perspective warper，并生成 warped source 图像。
    virtual bool runWarp(RegistrationContext& ctx);

    /// 7.统一执行当前启用的质量验证项，作为最终 success 判定入口。
    virtual bool validateRegistrationQuality(RegistrationContext& ctx);

    /// 7.1 执行方法特有的质量验证，只进入当前方法族真正需要的验证分支。
    virtual bool validateMethodSpecificQuality(RegistrationContext& ctx);

    /// 验证方法自身产物是否充足，例如点特征数量、结构数量和结构候选匹配数量。
    virtual bool validateMethodFeatureQuality(RegistrationContext& ctx);

    /// 7.2 执行各方法族共享的最终图像级验证，例如 overlap / photometric / edge。
    virtual bool validateSharedFinalQuality(RegistrationContext& ctx);

    /// 验证匹配/关联质量，例如内点数、内点比例和重投影误差。
    virtual bool validateMatchQuality(RegistrationContext& ctx);

    /// 验证直接法专属质量信号，例如 ECC/相位相关的 confidence。
    virtual bool validateDirectQuality(RegistrationContext& ctx);

    /// 验证结构响应图在 warp 后是否与目标结构响应图足够重合。
    virtual bool validateStructureOverlap(RegistrationContext& ctx);

    /// 验证 warped source 与 target 的前景重叠和光度误差。
    virtual bool validateWarpQuality(RegistrationContext& ctx);

    /// 8.保存通用输出图像；子类可扩展保存专属可视化。
    virtual bool saveOutputs(RegistrationContext& ctx);

    /// 9.生成当前样本输出文件名前缀，子类可加入算法名称。
    virtual std::string buildOutputStem(const RegistrationContext& ctx) const;

    PipelineConfig _config;

    /// 评估器，用于在流程结束后计算当前启用的评测指标。
    Evaluator _evaluator;
};

} // namespace ir

