#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_structure_associator.h"

namespace ir {

/// 线段级结构关联器。
///
/// 用于直线结构法：根据 source / target 中提取出的线段，先按方向、长度和中心位移
/// 生成候选线段匹配。它是几何 baseline 匹配器，不负责最终 RANSAC 几何模型估计。
class LineSegmentAssociator : public IStructureAssociator {
public:
    /// 从 `association.params.line_segment` 或兼容节点中读取线段匹配参数。
    explicit LineSegmentAssociator(const YAML::Node& cfg);

    /// 返回结构关联方法名称，用于日志和输出文件命名。
    std::string name() const override { return "LINE_SEGMENT"; }

    /// 执行线段级 baseline 关联，并写回 `StructureMatchData` 中的候选匹配。
    bool associate(RegistrationContext& ctx) override;

private:
    /// 候选线段方向差阈值，单位为度；线段方向按无向直线处理。
    double _angleThresholdDeg = 10.0;

    /// 候选线段长度比例下限，越大表示只接受长度越接近的线段。
    double _minLengthRatio = 0.60;

    /// 候选线段中心位移的最大距离，默认值较大，近似不限制位移范围。
    double _maxShiftDistance = 100000.0;

    /// baseline 预筛选使用的中心位移一致性阈值，单位为像素。
    double _shiftConsistencyThreshold = 8.0;

    /// 进入几何估计前要求的最少候选线段匹配数。
    int _minMatches = 4;

    /// 每条 source 线段最多保留的 target 候选数量，避免重复线段造成候选爆炸。
    int _maxCandidatesPerLine = 5;
};

} // namespace ir
