#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

#include "interfaces/i_structure_associator.h"

namespace ir {

/// 线描述子结构关联器。
///
/// 当前优先支持 OpenCV contrib line_descriptor 模块中的 LBD。关联器只复用上游线提取器
/// 输出的线段来计算描述子并匹配，不再自行重新检测线段。
class LineDescriptorAssociator : public IStructureAssociator {
public:
    /// 从 `association.params.line_descriptor` 或兼容节点中读取线描述子匹配参数。
    explicit LineDescriptorAssociator(const YAML::Node& cfg);

    /// 返回结构关联方法名称，用于日志和输出文件命名。
    std::string name() const override { return "LINE_DESCRIPTOR"; }

    /// 执行线描述子匹配，并写回 `StructureMatchData::line_matches` 与平移仿射。
    bool associate(RegistrationContext& ctx) override;

private:
    /// 当前描述子名称；首个实现为 LBD。
    std::string _descriptor = "LBD";

    /// KNN ratio test 阈值；接近 1.0 时保留多邻居候选交给几何一致性筛选。
    double _ratio = 0.75;

    /// 每条源线段保留的描述子近邻候选数量。
    int _knnK = 8;

    /// 描述子匹配后的无向线段方向差阈值，单位为度。
    double _angleThresholdDeg = 15.0;

    /// 描述子匹配后的线段长度比例下限。
    double _minLengthRatio = 0.50;

    /// 描述子匹配后的线段中心最大位移，默认近似不限制。
    double _maxShiftDistance = 100000.0;

    /// 描述子匹配后的中心位移一致性阈值，单位为像素。
    double _shiftConsistencyThreshold = 12.0;

    /// 进入仿射估计前要求的最少几何一致线段匹配数。
    int _minMatches = 4;
};

} // namespace ir
