#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_structure_associator.h"

namespace ir {

/// 线描述子结构关联器。
///
/// 当前优先支持 OpenCV contrib line_descriptor 模块中的 LBD。关联器只负责生成
/// 线段描述子匹配，最终 RANSAC 几何模型仍由 StructurePipeline 的 geometry estimator 负责。
class LineDescriptorAssociator : public IStructureAssociator {
public:
    /// 从 `association.params.line_descriptor` 或兼容节点中读取线描述子匹配参数。
    explicit LineDescriptorAssociator(const YAML::Node& cfg);

    /// 返回结构关联方法名称，用于日志和输出文件命名。
    std::string name() const override { return "LINE_DESCRIPTOR"; }

    /// 执行线描述子匹配，并写回 `StructureMatchData::line_matches`。
    bool associate(RegistrationContext& ctx) override;

private:
    /// 当前描述子名称；首个实现为 LBD。
    std::string _descriptor = "LBD";

    /// 当前匹配器名称；LBD 使用二进制描述子，默认 BF/Hamming。
    std::string _matcher = "BF";

    /// KNN ratio test 阈值。
    double _ratio = 0.75;

    /// 进入几何估计前要求的最少线段匹配数。
    int _minMatches = 4;
};

} // namespace ir
