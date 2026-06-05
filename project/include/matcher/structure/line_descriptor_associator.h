#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

#include "interfaces/i_structure_associator.h"

namespace ir {

/// 线描述子结构关联器。
///
/// 当前优先支持 OpenCV contrib line_descriptor 模块中的 LBD。关联器只复用上游线提取器
/// 输出的线段来计算描述子并匹配（输出原始 KNN 到 raw_matches_knn）。
///
/// 关联器内部保留线特有的几何一致性筛选（方向/长度/中心位移投票），
/// 因为它依赖线段几何属性，不适合作为通用 IFilter。
/// Ratio test / 距离过滤等通用描述子过滤交给外部 IFilter 链。
class LineDescriptorAssociator : public IStructureAssociator {
public:
    /// 从 `association.params.line_descriptor` 或兼容节点中读取参数。
    explicit LineDescriptorAssociator(const YAML::Node& cfg);

    /// 返回结构关联方法名称，用于日志和输出文件命名。
    std::string name() const override { return "LINE_DESCRIPTOR"; }

    /// 执行 LBD 描述子计算、KNN 匹配与几何一致性筛选，
    /// 输出 raw_matches_knn + filtered_matches 到 StructureMatchData。
    bool associate(RegistrationContext& ctx) override;

private:
    /// 当前描述子名称；首个实现为 LBD。
    std::string _descriptor = "LBD";

    /// 每条源线段的 KNN 邻居数量。
    int _knnK = 8;

    /// 进入后续阶段前要求的最少匹配数。
    int _minMatches = 2;

    // ---- 几何一致性筛选参数（线匹配特有） ----

    /// 是否启用线段几何一致性筛选。
    bool _geometricFilter = true;

    /// 线段方向差阈值，单位度。
    double _angleThresholdDeg = 30.0;

    /// 线段长度比例下限。
    double _minLengthRatio = 0.30;

    /// 中心位移一致性阈值，单位像素。
    double _shiftConsistencyThreshold = 30.0;
};

} // namespace ir
