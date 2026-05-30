#pragma once

#include <memory>

#include <yaml-cpp/yaml.h>

#include "interfaces/i_feature_extractor.h"
#include "interfaces/i_filter.h"
#include "interfaces/i_geometry_estimator.h"
#include "interfaces/i_matcher.h"
#include "interfaces/i_structure_extractor.h"

namespace ir {

/// 根据 YAML 配置创建具体的配准组件。
///
/// Factory 统一管理 YAML 中 `type` 字段到具体实现的映射，包括点特征提取器、
/// 结构特征提取器、匹配器、过滤器以及几何估计器。
class Factory {
public:
    /// 创建点特征提取器，例如 SIFT、SURF、ORB、BRISK、KAZE 或 AKAZE。
    static std::shared_ptr<IFeatureExtractor> createFeatureExtractor(const YAML::Node& cfg);

    /// 创建结构特征提取器，例如边缘、直线或轮廓。
    static std::shared_ptr<IStructureExtractor> createStructureExtractor(const YAML::Node& cfg);

    /// 创建描述子匹配器，例如 BFMatcher 或 FlannMatcher。
    static std::shared_ptr<IMatcher> createMatcher(const YAML::Node& cfg);

    /// 创建匹配过滤器，例如 ratio test、cross-check 或 GMS。
    static std::shared_ptr<IFilter> createFilter(const YAML::Node& cfg);

    /// 创建几何估计器，例如单应、仿射、刚体或相似变换估计器。
    static std::shared_ptr<IGeometryEstimator> createGeometryEstimator(const YAML::Node& cfg);
};

} // namespace ir
