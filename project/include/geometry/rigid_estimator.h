#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

#include "interfaces/i_geometry_estimator.h"

namespace ir {

/// 刚体变换估计器。
class RigidEstimator : public IGeometryEstimator {
public:
    /// 从 YAML 配置初始化刚体估计器参数。
    explicit RigidEstimator(const YAML::Node& cfg);

    /// 返回估计器名称。
    std::string name() const override { return "Rigid2D"; }

    /// 返回当前估计器支持的几何类型。
    GeometryType type() const override { return GeometryType::RIGID; }

    /// 估计 2x3 刚体变换，并将结果写回上下文。
    bool estimate(RegistrationContext& ctx) override;

private:
    /// OpenCV 刚体/部分仿射估计方法编号；当前默认对应 RANSAC。
    int _method = 8;
    /// RANSAC 内点判定使用的最大重投影误差阈值，单位像素。
    double _ransacReprojThreshold = 3.0;
    /// RANSAC 最大迭代次数。
    int _maxIters = 2000;
    /// RANSAC 期望置信度。
    double _confidence = 0.99;
    /// OpenCV 估计完成后额外局部优化的迭代次数。
    int _refineIters = 10;
    /// 最终模型至少需要保留的内点数。
    int _minInliers = 3;
    /// 刚体估计后端；例如 OPENCV_PARTIAL_AFFINE。
    std::string _estimatorBackend = "OPENCV_PARTIAL_AFFINE";
    /// 刚体模型细化方式；当前仅保留 SVD 和 NONE。
    std::string _rigidRefineMode = "SVD";
    /// 是否从 filtered matches 中额外生成多候选 rigid 模型。
    bool _enableFilteredMatchCandidates = false;
    /// 生成候选时，按排序优先参与抽样的前 Top-K 个匹配数。
    int _filteredMatchCandidateTopK = 24;
    /// 候选生成阶段可使用的匹配池大小上限。
    int _filteredMatchCandidatePoolSize = 64;
    /// 计划生成的 filtered-match 候选模型数量上限。
    int _filteredMatchCandidateCount = 12;
    /// 生成候选点对时，两点之间要求的最小间距，避免退化小对。
    double _filteredMatchCandidateMinPairDistance = 20.0;
    /// filtered seed 两点组合的调度策略，支持 COVERAGE_FIRST / LEGACY_LEXICOGRAPHIC。
    std::string _filteredMatchCandidatePairStrategy = "COVERAGE_FIRST";
    /// 是否在候选选择阶段加入前景 mask 几何评分。
    bool _enableCandidateMaskScoring = false;
    /// 候选 mask 与边缘 IoU 评分时，将图像二值化为前景 mask 使用的阈值。
    int _candidateMaskForegroundThreshold = 10;
    /// 与最高 containment 相差不超过该值的候选进入边缘 IoU 比较。
    double _candidateContainmentTieMargin = 0.20;
    /// 候选去重时允许的最大旋转角差，单位度。
    double _candidateDedupRotationDiffDeg = 2.0;
    /// 候选去重时允许的最大平移差，单位像素。
    double _candidateDedupTranslationDiff = 3.0;
};

} // namespace ir
