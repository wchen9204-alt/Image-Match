#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_geometry_estimator.h"

namespace ir {

/// 多层暗部专用刚体估计器。
///
/// 根据配置选择关键点方向投票或点对几何投票，随后对候选簇执行严格刚体RANSAC，并将评分最高的有效模型写回注册上下文。
class MultilayerDarkRigidEstimator final : public IGeometryEstimator {
public:
    /// 从 YAML 配置读取投票、RANSAC 和候选簇筛选参数。
    explicit MultilayerDarkRigidEstimator(const YAML::Node& cfg);

    /// 返回多层暗部刚体估计器的名称。
    std::string name() const override { return "Rigid2D_DARK_MULTILAYER"; }

    /// 返回当前估计器支持的几何模型类型。
    GeometryType type() const override { return GeometryType::RIGID; }

    /// 根据上下文中的匹配结果估计刚体变换，并写回几何结果和内点信息。
    bool estimate(RegistrationContext& ctx) override;

private:
    /// 一个投票簇执行严格刚体 RANSAC 所需的最少内点数。
    int _min_inliers = 6;
    /// 旋转投票箱宽度，单位为度。
    double _rotation_bin_degrees = 5.0;
    /// 平移投票箱宽度，单位为像素。
    double _translation_bin_size = 10.0;
    /// RANSAC 内点判定使用的最大重投影误差，单位为像素。
    double _reprojection_threshold = 5.0;
    /// 严格刚体 RANSAC 的最大迭代次数。
    int _ransac_iterations = 1000;
    /// 参与后续 RANSAC 和评分的最大投票簇数量。
    int _max_clusters = 5;
    /// 投票方式：KEYPOINT_ANGLE 或 POINT_PAIR。
    std::string _voting_method = "KEYPOINT_ANGLE";
    /// 点对长度匹配时允许的最大绝对误差，单位为像素。
    double _point_pair_distance_tolerance = 3.0;
    /// 每个外层点对最多尝试的目标点对候选数量。
    int _point_pair_max_candidates = 64;
    /// 点对最小距离相对于两幅图前景包围盒宽度的比例。
    double _point_pair_min_distance_ratio = 0.5;
    /// 同一几何点对两端独立计算的平移分量允许的最大差值，单位为像素。
    double _point_pair_translation_tolerance = 10.0;
};

} // namespace ir
