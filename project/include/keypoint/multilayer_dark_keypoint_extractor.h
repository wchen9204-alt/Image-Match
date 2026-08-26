#pragma once

#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "interfaces/i_keypoint_extractor.h"

namespace ir {

/// 在多个暗部灰度层上复用已配置的点特征提取器，并合并各层特征。
class MultilayerDarkKeypointExtractor final : public IKeypointExtractor {
public:
    /// 使用原点特征 YAML 创建层内提取器，保证算法与参数保持一致。
    MultilayerDarkKeypointExtractor(const YAML::Node& keypoint_config,
                                    std::vector<int> thresholds);

    std::string name() const override { return _name; }
    KeypointType type() const override { return _extractor->type(); }
    NormType normType() const override { return _extractor->normType(); }
    cv::Ptr<cv::Feature2D> feature2D() const override { return _extractor->feature2D(); }
    bool extract(RegistrationContext& ctx) override;

private:
    std::shared_ptr<IKeypointExtractor> _extractor;
    YAML::Node _keypoint_config;
    std::string _name;
    std::vector<int> _thresholds;
};

} // namespace ir
