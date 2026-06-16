#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_keypoint_extractor.h"

namespace ir {

/// KAZE ��������ȡ����
class KazeExtractor : public IKeypointExtractor {
public:
    /// ���� YAML ��ʼ�� KAZE ������
    explicit KazeExtractor(const YAML::Node& cfg);

    std::string name() const override { return "KAZE"; }
    KeypointType type() const override { return KeypointType::KAZE; }
    NormType normType() const override { return _norm; }

    /// ���ؼ������������ȡ���д����׼�����ġ�
    bool extract(RegistrationContext& ctx) override;

private:
    bool _extended = false;
    bool _upright = false;
    float _threshold = 0.001f;
    int _nOctaves = 4;
    int _nOctaveLayers = 4;
    int _diffusivity = static_cast<int>(cv::KAZE::DIFF_PM_G2);

    NormType _norm = NormType::L2;
    cv::Ptr<cv::KAZE> _impl;
};

} // namespace ir
