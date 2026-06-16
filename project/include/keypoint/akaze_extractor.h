#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_keypoint_extractor.h"

namespace ir {

/// AKAZE ������ȡ����
class AkazeExtractor : public IKeypointExtractor {
public:
    /// ���� YAML ���ó�ʼ�� AKAZE ������
    explicit AkazeExtractor(const YAML::Node& cfg);

    std::string name() const override { return "AKAZE"; }
    KeypointType type() const override { return KeypointType::AKAZE; }
    /// AKAZE Ĭ��ʹ�ö����������ӣ�ͨ����Ӧ Hamming ���롣
    NormType normType() const override { return _norm; }

    /// ������������ȡ AKAZE �ؼ���������ӡ�
    bool extract(RegistrationContext& ctx) override;

private:
    int _descriptorType = static_cast<int>(cv::AKAZE::DESCRIPTOR_MLDB);
    int _descriptorSize = 0;
    int _descriptorChannels = 3;
    float _threshold = 0.001f;
    int _nOctaves = 4;
    int _nOctaveLayers = 4;
    int _diffusivity = static_cast<int>(cv::KAZE::DIFF_PM_G2);

    NormType _norm = NormType::HAMMING;
    cv::Ptr<cv::AKAZE> _impl;
};

} // namespace ir

