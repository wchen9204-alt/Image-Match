#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_keypoint_extractor.h"

namespace ir {

/// BRISK ��������ȡ����
class BriskExtractor : public IKeypointExtractor {
public:
    /// ���� YAML ��ʼ�� BRISK ������
    explicit BriskExtractor(const YAML::Node& cfg);

    std::string name() const override { return "BRISK"; }
    KeypointType type() const override { return KeypointType::BRISK; }
    NormType normType() const override { return _norm; }

    /// ���ؼ������������ȡ���д����׼�����ġ�
    bool extract(RegistrationContext& ctx) override;

private:
    int _thresh = 30;
    int _octaves = 3;
    float _patternScale = 1.0f;

    NormType _norm = NormType::HAMMING;
    cv::Ptr<cv::BRISK> _impl;
};

} // namespace ir
