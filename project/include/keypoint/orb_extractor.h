#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_keypoint_extractor.h"

namespace ir {

/// ORB ������ȡ����
class OrbExtractor : public IKeypointExtractor {
public:
    /// ���� YAML ���ó�ʼ�� ORB ������
    explicit OrbExtractor(const YAML::Node& cfg);

    std::string name() const override { return "ORB"; }
    KeypointType type() const override { return KeypointType::ORB; }
    NormType normType() const override { return _norm; }

    /// ������������ȡ ORB �ؼ���������ӡ�
    bool extract(RegistrationContext& ctx) override;

private:
    int _nfeatures = 2000;
    float _scaleFactor = 1.2f;
    int _nlevels = 8;
    int _edgeThreshold = 31;
    int _firstLevel = 0;
    int _wtaK = 2;
    int _scoreType = static_cast<int>(cv::ORB::HARRIS_SCORE);
    int _patchSize = 31;
    int _fastThreshold = 20;

    NormType _norm = NormType::HAMMING;
    cv::Ptr<cv::ORB> _impl;
};

} // namespace ir

