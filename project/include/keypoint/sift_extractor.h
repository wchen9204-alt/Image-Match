#pragma once

#include <opencv2/features2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_keypoint_extractor.h"

namespace ir {

/// SIFT ��������ȡ����
class SiftExtractor : public IKeypointExtractor {
public:
    /// ���� YAML ��ʼ�� SIFT ������
    explicit SiftExtractor(const YAML::Node& cfg);

    std::string name() const override { return "SIFT"; }
    KeypointType type() const override { return KeypointType::SIFT; }
    NormType normType() const override { return _norm; }

    /// ���ؼ������������ȡ���д����׼�����ġ�
    bool extract(RegistrationContext& ctx) override;

private:
    int _nfeatures = 0;
    int _nOctaveLayers = 3;
    double _contrastThreshold = 0.04;
    double _edgeThreshold = 10.0;
    double _sigma = 1.6;

    NormType _norm = NormType::L2;
    cv::Ptr<cv::SIFT> _impl;
};

} // namespace ir
