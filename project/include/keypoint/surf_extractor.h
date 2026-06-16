#pragma once

#include <opencv2/xfeatures2d.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_keypoint_extractor.h"

namespace ir {

/// SURF ��������ȡ����
class SurfExtractor : public IKeypointExtractor {
public:
    /// ���� YAML ��ʼ�� SURF ������
    explicit SurfExtractor(const YAML::Node& cfg);

    std::string name() const override { return "SURF"; }
    KeypointType type() const override { return KeypointType::SURF; }
    NormType normType() const override { return _norm; }

    /// ���ؼ������������ȡ���д����׼�����ġ�
    bool extract(RegistrationContext& ctx) override;

private:
    double _hessianThreshold = 400.0;
    int _nOctaves = 4;
    int _nOctaveLayers = 3;
    bool _extended = false;
    bool _upright = false;

    NormType _norm = NormType::L2;
    cv::Ptr<cv::xfeatures2d::SURF> _impl;
};

} // namespace ir
