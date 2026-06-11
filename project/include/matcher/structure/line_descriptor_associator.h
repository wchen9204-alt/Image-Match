#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include "interfaces/i_structure_associator.h"

namespace ir {

// 浮点型线描述子接口，供关联器复用。
bool computeMsld(const cv::Mat& gray,
                 const std::vector<cv::Vec4i>& lines,
                 cv::Mat& descriptors,
                 int bandWidth,
                 int strips,
                 std::string& message);

bool computeLineSift(const cv::Mat& gray,
                     const std::vector<cv::Vec4i>& lines,
                     cv::Mat& descriptors,
                     int bandWidth,
                     int strips,
                     int bands,
                     int bins,
                     std::string& message);

class LineDescriptorAssociator : public IStructureAssociator {
public:
    explicit LineDescriptorAssociator(const YAML::Node& cfg);
    std::string name() const override { return "LINE_DESCRIPTOR"; }
    bool associate(RegistrationContext& ctx) override;

private:
    std::string _descriptor = "LBD";
    std::string _matcher = "BF";
    std::string _matchMode = "KNN";
    int _knnK = 2;
    float _matchRadius = 50.0f;
    int _minMatches = 2;

    bool _geometricFilter = true;
    double _angleThresholdDeg = 30.0;
    double _minLengthRatio = 0.30;
    double _shiftConsistencyThreshold = 30.0;

    int _msldBandWidth = 12;
    int _msldStrips = 9;

    int _lineSiftBandWidth = 20;
    int _lineSiftStrips = 4;
    int _lineSiftBands = 4;
    int _lineSiftBins = 8;
};

} // namespace ir
