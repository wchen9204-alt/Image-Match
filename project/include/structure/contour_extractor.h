#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

#include "interfaces/i_structure_extractor.h"

namespace ir {

class ContourExtractor : public IStructureExtractor {
public:
    explicit ContourExtractor(const YAML::Node& cfg);

    std::string name() const override { return "CONTOUR"; }
    StructureType type() const override { return StructureType::CONTOUR; }
    bool extract(RegistrationContext& ctx) override;

private:
    int _blurKernel = 0;
    bool _autoCanny = false;
    double _cannyThreshold1 = 50.0;
    double _cannyThreshold2 = 150.0;
    int _apertureSize = 3;
    std::string _retrievalMode = "EXTERNAL";
    std::string _chainApprox = "SIMPLE";
    double _minArea = 20.0;
    double _minPerimeter = 0.0;
    int _minPoints = 3;
    int _minBboxWidth = 0;
    int _minBboxHeight = 0;
    double _minExtent = 0.0;
    double _maxAspectRatio = 0.0;
    int _maxContours = 1000;
    int _contourThickness = 1;
};

} // namespace ir

