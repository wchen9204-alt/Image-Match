#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_structure_extractor.h"

namespace ir {

class LineExtractor : public IStructureExtractor {
public:
    explicit LineExtractor(const YAML::Node& cfg);

    std::string name() const override { return "LINE"; }
    StructureType type() const override { return StructureType::LINE; }
    bool extract(RegistrationContext& ctx) override;

private:
    double _cannyThreshold1 = 50.0;
    double _cannyThreshold2 = 150.0;
    int _apertureSize = 3;
    double _rho = 1.0;
    double _thetaDegrees = 1.0;
    int _threshold = 50;
    double _minLineLength = 30.0;
    double _maxLineGap = 10.0;
    int _lineThickness = 2;
};

} // namespace ir
