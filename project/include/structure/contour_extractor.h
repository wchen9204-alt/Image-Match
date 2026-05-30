#pragma once

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
    double _cannyThreshold1 = 50.0;
    double _cannyThreshold2 = 150.0;
    int _apertureSize = 3;
    double _minArea = 20.0;
    int _maxContours = 1000;
    int _contourThickness = 1;
};

} // namespace ir
