#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_structure_extractor.h"

namespace ir {

class EdgeExtractor : public IStructureExtractor {
public:
    explicit EdgeExtractor(const YAML::Node& cfg);

    std::string name() const override { return "EDGE"; }
    StructureType type() const override { return StructureType::EDGE; }
    bool extract(RegistrationContext& ctx) override;

private:
    double _threshold1 = 50.0;
    double _threshold2 = 150.0;
    int _apertureSize = 3;
    bool _l2Gradient = false;
    int _blurKernel = 0;
    int _dilateIterations = 0;
};

} // namespace ir
