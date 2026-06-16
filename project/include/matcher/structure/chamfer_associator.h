#pragma once

#include <yaml-cpp/yaml.h>

#include <string>

#include "interfaces/i_structure_associator.h"

namespace ir {

class ChamferAssociator : public IStructureAssociator {
public:
    explicit ChamferAssociator(const YAML::Node& cfg);

    std::string name() const override { return "CHAMFER"; }
    bool associate(RegistrationContext& ctx) override;

private:
    int _searchRadius = 20;
    int _step = 1;
    int _maxPoints = 2000;
    int _phaseBlurKernel = 5;
    double _scoreThreshold = 0.25;
    bool _bidirectional = true;
    std::string _initialization = "PHASE_CORRELATE";
};

} // namespace ir

