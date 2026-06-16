#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_structure_associator.h"

namespace ir {

class PhaseCorrelateAssociator : public IStructureAssociator {
public:
    explicit PhaseCorrelateAssociator(const YAML::Node& cfg);

    std::string name() const override { return "PHASE_CORRELATE"; }
    bool associate(RegistrationContext& ctx) override;

private:
    double _scoreThreshold = 0.01;
    int _blurKernel = 5;
};

} // namespace ir

