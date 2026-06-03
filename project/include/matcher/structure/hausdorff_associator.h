#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_structure_associator.h"

namespace ir {

class HausdorffAssociator : public IStructureAssociator {
public:
    explicit HausdorffAssociator(const YAML::Node& cfg);

    std::string name() const override { return "HAUSDORFF"; }
    bool associate(RegistrationContext& ctx) override;

private:
    int _searchRadius = 20;
    int _step = 1;
    int _maxPoints = 2000;
    double _scoreThreshold = 3.0;
    double _percentile = 0.95;
    bool _bidirectional = true;
};

} // namespace ir
