#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

#include "interfaces/i_structure_associator.h"

namespace ir {

class IcpAssociator : public IStructureAssociator {
public:
    explicit IcpAssociator(const YAML::Node& cfg);

    std::string name() const override { return "ICP"; }
    bool associate(RegistrationContext& ctx) override;

private:
    int _maxPoints = 2000;
    int _maxIterations = 30;
    int _minCorrespondences = 20;
    double _maxCorrespondenceDistance = 10.0;
    double _tolerance = 0.01;
    double _scoreThreshold = 2.0;
    std::string _initialization = "CENTROID";
};

} // namespace ir
