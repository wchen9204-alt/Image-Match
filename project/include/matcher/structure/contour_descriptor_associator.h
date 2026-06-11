#pragma once

#include <string>

#include <yaml-cpp/yaml.h>

#include "interfaces/i_structure_associator.h"

namespace ir {

class ContourDescriptorAssociator : public IStructureAssociator {
public:
    explicit ContourDescriptorAssociator(const YAML::Node& cfg);
    std::string name() const override { return "CONTOUR_DESCRIPTOR"; }
    bool associate(RegistrationContext& ctx) override;

private:
    std::string _descriptor = "HU";
    std::string _matcher = "BF";
    std::string _matchMode = "KNN";
    int _knnK = 2;
    float _matchRadius = 50.0f;
    int _minMatches = 2;

    bool _geometricFilter = true;
    std::string _geometricModel = "RIGID";
    double _areaRatioMin = 0.30;
    double _shiftConsistencyThreshold = 30.0;
    double _rigidRansacThreshold = 8.0;
    int _rigidRansacIterations = 200;
    int _rigidMinInliers = 2;

    int _fourierSamplePoints = 128;
    int _fourierCoefficients = 16;

    int _efdHarmonics = 12;
    bool _efdNormalizeRotation = true;
    bool _efdNormalizeScale = true;

    int _shapeContextSamplePoints = 64;
    int _shapeContextRadialBins = 5;
    int _shapeContextAngularBins = 12;
    float _shapeContextInnerRadius = 0.125f;
    float _shapeContextOuterRadius = 2.0f;
};

} // namespace ir
