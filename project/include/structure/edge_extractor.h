#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_structure_extractor.h"

namespace ir {

enum class EdgeOperatorType { CANNY, SOBEL, LOG, LAPLACIAN };

class EdgeExtractor : public IStructureExtractor {
public:
    explicit EdgeExtractor(const YAML::Node& cfg);

    std::string name() const override { return "EDGE"; }
    std::string outputLabel() const override;
    StructureType type() const override { return StructureType::EDGE; }
    bool extract(RegistrationContext& ctx) override;

private:
    EdgeOperatorType _operatorType = EdgeOperatorType::CANNY;
    double _threshold1 = 50.0;
    double _threshold2 = 150.0;
    int _apertureSize = 3;
    bool _l2Gradient = false;
    int _blurKernel = 0;
    int _dilateIterations = 0;
    int _kernelSize = 3;
    double _scale = 1.0;
    double _delta = 0.0;
    double _responseThreshold = 50.0;
};

} // namespace ir
