#pragma once

#include <yaml-cpp/yaml.h>

#include "interfaces/i_structure_extractor.h"

namespace ir {

enum class LineDetectorType { HOUGH_LINES, HOUGH_LINES_P, LSD, FLD };

class LineExtractor : public IStructureExtractor {
public:
    explicit LineExtractor(const YAML::Node& cfg);

    std::string name() const override { return "LINE"; }
    std::string outputLabel() const override;
    StructureType type() const override { return StructureType::LINE; }
    bool extract(RegistrationContext& ctx) override;

private:
    LineDetectorType _method = LineDetectorType::HOUGH_LINES_P;
    double _cannyThreshold1 = 50.0;
    double _cannyThreshold2 = 150.0;
    int _apertureSize = 3;
    double _rho = 1.0;
    double _thetaDegrees = 1.0;
    int _threshold = 50;
    int _maxLines = 300;
    double _minLineLength = 30.0;
    double _maxLineGap = 10.0;
    int _lineThickness = 2;
    bool _deduplicateLines = true;
    double _duplicateAngleDeg = 3.0;
    double _duplicateDistance = 8.0;

    int _lsdRefine = 1;
    double _lsdScale = 0.8;
    double _lsdSigmaScale = 0.6;
    double _lsdQuant = 2.0;
    double _lsdAngTh = 22.5;
    double _lsdLogEps = 0.0;
    double _lsdDensityTh = 0.7;
    int _lsdNBins = 1024;
    int _lsdDetectorScale = 2;
    int _lsdDetectorNumOctaves = 2;

    int _fldLengthThreshold = 10;
    double _fldDistanceThreshold = 1.414213562;
    double _fldCannyThreshold1 = 50.0;
    double _fldCannyThreshold2 = 50.0;
    int _fldCannyApertureSize = 3;
    bool _fldDoMerge = false;
};

} // namespace ir

