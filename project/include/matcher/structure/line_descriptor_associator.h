#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#ifdef IR_HAS_OPENCV_LINE_DESCRIPTOR
#include <opencv2/line_descriptor.hpp>
#endif

#include "interfaces/i_structure_associator.h"

namespace ir {

// ============================================================================
// 线描述子计算函数（仅被 LineDescriptorAssociator 使用）
// ============================================================================

// 计算 LBD 二进制描述子。依赖 OpenCV contrib line_descriptor 模块。
bool computeLbd(const cv::Mat& gray,
                std::vector<cv::line_descriptor::KeyLine>& keyLines,
                cv::Mat& descriptors,
                std::string& message);

// 计算 MSLD 浮点描述子。bandWidth: 矩形带宽度  strips: 沿线段子区域数  维数 = strips×2×4
bool computeMsld(const cv::Mat& gray,
                 const std::vector<cv::Vec4i>& lines,
                 cv::Mat& descriptors,
                 int bandWidth,
                 int strips,
                 std::string& message);

// 计算 line-SIFT 浮点描述子。将 SIFT 梯度直方图思想迁移到线段带。
// strips × bands 个子区域，每区 bins 个方向 bin，维数 = strips × bands × bins
bool computeLineSift(const cv::Mat& gray,
                     const std::vector<cv::Vec4i>& lines,
                     cv::Mat& descriptors,
                     int bandWidth, int strips, int bands, int bins,
                     std::string& message);

// ============================================================================
// 线描述子结构关联器
// ============================================================================

class LineDescriptorAssociator : public IStructureAssociator {
public:
    explicit LineDescriptorAssociator(const YAML::Node& cfg);
    std::string name() const override { return "LINE_DESCRIPTOR"; }
    bool associate(RegistrationContext& ctx) override;

private:
    std::string _descriptor = "LBD";
    std::string _matcher = "BF";  // BF / FLANN（仅 float 描述子支持 FLANN）
    int _knnK = 8;
    int _minMatches = 2;

    bool _geometricFilter = true;
    double _angleThresholdDeg = 30.0;
    double _minLengthRatio = 0.30;
    double _shiftConsistencyThreshold = 30.0;

    // MSLD
    int _msldBandWidth = 12;
    int _msldStrips = 9;

    // line-SIFT
    int _lineSiftBandWidth = 20;
    int _lineSiftStrips = 4;
    int _lineSiftBands = 4;
    int _lineSiftBins = 8;
};

} // namespace ir