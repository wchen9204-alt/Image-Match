#include "matcher/structure/contour_descriptor_associator.h"

#include <algorithm>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "core/types.h"
#include "matcher/structure/contour_descriptor_helpers.h"
#include "utils/logger.h"
#include "utils/string_utils.h"
#include "utils/yaml_utils.h"

namespace ir {

// 读取 contour descriptor 关联器配置，并缓存不同描述子与几何筛选参数。
ContourDescriptorAssociator::ContourDescriptorAssociator(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    _descriptor = yaml_utils::getString(params, "descriptor", "HU");
    _matcher = yaml_utils::getString(params, "matcher", "BF");
    _matchMode = yaml_utils::getString(params, "match_mode", "KNN");
    _knnK = yaml_utils::getInt(params, "knn_k", 2);
    _matchRadius = yaml_utils::getFloat(params, "match_radius", 50.0f);
    _minMatches = yaml_utils::getInt(params, "min_matches", 2);
    _geometricFilter = yaml_utils::getBool(params, "geometric_filter", true);
    _areaRatioMin = yaml_utils::getDouble(params, "area_ratio_min", 0.30);
    _shiftConsistencyThreshold =
        yaml_utils::getDouble(params, "shift_consistency_threshold", 30.0);
    _geometricModel = yaml_utils::getString(params, "geometric_model", "RIGID");
    _rigidRansacThreshold = yaml_utils::getDouble(params, "rigid_ransac_threshold", 8.0);
    _rigidRansacIterations = yaml_utils::getInt(params, "rigid_ransac_iterations", 200);
    _rigidMinInliers = yaml_utils::getInt(params, "rigid_min_inliers", 2);
    _fourierSamplePoints = yaml_utils::getInt(params, "fourier_sample_points", 128);
    _fourierCoefficients = yaml_utils::getInt(params, "fourier_coefficients", 16);
    _efdHarmonics = yaml_utils::getInt(params, "efd_harmonics", 12);
    _efdNormalizeRotation = yaml_utils::getBool(params, "efd_normalize_rotation", true);
    _efdNormalizeScale = yaml_utils::getBool(params, "efd_normalize_scale", true);
    _shapeContextSamplePoints = yaml_utils::getInt(params, "shape_context_sample_points", 64);
    _shapeContextRadialBins = yaml_utils::getInt(params, "shape_context_radial_bins", 5);
    _shapeContextAngularBins = yaml_utils::getInt(params, "shape_context_angular_bins", 12);
    _shapeContextInnerRadius =
        yaml_utils::getFloat(params, "shape_context_inner_radius", 0.125f);
    _shapeContextOuterRadius =
        yaml_utils::getFloat(params, "shape_context_outer_radius", 2.0f);

    IR_LOG_INFO("ContourDescriptorAssociator: descriptor=", _descriptor,
                ", matcher=", _matcher,
                ", mode=", _matchMode,
                ", knnK=", _knnK,
                ", minMatches=", _minMatches,
                ", geometricFilter=", _geometricFilter,
                ", geometricModel=", _geometricModel,
                ", areaRatioMin=", _areaRatioMin,
                ", shiftThr=", _shiftConsistencyThreshold,
                ", rigidThr=", _rigidRansacThreshold);
    if (string_utils::toUpperAscii(_descriptor) == "FD") {
        IR_LOG_INFO("ContourDescriptorAssociator FourierDescriptor: samplePoints=",
                    _fourierSamplePoints,
                    ", coefficients=", _fourierCoefficients);
    }
    if (string_utils::toUpperAscii(_descriptor) == "EFD") {
        IR_LOG_INFO("ContourDescriptorAssociator EFD: harmonics=",
                    _efdHarmonics,
                    ", normalizeRotation=", _efdNormalizeRotation,
                    ", normalizeScale=", _efdNormalizeScale);
    }
    if (string_utils::toUpperAscii(_descriptor) == "SHAPE_CONTEXT") {
        IR_LOG_INFO("ContourDescriptorAssociator ShapeContext: samplePoints=",
                    _shapeContextSamplePoints,
                    ", radialBins=", _shapeContextRadialBins,
                    ", angularBins=", _shapeContextAngularBins,
                    ", innerRadius=", _shapeContextInnerRadius,
                    ", outerRadius=", _shapeContextOuterRadius);
    }
}

// 执行轮廓描述子关联：
// 1. 按配置计算源/目标轮廓描述子或 matchShapes 距离。
// 2. 执行 BF 匹配并展开候选集合。
// 3. 做几何一致性筛选，得到最终一对一轮廓对应。
// 4. 回写匹配结果、平均平移和调试信息。
bool ContourDescriptorAssociator::associate(RegistrationContext& ctx) {
    auto& md = ctx.structure_match_data;
    md.clear();
    md.method = name();

    if (ctx.structure_data.type != StructureType::CONTOUR) {
        md.message = "ContourDescriptorAssociator requires CONTOUR structure data";
        return false;
    }

    const auto& srcContours = ctx.structure_data.first.contours;
    const auto& dstContours = ctx.structure_data.second.contours;
    IR_LOG_INFO("ContourDescriptorAssociator input: srcContours=", srcContours.size(),
                ", dstContours=", dstContours.size());

    const std::string descriptor = string_utils::toUpperAscii(_descriptor);
    const std::string mode = string_utils::toUpperAscii(_matchMode);

    // 先根据配置构建轮廓描述子，并生成原始 KNN/RADIUS/MATCH 候选。
    if (descriptor == "HU" || descriptor == "SHAPE_CONTEXT" || descriptor == "FD" ||
        descriptor == "EFD") {
        cv::Mat srcDesc;
        cv::Mat dstDesc;
        std::vector<int> srcIndices;
        std::vector<int> dstIndices;

        bool ok = false;
        if (descriptor == "HU") {
            ok = computeHuMoments(srcContours, srcDesc, md.message, &srcIndices) &&
                 computeHuMoments(dstContours, dstDesc, md.message, &dstIndices);
        } else if (descriptor == "FD") {
            ok = computeFourierDescriptor(srcContours,
                                          srcDesc,
                                          _fourierSamplePoints,
                                          _fourierCoefficients,
                                          md.message,
                                          &srcIndices) &&
                 computeFourierDescriptor(dstContours,
                                          dstDesc,
                                          _fourierSamplePoints,
                                          _fourierCoefficients,
                                          md.message,
                                          &dstIndices);
        } else if (descriptor == "EFD") {
            ok = computeEllipticFourierDescriptor(srcContours,
                                                  srcDesc,
                                                  _efdHarmonics,
                                                  _efdNormalizeRotation,
                                                  _efdNormalizeScale,
                                                  md.message,
                                                  &srcIndices) &&
                 computeEllipticFourierDescriptor(dstContours,
                                                  dstDesc,
                                                  _efdHarmonics,
                                                  _efdNormalizeRotation,
                                                  _efdNormalizeScale,
                                                  md.message,
                                                  &dstIndices);
        } else {
            ok = computeShapeContext(srcContours,
                                     srcDesc,
                                     _shapeContextSamplePoints,
                                     _shapeContextRadialBins,
                                     _shapeContextAngularBins,
                                     _shapeContextInnerRadius,
                                     _shapeContextOuterRadius,
                                     md.message,
                                     &srcIndices) &&
                 computeShapeContext(dstContours,
                                     dstDesc,
                                     _shapeContextSamplePoints,
                                     _shapeContextRadialBins,
                                     _shapeContextAngularBins,
                                     _shapeContextInnerRadius,
                                     _shapeContextOuterRadius,
                                     md.message,
                                     &dstIndices);
        }
        if (!ok) {
            IR_LOG_WARN("ContourDescriptorAssociator ", descriptor, " failed: ", md.message);
            return false;
        }

        md.raw_matches_knn = remapContourMatches(
            matchContourDescriptors(srcDesc, dstDesc, mode, _knnK, _matchRadius),
            srcIndices,
            dstIndices);
    } else if (descriptor == "MATCH_SHAPES") {
        const int keep = mode == "MATCH" ? 1 : std::max(1, _knnK);
        const float radius = mode == "RADIUS" ? std::max(0.0f, _matchRadius) : -1.0f;
        md.raw_matches_knn.reserve(srcContours.size());

        for (size_t qi = 0; qi < srcContours.size(); ++qi) {
            if (srcContours[qi].size() < 3 || cv::contourArea(srcContours[qi]) <= 0.0) {
                continue;
            }
            std::vector<cv::DMatch> candidates;
            for (size_t ti = 0; ti < dstContours.size(); ++ti) {
                if (dstContours[ti].size() < 3 || cv::contourArea(dstContours[ti]) <= 0.0) {
                    continue;
                }
                const double d = cv::matchShapes(srcContours[qi], dstContours[ti],
                                                 cv::CONTOURS_MATCH_I1, 0);
                if (radius > 0.0f && d > radius) {
                    continue;
                }
                candidates.emplace_back(static_cast<int>(qi),
                                        static_cast<int>(ti),
                                        static_cast<float>(d));
            }
            if (candidates.empty()) {
                continue;
            }

            std::sort(candidates.begin(), candidates.end(),
                      [](const cv::DMatch& a, const cv::DMatch& b) {
                          return a.distance < b.distance;
                      });
            const size_t n = std::min(static_cast<size_t>(keep), candidates.size());
            md.raw_matches_knn.emplace_back(candidates.begin(), candidates.begin() + n);
        }
    } else {
        md.message = "unsupported descriptor: " + _descriptor;
        return false;
    }

    IR_LOG_INFO("ContourDescriptorAssociator KNN: raw groups=", md.raw_matches_knn.size());
    if (md.raw_matches_knn.empty()) {
        md.message = "matching produced no matches";
        return false;
    }

    // 将分组候选打平，便于后续统一做一对一约束和几何一致性筛选。
    std::vector<cv::DMatch> rawFlat;
    for (const auto& nb : md.raw_matches_knn) {
        for (const auto& m : nb) {
            rawFlat.push_back(m);
        }
    }

    std::vector<cv::DMatch> selected;
    cv::Mat estimatedAffine;
    if (_geometricFilter) {
        selected = filterContourGeometric(rawFlat,
                                          srcContours,
                                          dstContours,
                                          _areaRatioMin,
                                          _geometricModel,
                                          _shiftConsistencyThreshold,
                                          _rigidRansacThreshold,
                                          _rigidRansacIterations,
                                          _rigidMinInliers,
                                          &estimatedAffine);
    } else {
        selected = filterUniqueByDistance(rawFlat, srcContours.size(), dstContours.size());
    }

    md.filtered_matches = selected;
    md.line_matches = selected;
    md.inlier_line_matches = selected;
    md.valid = static_cast<int>(selected.size()) >= _minMatches;
    md.score = srcContours.empty()
                   ? 0.0
                   : static_cast<double>(selected.size()) /
                         static_cast<double>(srcContours.size());

    if (!md.valid) {
        md.message = "not enough contour matches: " + std::to_string(selected.size());
        IR_LOG_WARN("ContourDescriptorAssociator rejected: ", md.message);
        return false;
    }

    // RIGID 模式若已估计出刚体矩阵，直接沿用该 affine；
    // 其他情况继续回退到平均质心位移。
    if (!estimatedAffine.empty() && estimatedAffine.rows == 2 && estimatedAffine.cols == 3) {
        estimatedAffine.convertTo(md.affine, CV_64F);
        md.translation = {md.affine.at<double>(0, 2), md.affine.at<double>(1, 2)};
    } else {
        cv::Point2d sumShift(0.0, 0.0);
        int shiftCount = 0;
        for (const auto& m : selected) {
            const cv::Point2d c1 = contourCentroid(srcContours[static_cast<size_t>(m.queryIdx)]);
            const cv::Point2d c2 = contourCentroid(dstContours[static_cast<size_t>(m.trainIdx)]);
            sumShift += (c2 - c1);
            ++shiftCount;
        }
        if (shiftCount > 0) {
            md.translation = {sumShift.x / shiftCount, sumShift.y / shiftCount};
            md.affine = (cv::Mat_<double>(2, 3) << 1.0,
                         0.0,
                         md.translation.x,
                         0.0,
                         1.0,
                         md.translation.y);
        }
    }

    IR_LOG_INFO("ContourDescriptorAssociator produced ", descriptor, " matches: ",
                selected.size(), " / ", rawFlat.size(), " raw, translation=",
                md.translation);
    return true;
}

} // namespace ir
