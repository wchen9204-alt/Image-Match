#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "structure/contour_feature.h"

namespace ir {

// 计算单条轮廓的质心，供位姿估计和几何一致性筛选复用。
cv::Point2d contourCentroid(const std::vector<cv::Point>& contour);

// 读取缓存特征中的质心，避免重复计算 moments。
cv::Point2d contourCentroid(const ContourFeature& feature);

// 计算 Hu 矩并按行拼接描述子矩阵。
bool computeHuMoments(const std::vector<std::vector<cv::Point>>& contours,
                      cv::Mat& descriptors,
                      std::string& message,
                      std::vector<int>* contourIndices = nullptr);

// 计算 Fourier Descriptor 并按行拼接描述子矩阵。
bool computeFourierDescriptor(const std::vector<std::vector<cv::Point>>& contours,
                              cv::Mat& descriptors,
                              int samplePoints,
                              int coefficientCount,
                              std::string& message,
                              std::vector<int>* contourIndices = nullptr);

// 计算 Shape Context 并按行拼接描述子矩阵。
bool computeShapeContext(const std::vector<std::vector<cv::Point>>& contours,
                         cv::Mat& descriptors,
                         int samplePoints,
                         int radialBins,
                         int angularBins,
                         float innerRadius,
                         float outerRadius,
                         std::string& message,
                         std::vector<int>* contourIndices = nullptr);

// 计算 Elliptic Fourier Descriptor 并按行拼接描述子矩阵。
bool computeEllipticFourierDescriptor(const std::vector<std::vector<cv::Point>>& contours,
                                      cv::Mat& descriptors,
                                      int harmonics,
                                      bool normalizeRotation,
                                      bool normalizeScale,
                                      std::string& message,
                                      std::vector<int>* contourIndices = nullptr);

// 将描述子矩阵匹配结果映射回原始轮廓索引。
std::vector<std::vector<cv::DMatch>> remapContourMatches(
    const std::vector<std::vector<cv::DMatch>>& rawMatches,
    const std::vector<int>& srcIndices,
    const std::vector<int>& dstIndices);

// 根据配置的匹配模式执行轮廓描述子匹配。
std::vector<std::vector<cv::DMatch>> matchContourDescriptors(const cv::Mat& srcDesc,
                                                             const cv::Mat& dstDesc,
                                                             const std::string& mode,
                                                             int knnK,
                                                             float radius);

// 按距离排序并执行一对一去重。
std::vector<cv::DMatch> filterUniqueByDistance(const std::vector<cv::DMatch>& matches,
                                               size_t srcCount,
                                               size_t dstCount);

// 根据面积比和几何模型保留一致的轮廓候选。
std::vector<cv::DMatch> filterContourGeometric(
    const std::vector<cv::DMatch>& raw,
    const std::vector<std::vector<cv::Point>>& srcContours,
    const std::vector<std::vector<cv::Point>>& dstContours,
    double areaRatioMin,
    const std::string& geometricModel,
    double shiftThreshold,
    double rigidReprojThreshold,
    int rigidRansacIterations,
    int rigidMinInliers,
    cv::Mat* estimatedAffine = nullptr);

} // namespace ir
