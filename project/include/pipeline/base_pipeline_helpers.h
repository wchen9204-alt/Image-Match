#pragma once

#include <filesystem>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "core/config.h"
#include "core/context.h"

namespace ir::base_pipeline_helpers {

/// 读取 pipeline 输入图像，同时输出显示用 BGR 图和算法用 8 位灰度图。
bool loadImageForPipeline(const std::filesystem::path& path, cv::Mat& color, cv::Mat& gray);

/// 根据灰度阈值生成前景 mask；黑色背景不会计入后续覆盖率统计。
bool buildForegroundMask(const cv::Mat& image, int thresholdValue, cv::Mat& mask);

/// 计算 warped 与 target 在 overlapMask 区域内的归一化平均绝对灰度差。
double computePhotometricError(const cv::Mat& warped,
                               const cv::Mat& target,
                               const cv::Mat& overlapMask);

/// 计算重叠区域内 warped source 与 target 的边缘 IoU，用于发现覆盖充分但内容错位的结果。
double computeEdgeAlignmentIou(const cv::Mat& warped,
                               const cv::Mat& target,
                               const cv::Mat& overlapMask,
                               int cannyLowThreshold,
                               int cannyHighThreshold,
                               int dilateSize,
                               int minEdgePixels);

/// 计算两个前景 mask 的交并比，主要用于结构重叠验证。
double computeMaskIou(const cv::Mat& a, const cv::Mat& b);

/// 计算 warped source 与 target 的局部包含率，支持一张图是另一张图局部的场景。
double computeMaskLocalContainment(const cv::Mat& sourceMask,
                                   const cv::Mat& warpedSourceMask,
                                   const cv::Mat& targetMask);

/// 计算前景 mask 经过 warp 后仍落在目标画布内的比例。
double computeMaskCoverage(const cv::Mat& originalMask, const cv::Mat& warpedMask);

/// 统计 keypoint/learning 等离散内点在 source/target 前景包围盒中的空间覆盖率。
double computeInlierSpatialCoverage(const std::vector<cv::KeyPoint>& sourceKeypoints,
                                    const std::vector<cv::KeyPoint>& targetKeypoints,
                                    const std::vector<cv::DMatch>& inlierMatches,
                                    const cv::Mat& sourceMask,
                                    const cv::Mat& targetMask,
                                    double& sourceCoverage,
                                    double& targetCoverage);

/// 统计直接法点对在 source/target 前景包围盒中的空间覆盖率。
double computePointSpatialCoverage(const std::vector<cv::Point2f>& sourcePoints,
                                   const std::vector<cv::Point2f>& targetPoints,
                                   const cv::Mat& sourceMask,
                                   const cv::Mat& targetMask,
                                   double& sourceCoverage,
                                   double& targetCoverage);

/// 从当前上下文中提取可用于 warp 的 2D 变换矩阵。
bool activeTransformMatrix(const RegistrationContext& ctx, cv::Mat& matrix);

/// 按需对二值 mask 做形态学膨胀，增强细线或稀疏结构的重叠稳定性。
void dilateMaskIfRequested(cv::Mat& mask, int dilateSize);

/// 将二值 mask 按当前 2D 变换 warp 到指定画布尺寸。
bool warpMaskToTargetSize(const cv::Mat& sourceMask,
                          const cv::Size& targetSize,
                          const cv::Mat& matrix,
                          cv::Mat& warpedMask);

/// 计算当前 2D 变换矩阵的逆矩阵，支持 2x3 仿射和 3x3 透视矩阵。
bool invertTransformMatrix(const cv::Mat& matrix, cv::Mat& inverseMatrix);

/// 使用上下文里的生效变换，将结构 mask warp 到目标画布。
bool warpStructureMask(const RegistrationContext& ctx,
                       const cv::Mat& sourceMask,
                       const cv::Size& targetSize,
                       cv::Mat& warpedMask);

/// 构建伪彩色重叠图：warped source 为红色通道，target 为绿色通道。
bool buildFalseColorOverlay(const cv::Mat& warped,
                            const cv::Mat& target,
                            int foregroundThreshold,
                            cv::Mat& overlay);

} // namespace ir::base_pipeline_helpers
