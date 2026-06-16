#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "core/context.h"

namespace ir {

/// 绘制线段结构匹配连线图，保留线段端点和中心连线的结构语义。
cv::Mat renderLineSegmentMatches(const RegistrationContext& ctx,
                                 const std::vector<cv::DMatch>& matches,
                                 int maxMatches);

/// 绘制轮廓结构匹配连线图，使用轮廓描边和质心连线表达对应关系。
cv::Mat renderContourMatches(const RegistrationContext& ctx,
                             const std::vector<cv::DMatch>& matches,
                             int maxMatches);

/// 绘制响应图类结构配准结果，将 source 响应点投影到 target 响应图附近。
cv::Mat renderStructureMatches(const RegistrationContext& ctx, int maxMatches);

} // namespace ir

