#include "data/correspondence_view.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "utils/string_utils.h"

namespace ir {

namespace {

/// 根据掩码（inlier_mask），从所有匹配对（filtered）里筛选出内点，存入 inliers 列表。
void fillInliersFromMask(CorrespondenceView& view) {
    view.inliers.clear();
    if (view.filtered.empty()) {
        return;
    }

    view.inliers.reserve(view.filtered.size());
    for (size_t i = 0; i < view.filtered.size(); ++i) {
        const bool isInlier = view.inlier_mask.empty() ||
                              (i < view.inlier_mask.size() && view.inlier_mask[i] != 0);
        if (isInlier) {
            view.inliers.push_back(view.filtered[i]);
        }
    }
}


CorrespondenceView emptyView() {
    CorrespondenceView view;
    view.source_name = "NONE";
    return view;
}

/// 判断两条线段的端点配对是否更像同一条线段，辅助结构法线段匹配的内点验证。
bool lineDirectionConsistent(const cv::Point2f& sp1,
                             const cv::Point2f& sp2,
                             const cv::Point2f& dp1,
                             const cv::Point2f& dp2) {
    const cv::Point2f sv = sp2 - sp1;
    const cv::Point2f dv = dp2 - dp1;
    if (cv::norm(sv) < 1e-6f || cv::norm(dv) < 1e-6f) {
        return true;
    }

    const float keep = static_cast<float>(cv::norm(sp1 - dp1) + cv::norm(sp2 - dp2));
    const float swap = static_cast<float>(cv::norm(sp1 - dp2) + cv::norm(sp2 - dp1));
    return keep <= swap;
}

/// 判断两个结构匹配是否为同一个线段匹配
bool sameStructureMatch(const cv::DMatch& a, const cv::DMatch& b) {
    return a.queryIdx == b.queryIdx && a.trainIdx == b.trainIdx;
}
/// 判断一个线段匹配是否是内点
bool isStructureInlierMatch(const cv::DMatch& match, const std::vector<cv::DMatch>& inliers) {
    if (inliers.empty()) {
        return true;
    }
    /// 检查当前这个 match，是否存在于内点列表 inliers 中
    return std::any_of(inliers.begin(), inliers.end(), [&](const cv::DMatch& inlier) {
        return sameStructureMatch(match, inlier);
    });
}

/// 往匹配数据里添加一对线段端点匹配
void appendLineEndpointPair(CorrespondenceView& view,
                            const cv::Point2f& srcPt,
                            const cv::Point2f& dstPt,
                            float distance,
                            int structureMatchIndex,
                            bool inlier) {
    const int idx = static_cast<int>(view.first_keypoints.size());
    view.first_keypoints.emplace_back(srcPt, 1.0f);
    view.second_keypoints.emplace_back(dstPt, 1.0f);
    view.filtered.emplace_back(idx, idx, distance);
    view.filtered.back().imgIdx = structureMatchIndex;
    view.inlier_mask.push_back(inlier ? 1 : 0);
    if (inlier) {
        view.inliers.push_back(view.filtered.back());
    }
}
/// 计算轮廓的质心，辅助结构法轮廓匹配的内点验证。
cv::Point2f contourCentroid(const std::vector<cv::Point>& contour) {
    const cv::Moments m = cv::moments(contour);
    if (std::abs(m.m00) < 1e-9) {
        return {-1.0f, -1.0f};
    }
    return {static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00)};
}

} // namespace

const char* toString(CorrespondenceSource source) {
    switch (source) {
    case CorrespondenceSource::KEYPOINT: return "KEYPOINT";
    case CorrespondenceSource::STRUCTURE:return "STRUCTURE";
    case CorrespondenceSource::DIRECT:   return "DIRECT";
    case CorrespondenceSource::LEARNING: return "LEARNING";
    case CorrespondenceSource::NONE:
    default:                             return "NONE";
    }
}

// 从字符串解析对应关系来源枚举；缺省或无法识别时返回 NONE。
CorrespondenceSource correspondenceSourceFromString(const std::string& source) {
    const std::string upper = string_utils::toUpperAscii(source);
    if (upper == "KEYPOINT") {
        return CorrespondenceSource::KEYPOINT;
    }
    if (upper == "STRUCTURE") {
        return CorrespondenceSource::STRUCTURE;
    }
    if (upper == "DIRECT") {
        return CorrespondenceSource::DIRECT;
    }
    if (upper == "LEARNING" || upper == "DEEP") {
        return CorrespondenceSource::LEARNING;
    }
    return CorrespondenceSource::NONE;
}

CorrespondenceSource correspondenceSourceFromContext(const RegistrationContext& ctx) {
    const CorrespondenceSource fromRuntime = correspondenceSourceFromString(ctx.correspondence_source);
    if (fromRuntime != CorrespondenceSource::NONE) {
        return fromRuntime;
    }
    return correspondenceSourceFromString(ctx.geometry_data.correspondence_source);
}

/// 构建点特征法对应关系视图
CorrespondenceView buildKeypointCorrespondenceView(const RegistrationContext& ctx) {
    const auto& kd = ctx.keypoint_data;
    const auto& md = ctx.keypoint_match_data;
    if (md.filtered.empty() || kd.first.keypoints.empty() || kd.second.keypoints.empty()) {
        return emptyView();
    }

    CorrespondenceView view;
    view.source = CorrespondenceSource::KEYPOINT;
    view.source_name = toString(view.source);
    view.first_keypoints = kd.first.keypoints;
    view.second_keypoints = kd.second.keypoints;
    view.filtered = md.filtered;
    view.inlier_mask = md.inlier_mask;
    view.inliers = md.inliers;
    if (view.inliers.empty()) {
        fillInliersFromMask(view);
    }
    return view;
}

/// 构建深度学习对应关系视图。
CorrespondenceView buildLearningCorrespondenceView(const RegistrationContext& ctx) {
    // 第一版学习方法输出的是稀疏匹配点对；这里显式转为统一视图，来源仍标记为 LEARNING。
    CorrespondenceView view = buildKeypointCorrespondenceView(ctx);
    if (view.empty()) {
        return view;
    }
    view.source = CorrespondenceSource::LEARNING;
    view.source_name = toString(view.source);
    return view;
}

/// 构建结构法对应关系视图
CorrespondenceView buildStructureCorrespondenceView(const RegistrationContext& ctx) {
    const auto& sd = ctx.structure_data;
    const auto& md = ctx.structure_match_data;
    if (md.line_matches.empty()) {
        return emptyView();
    }

    CorrespondenceView view;
    view.source = CorrespondenceSource::STRUCTURE;
    view.source_name = md.method.empty() ? toString(view.source) : md.method;

    if (sd.type == StructureType::LINE) {
        const auto& srcLines = sd.first.lines;
        const auto& dstLines = sd.second.lines;
        for (size_t i = 0; i < md.line_matches.size(); ++i) {
            const cv::DMatch& lm = md.line_matches[i];
            if (lm.queryIdx < 0 || lm.trainIdx < 0 ||
                lm.queryIdx >= static_cast<int>(srcLines.size()) ||
                lm.trainIdx >= static_cast<int>(dstLines.size())) {
                continue;
            }

            const cv::Vec4i& srcLine = srcLines[static_cast<size_t>(lm.queryIdx)];
            const cv::Vec4i& dstLine = dstLines[static_cast<size_t>(lm.trainIdx)];
            const cv::Point2f sp1(static_cast<float>(srcLine[0]), static_cast<float>(srcLine[1]));
            const cv::Point2f sp2(static_cast<float>(srcLine[2]), static_cast<float>(srcLine[3]));
            cv::Point2f dp1(static_cast<float>(dstLine[0]), static_cast<float>(dstLine[1]));
            cv::Point2f dp2(static_cast<float>(dstLine[2]), static_cast<float>(dstLine[3]));
            if (!lineDirectionConsistent(sp1, sp2, dp1, dp2)) {
                std::swap(dp1, dp2);
            }

            // 线段匹配转为两个端点点对，保持与结构几何估计阶段相同的点对语义。
            const bool inlier = isStructureInlierMatch(lm, md.inlier_line_matches);
            appendLineEndpointPair(view, sp1, dp1, lm.distance, static_cast<int>(i), inlier);
            appendLineEndpointPair(view, sp2, dp2, lm.distance, static_cast<int>(i), inlier);
        }
    } else if (sd.type == StructureType::CONTOUR) {
        const auto& srcContours = sd.first.contours;
        const auto& dstContours = sd.second.contours;
        for (size_t i = 0; i < md.line_matches.size(); ++i) {
            const cv::DMatch& lm = md.line_matches[i];
            if (lm.queryIdx < 0 || lm.trainIdx < 0 ||
                lm.queryIdx >= static_cast<int>(srcContours.size()) ||
                lm.trainIdx >= static_cast<int>(dstContours.size())) {
                continue;
            }

            const cv::Point2f srcCentroid = contourCentroid(srcContours[static_cast<size_t>(lm.queryIdx)]);
            const cv::Point2f dstCentroid = contourCentroid(dstContours[static_cast<size_t>(lm.trainIdx)]);
            if (srcCentroid.x < 0.0f || srcCentroid.y < 0.0f ||
                dstCentroid.x < 0.0f || dstCentroid.y < 0.0f) {
                continue;
            }

            // 轮廓匹配转为质心点对，保持与结构几何估计阶段相同的点对语义。
            const bool inlier = isStructureInlierMatch(lm, md.inlier_line_matches);
            appendLineEndpointPair(view, srcCentroid, dstCentroid, lm.distance, static_cast<int>(i), inlier);
        }
    }

    return view.empty() ? emptyView() : view;
}

/// 构建直接法对应关系视图
CorrespondenceView buildDirectCorrespondenceView(const RegistrationContext& ctx) {
    const auto& dd = ctx.direct_data;
    const size_t n = std::min(dd.points1.size(), dd.points2.size());
    if (n == 0) {
        return emptyView();
    }

    CorrespondenceView view;
    view.source = CorrespondenceSource::DIRECT;
    view.source_name = dd.method.empty() ? toString(view.source) : dd.method;
    view.first_keypoints.reserve(n);
    view.second_keypoints.reserve(n);
    view.filtered.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        view.first_keypoints.emplace_back(dd.points1[i], 1.0f);
        view.second_keypoints.emplace_back(dd.points2[i], 1.0f);

        // 直接法点对没有描述子距离；若算法写入了距离则沿用，否则用位移长度作为排序参考。
        const float distance =
            i < dd.matches.size() ? dd.matches[i].distance : static_cast<float>(cv::norm(dd.points2[i] - dd.points1[i]));
        view.filtered.emplace_back(static_cast<int>(i), static_cast<int>(i), distance);
    }

    view.inlier_mask = dd.inlier_mask;
    fillInliersFromMask(view);
    return view;
}

CorrespondenceView buildCorrespondenceView(const RegistrationContext& ctx,
                                           CorrespondenceSource source) {
    switch (source) {
    case CorrespondenceSource::KEYPOINT:
        return buildKeypointCorrespondenceView(ctx);
    case CorrespondenceSource::STRUCTURE:
        return buildStructureCorrespondenceView(ctx);
    case CorrespondenceSource::DIRECT:
        return buildDirectCorrespondenceView(ctx);
    case CorrespondenceSource::LEARNING:
        return buildLearningCorrespondenceView(ctx);
    case CorrespondenceSource::NONE:
    default:
        return emptyView();
    }
}

/// 在来源未知时，优先直接法，其次结构法，最后点特征法。
CorrespondenceView buildBestCorrespondenceView(const RegistrationContext& ctx) {
    const CorrespondenceSource source = correspondenceSourceFromContext(ctx);
    if (source != CorrespondenceSource::NONE) {
        return buildCorrespondenceView(ctx, source);
    }

    // 优先直接法：DirectPipeline 不再把点对伪装成 keypoint，上层读取时直接使用 direct_data。
    CorrespondenceView direct = buildDirectCorrespondenceView(ctx);
    if (!direct.empty()) {
        return direct;
    }

    // 结构法显式从结构匹配构造视图，避免把线段端点/轮廓质心误标成点特征来源。
    CorrespondenceView structure = buildStructureCorrespondenceView(ctx);
    if (!structure.empty()) {
        return structure;
    }

    // 点特征作为最后的通用离散点对来源。
    CorrespondenceView keypoint = buildKeypointCorrespondenceView(ctx);
    if (!keypoint.empty()) {
        return keypoint;
    }

    return emptyView();
}

} // namespace ir
