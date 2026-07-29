#include "data/correspondence_view.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "utils/string_utils.h"

namespace ir {

namespace {

template <typename T>
std::span<const T> asReadOnlySpan(const std::vector<T>& values) {
    return {values.data(), values.size()};
}

/// 创建不携带任何对应点的空快照，供各来源缺少有效数据时统一返回。
CorrespondenceSnapshot emptySnapshot() {
    return {};
}

/// 根据掩码从 DIRECT 的 filtered 中重建内点列表。
/// 直接法仍只写入 mask，因此必须在拥有型快照中保存一份可供评测/可视化读取的内点列表。
void fillInliersFromMask(CorrespondenceSnapshot& snapshot) {
    auto& inliers = snapshot.owned_inliers;
    const auto& filtered = snapshot.owned_filtered;
    const auto& mask = snapshot.owned_inlier_mask;

    inliers.clear();
    if (filtered.empty()) {
        return;
    }

    inliers.reserve(filtered.size());
    for (size_t i = 0; i < filtered.size(); ++i) {
        const bool isInlier = mask.empty() || (i < mask.size() && mask[i] != 0);
        if (isInlier) {
            inliers.push_back(filtered[i]);
        }
    }
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

/// 判断两个结构匹配是否为同一个线段匹配。
bool sameStructureMatch(const cv::DMatch& a, const cv::DMatch& b) {
    return a.queryIdx == b.queryIdx && a.trainIdx == b.trainIdx;
}

/// 判断一个线段匹配是否是内点。
bool isStructureInlierMatch(const cv::DMatch& match, const std::vector<cv::DMatch>& inliers) {
    if (inliers.empty()) {
        return true;
    }
    return std::any_of(inliers.begin(), inliers.end(), [&](const cv::DMatch& inlier) {
        return sameStructureMatch(match, inlier);
    });
}

/// 往结构法拥有型快照添加一对可投影的端点/质心匹配。
void appendLineEndpointPair(CorrespondenceSnapshot& snapshot,
                            std::vector<cv::DMatch>& matches,
                            const cv::Point2f& srcPt,
                            const cv::Point2f& dstPt,
                            float distance,
                            int structureMatchIndex,
                            int inlierFlag) {
    const int idx = static_cast<int>(snapshot.owned_first_keypoints.size());
    snapshot.owned_first_keypoints.emplace_back(srcPt, 1.0f);
    snapshot.owned_second_keypoints.emplace_back(dstPt, 1.0f);
    matches.emplace_back(idx, idx, distance);
    matches.back().imgIdx = structureMatchIndex;
    if (inlierFlag >= 0) {
        snapshot.owned_inlier_mask.push_back(inlierFlag ? 1 : 0);
        if (inlierFlag) {
            snapshot.owned_inliers.push_back(matches.back());
        }
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

/// 将快照中的借用数据或拥有数据统一暴露为轻量 span 视图。
CorrespondenceView CorrespondenceSnapshot::view() const {
    CorrespondenceView result;
    result.source = source;
    result.source_name = source_name;

    if (owns_storage) {
        result.first_keypoints = asReadOnlySpan(owned_first_keypoints);
        result.second_keypoints = asReadOnlySpan(owned_second_keypoints);
        result.first_keypoints_storage = &owned_first_keypoints;
        result.second_keypoints_storage = &owned_second_keypoints;
        result.raw = raw_falls_back_to_filtered ? asReadOnlySpan(owned_filtered)
                                                : asReadOnlySpan(owned_raw);
        result.filtered = asReadOnlySpan(owned_filtered);
        result.inlier_mask = asReadOnlySpan(owned_inlier_mask);
        result.inliers = asReadOnlySpan(owned_inliers);
    } else {
        result.first_keypoints = borrowed_first_keypoints;
        result.second_keypoints = borrowed_second_keypoints;
        result.first_keypoints_storage = borrowed_first_keypoints_storage;
        result.second_keypoints_storage = borrowed_second_keypoints_storage;
        result.raw = raw_falls_back_to_filtered ? borrowed_filtered : borrowed_raw;
        result.filtered = borrowed_filtered;
        result.inlier_mask = borrowed_inlier_mask;
        result.inliers = borrowed_inliers;
    }
    return result;
}

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

/// 从字符串解析对应关系来源枚举；缺省或无法识别时返回 NONE。
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

/// 构建点特征法对应点快照；所有字段直接借用上下文，避免深拷贝。
CorrespondenceSnapshot buildKeypointCorrespondenceSnapshot(const RegistrationContext& ctx) {
    const auto& kd = ctx.keypoint_data;
    const auto& md = ctx.keypoint_match_data;
    if ((md.raw_matches.empty() && md.filtered_matches.empty()) ||
        kd.first.keypoints.empty() ||
        kd.second.keypoints.empty()) {
        return emptySnapshot();
    }

    CorrespondenceSnapshot snapshot;
    snapshot.source = CorrespondenceSource::KEYPOINT;
    snapshot.source_name = toString(snapshot.source);
    snapshot.borrowed_first_keypoints = asReadOnlySpan(kd.first.keypoints);
    snapshot.borrowed_second_keypoints = asReadOnlySpan(kd.second.keypoints);
    snapshot.borrowed_first_keypoints_storage = &kd.first.keypoints;
    snapshot.borrowed_second_keypoints_storage = &kd.second.keypoints;
    snapshot.borrowed_raw = asReadOnlySpan(md.raw_matches);
    snapshot.borrowed_filtered = asReadOnlySpan(md.filtered_matches);
    snapshot.borrowed_inlier_mask = asReadOnlySpan(md.inlier_mask);
    snapshot.borrowed_inliers = asReadOnlySpan(md.inlier_matches);
    return snapshot;
}

/// 构建深度学习对应点快照；学习匹配器已把稀疏点对写入统一的 keypoint 容器。
CorrespondenceSnapshot buildLearningCorrespondenceSnapshot(const RegistrationContext& ctx) {
    CorrespondenceSnapshot snapshot = buildKeypointCorrespondenceSnapshot(ctx);
    if (snapshot.empty()) {
        return snapshot;
    }
    snapshot.source = CorrespondenceSource::LEARNING;
    snapshot.source_name = toString(snapshot.source);
    return snapshot;
}

/// 构建结构法拥有型快照，将线段端点或轮廓质心显式转换为可投影点对。
CorrespondenceSnapshot buildStructureCorrespondenceSnapshot(const RegistrationContext& ctx) {
    const auto& sd = ctx.structure_data;
    const auto& md = ctx.structure_match_data;
    if (md.raw_matches_knn.empty() && md.line_matches.empty()) {
        return emptySnapshot();
    }

    CorrespondenceSnapshot snapshot;
    snapshot.source = CorrespondenceSource::STRUCTURE;
    snapshot.source_name = md.method.empty() ? toString(snapshot.source) : md.method;
    snapshot.owns_storage = true;

    if (sd.type == StructureType::LINE) {
        const auto& srcLines = sd.first.lines;
        const auto& dstLines = sd.second.lines;
        const auto appendLineMatch = [&](const cv::DMatch& lm,
                                         int structureMatchIndex,
                                         std::vector<cv::DMatch>& matches,
                                         int inlierFlag) {
            if (lm.queryIdx < 0 || lm.trainIdx < 0 ||
                lm.queryIdx >= static_cast<int>(srcLines.size()) ||
                lm.trainIdx >= static_cast<int>(dstLines.size())) {
                return;
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

            appendLineEndpointPair(snapshot, matches, sp1, dp1, lm.distance, structureMatchIndex, inlierFlag);
            appendLineEndpointPair(snapshot, matches, sp2, dp2, lm.distance, structureMatchIndex, inlierFlag);
        };

        int rawIndex = 0;
        for (const auto& neighbours : md.raw_matches_knn) {
            for (const cv::DMatch& lm : neighbours) {
                appendLineMatch(lm, rawIndex++, snapshot.owned_raw, -1);
            }
        }

        for (size_t i = 0; i < md.line_matches.size(); ++i) {
            const cv::DMatch& lm = md.line_matches[i];
            const bool inlier = isStructureInlierMatch(lm, md.inlier_line_matches);
            appendLineMatch(lm, static_cast<int>(i), snapshot.owned_filtered, inlier ? 1 : 0);
        }
    } else if (sd.type == StructureType::CONTOUR) {
        const auto& srcContours = sd.first.contours;
        const auto& dstContours = sd.second.contours;
        const auto appendContourMatch = [&](const cv::DMatch& lm,
                                            int structureMatchIndex,
                                            std::vector<cv::DMatch>& matches,
                                            int inlierFlag) {
            if (lm.queryIdx < 0 || lm.trainIdx < 0 ||
                lm.queryIdx >= static_cast<int>(srcContours.size()) ||
                lm.trainIdx >= static_cast<int>(dstContours.size())) {
                return;
            }

            const cv::Point2f srcCentroid = contourCentroid(srcContours[static_cast<size_t>(lm.queryIdx)]);
            const cv::Point2f dstCentroid = contourCentroid(dstContours[static_cast<size_t>(lm.trainIdx)]);
            if (srcCentroid.x < 0.0f || srcCentroid.y < 0.0f ||
                dstCentroid.x < 0.0f || dstCentroid.y < 0.0f) {
                return;
            }

            appendLineEndpointPair(
                snapshot, matches, srcCentroid, dstCentroid, lm.distance, structureMatchIndex, inlierFlag);
        };

        int rawIndex = 0;
        for (const auto& neighbours : md.raw_matches_knn) {
            for (const cv::DMatch& lm : neighbours) {
                appendContourMatch(lm, rawIndex++, snapshot.owned_raw, -1);
            }
        }

        for (size_t i = 0; i < md.line_matches.size(); ++i) {
            const cv::DMatch& lm = md.line_matches[i];
            const bool inlier = isStructureInlierMatch(lm, md.inlier_line_matches);
            appendContourMatch(lm, static_cast<int>(i), snapshot.owned_filtered, inlier ? 1 : 0);
        }
    }

    snapshot.raw_falls_back_to_filtered = snapshot.owned_raw.empty();
    return snapshot.empty() ? emptySnapshot() : snapshot;
}

/// 构建直接法拥有型快照；直接法点对必须转换为 keypoint 以复用通用绘制和评测逻辑。
CorrespondenceSnapshot buildDirectCorrespondenceSnapshot(const RegistrationContext& ctx) {
    const auto& dd = ctx.direct_data;
    const size_t n = std::min(dd.points1.size(), dd.points2.size());
    if (n == 0) {
        return emptySnapshot();
    }

    CorrespondenceSnapshot snapshot;
    snapshot.source = CorrespondenceSource::DIRECT;
    snapshot.source_name = dd.method.empty() ? toString(snapshot.source) : dd.method;
    snapshot.owns_storage = true;
    snapshot.raw_falls_back_to_filtered = true;
    snapshot.owned_first_keypoints.reserve(n);
    snapshot.owned_second_keypoints.reserve(n);
    snapshot.owned_filtered.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        snapshot.owned_first_keypoints.emplace_back(dd.points1[i], 1.0f);
        snapshot.owned_second_keypoints.emplace_back(dd.points2[i], 1.0f);

        // 直接法点对通常没有描述子距离；若方法没有提供距离，则以位移长度作为排序参考。
        const float distance =
            i < dd.matches.size() ? dd.matches[i].distance : static_cast<float>(cv::norm(dd.points2[i] - dd.points1[i]));
        snapshot.owned_filtered.emplace_back(static_cast<int>(i), static_cast<int>(i), distance);
    }

    snapshot.owned_inlier_mask = dd.inlier_mask;
    fillInliersFromMask(snapshot);
    return snapshot;
}

/// 按显式来源构造对应点快照。
CorrespondenceSnapshot buildCorrespondenceSnapshot(const RegistrationContext& ctx,
                                                    CorrespondenceSource source) {
    switch (source) {
    case CorrespondenceSource::KEYPOINT:
        return buildKeypointCorrespondenceSnapshot(ctx);
    case CorrespondenceSource::STRUCTURE:
        return buildStructureCorrespondenceSnapshot(ctx);
    case CorrespondenceSource::DIRECT:
        return buildDirectCorrespondenceSnapshot(ctx);
    case CorrespondenceSource::LEARNING:
        return buildLearningCorrespondenceSnapshot(ctx);
    case CorrespondenceSource::NONE:
    default:
        return emptySnapshot();
    }
}

/// 在来源未知时，优先直接法，其次结构法，最后点特征法。
CorrespondenceSnapshot buildBestCorrespondenceSnapshot(const RegistrationContext& ctx) {
    const CorrespondenceSource source = correspondenceSourceFromContext(ctx);
    if (source != CorrespondenceSource::NONE) {
        return buildCorrespondenceSnapshot(ctx, source);
    }

    CorrespondenceSnapshot direct = buildDirectCorrespondenceSnapshot(ctx);
    if (!direct.empty()) {
        return direct;
    }

    CorrespondenceSnapshot structure = buildStructureCorrespondenceSnapshot(ctx);
    if (!structure.empty()) {
        return structure;
    }

    CorrespondenceSnapshot keypoint = buildKeypointCorrespondenceSnapshot(ctx);
    if (!keypoint.empty()) {
        return keypoint;
    }

    return emptySnapshot();
}

/// 刷新当前运行的共享快照；几何估计前后调用可分别覆盖输入匹配和最新内点。
void refreshCorrespondenceSnapshot(RegistrationContext& ctx) {
    auto snapshot = std::make_shared<CorrespondenceSnapshot>(buildBestCorrespondenceSnapshot(ctx));
    if (snapshot->empty()) {
        ctx.correspondence_snapshot.reset();
        return;
    }
    ctx.correspondence_snapshot = std::move(snapshot);
}

/// 为几何估计取得缓存视图；初始化器等非 BasePipeline 调用点可在此按需建立缓存。
CorrespondenceView ensureCorrespondenceView(RegistrationContext& ctx) {
    const CorrespondenceSource source = correspondenceSourceFromContext(ctx);
    if (!ctx.correspondence_snapshot ||
        (source != CorrespondenceSource::NONE && ctx.correspondence_snapshot->source != source)) {
        refreshCorrespondenceSnapshot(ctx);
    }
    return cachedCorrespondenceView(ctx);
}

/// 评测和可视化只读取共享快照，避免每个消费者重新构建对应点容器。
CorrespondenceView cachedCorrespondenceView(const RegistrationContext& ctx) {
    return ctx.correspondence_snapshot ? ctx.correspondence_snapshot->view() : CorrespondenceView{};
}

} // namespace ir