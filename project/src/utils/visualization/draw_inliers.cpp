#include "utils/visualization/draw_inliers.h"

#include <algorithm>
#include <vector>

#include <opencv2/features2d.hpp>

#include "data/correspondence_view.h"
#include "utils/logger.h"

namespace ir {

cv::Mat DrawInliers::render(const RegistrationContext& ctx, const Options& opt) {
    const auto& images = ctx.images;

    if (images.first.empty() || images.second.empty()) {
        IR_LOG_WARN("DrawInliers: empty source images.");
        return {};
    }
    const CorrespondenceView view = cachedCorrespondenceView(ctx);
    // draw_inliers 直接使用几何阶段已经确认并写回的内点，不在可视化阶段重新推导。
    if (!view.first_keypoints_storage || !view.second_keypoints_storage || view.inliers.empty()) {
        IR_LOG_DEBUG("DrawInliers: no inlier matches available.");
        return {};
    }

    std::vector<cv::DMatch> inliers(view.inliers.begin(), view.inliers.end());
    if (opt.max_inliers > 0 && static_cast<int>(inliers.size()) > opt.max_inliers) {
        // 内点数量过多时优先保留距离更小的匹配，避免可视化过于拥挤。
        std::partial_sort(inliers.begin(),
                          inliers.begin() + opt.max_inliers,
                          inliers.end(),
                          [](const cv::DMatch& a, const cv::DMatch& b) {
                              return a.distance < b.distance;
                          });
        inliers.resize(opt.max_inliers);
    }

    cv::Mat overlay;
    cv::drawMatches(images.first,
                    *view.first_keypoints_storage,
                    images.second,
                    *view.second_keypoints_storage,
                    inliers,
                    overlay,
                    opt.inlier_color,
                    opt.inlier_color,
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return overlay;
}

} // namespace ir

