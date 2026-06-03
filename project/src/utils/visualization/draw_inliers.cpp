#include "utils/visualization/draw_inliers.h"

#include <algorithm>
#include <vector>

#include <opencv2/features2d.hpp>

#include "utils/logger.h"

namespace ir {

cv::Mat DrawInliers::render(const RegistrationContext& ctx, const Options& opt) {
    const auto& fd = ctx.keypoint_data;
    const auto& images = ctx.images;
    const auto& md = ctx.keypoint_match_data;

    if (images.first.empty() || images.second.empty()) {
        IR_LOG_WARN("DrawInliers: empty source images.");
        return {};
    }
    if (md.inliers.empty()) {
        IR_LOG_WARN("DrawInliers: no inlier matches available.");
        return {};
    }

    cv::Mat canvas;

    if (opt.draw_outliers && !md.filtered.empty() && !md.inlier_mask.empty()) {
        // 先绘制外点层，再把内点覆盖到上层，突出鲁棒估计真正接受的对应关系。
        std::vector<cv::DMatch> outliers;
        outliers.reserve(md.filtered.size());
        for (size_t i = 0; i < md.filtered.size() && i < md.inlier_mask.size(); ++i) {
            if (!md.inlier_mask[i]) {
                outliers.push_back(md.filtered[i]);
            }
        }

        cv::drawMatches(images.first,
                        fd.first.keypoints,
                        images.second,
                        fd.second.keypoints,
                        outliers,
                        canvas,
                        opt.non_inlier_color,
                        opt.non_inlier_color,
                        std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    }

    std::vector<cv::DMatch> inliers = md.inliers;
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
                    fd.first.keypoints,
                    images.second,
                    fd.second.keypoints,
                    inliers,
                    overlay,
                    opt.inlier_color,
                    opt.inlier_color,
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    if (canvas.empty()) {
        canvas = overlay;
    } else {
        // 外点层与内点层做半透明合成，既保留整体分布，也不掩盖最终有效匹配。
        cv::addWeighted(canvas, 0.5, overlay, 0.5, 0.0, canvas);
    }
    return canvas;
}

} // namespace ir
