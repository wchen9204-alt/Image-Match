#include "utils/visualization/draw_inliers.h"

#include <algorithm>
#include <opencv2/features2d.hpp>

#include "utils/logger.h"

namespace ir {

cv::Mat DrawInliers::render(const RegistrationContext& ctx, const Options& opt) {
    const auto& fd = ctx.feature_data;
    const auto& md = ctx.match_data;

    if (fd.first.image.empty() || fd.second.image.empty()) {
        IR_LOG_WARN("DrawInliers: empty source images.");
        return {};
    }
    if (md.inliers.empty() && md.filtered.empty()) {
        IR_LOG_WARN("DrawInliers: no matches available.");
        return {};
    }

    cv::Mat canvas;

    if (opt.draw_outliers && !md.filtered.empty() && !md.inlier_mask.empty()) {
        // 先画外点，再把内点覆盖到上层。
        std::vector<cv::DMatch> outliers;
        outliers.reserve(md.filtered.size());
        for (size_t i = 0; i < md.filtered.size() && i < md.inlier_mask.size(); ++i) {
            if (!md.inlier_mask[i]) outliers.push_back(md.filtered[i]);
        }
        cv::drawMatches(
            fd.first.image,  fd.first.keypoints,
            fd.second.image, fd.second.keypoints,
            outliers,
            canvas,
            opt.non_inlier_color,
            opt.non_inlier_color,
            std::vector<char>(),
            cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    }

    std::vector<cv::DMatch> inliers = md.inliers;
    if (opt.max_inliers > 0 && static_cast<int>(inliers.size()) > opt.max_inliers) {
        std::partial_sort(
            inliers.begin(),
            inliers.begin() + opt.max_inliers,
            inliers.end(),
            [](const cv::DMatch& a, const cv::DMatch& b) {
                return a.distance < b.distance;
            });
        inliers.resize(opt.max_inliers);
    }

    cv::Mat overlay;
    cv::drawMatches(
        fd.first.image,  fd.first.keypoints,
        fd.second.image, fd.second.keypoints,
        inliers,
        overlay,
        opt.inlier_color,
        opt.inlier_color,
        std::vector<char>(),
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    if (canvas.empty()) {
        canvas = overlay;
    } else {
        // 将外点层和内点层合成到同一画布。
        cv::addWeighted(canvas, 0.5, overlay, 0.5, 0.0, canvas);
    }
    return canvas;
}

} // namespace ir
