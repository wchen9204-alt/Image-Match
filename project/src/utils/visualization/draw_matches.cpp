#include "utils/visualization/draw_matches.h"

#include <algorithm>
#include <vector>

#include <opencv2/features2d.hpp>

#include "utils/logger.h"

namespace ir {

cv::Mat DrawMatches::render(const RegistrationContext& ctx, const Options& opt) {
    const auto& fd = ctx.keypoint_data;
    const auto& images = ctx.images;
    const auto& md = ctx.keypoint_match_data;

    if (images.first.empty() || images.second.empty()) {
        IR_LOG_WARN("DrawMatches::render - source images empty, returning empty Mat.");
        return {};
    }

    // `draw_inliers_only` 为 true 时严格只绘制真实内点，不再回退到 filtered。
    const std::vector<cv::DMatch>& source = opt.draw_inliers_only ? md.inliers : md.filtered;
    if (source.empty()) {
        IR_LOG_WARN("DrawMatches::render - no matches available for current draw mode.");
        return {};
    }

    // 匹配过多时按距离择优抽样，兼顾可读性与代表性。
    std::vector<cv::DMatch> draw = source;
    if (opt.max_matches > 0 && static_cast<int>(draw.size()) > opt.max_matches) {
        std::partial_sort(draw.begin(),
                          draw.begin() + opt.max_matches,
                          draw.end(),
                          [](const cv::DMatch& a, const cv::DMatch& b) {
                              return a.distance < b.distance;
                          });
        draw.resize(opt.max_matches);
    }

    // 绘制阶段保持纯渲染职责，不在这里改写上下文中的任何匹配结果。
    cv::Mat canvas;
    cv::drawMatches(images.first,
                    fd.first.keypoints,
                    images.second,
                    fd.second.keypoints,
                    draw,
                    canvas,
                    opt.match_color,
                    opt.single_point,
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    return canvas;
}

} // namespace ir
