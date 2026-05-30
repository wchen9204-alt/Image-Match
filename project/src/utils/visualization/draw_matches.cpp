#include "utils/visualization/draw_matches.h"

#include <algorithm>
#include <opencv2/features2d.hpp>

#include "utils/logger.h"

namespace ir {

cv::Mat DrawMatches::render(const RegistrationContext& ctx, const Options& opt) {
    const auto& fd = ctx.feature_data;
    const auto& md = ctx.match_data;

    if (fd.first.image.empty() || fd.second.image.empty()) {
        IR_LOG_WARN("DrawMatches::render - source images empty, returning empty Mat.");
        return {};
    }

    const std::vector<cv::DMatch>& source =
        (opt.draw_inliers_only && !md.inliers.empty()) ? md.inliers : md.filtered;

    // 匹配过多时按距离择优抽样，兼顾可读性与代表性。
    std::vector<cv::DMatch> draw = source;
    if (opt.max_matches > 0 && static_cast<int>(draw.size()) > opt.max_matches) {
        std::partial_sort(
            draw.begin(),
            draw.begin() + opt.max_matches,
            draw.end(),
            [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });
        draw.resize(opt.max_matches);
    }

    // 绘制阶段保持纯渲染职责，不在这里改写上下文中的任何匹配结果。
    cv::Mat canvas;
    cv::drawMatches(fd.first.image,
                    fd.first.keypoints,
                    fd.second.image,
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
