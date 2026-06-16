#include "utils/visualization/draw_matches.h"

#include <algorithm>
#include <vector>

#include <opencv2/features2d.hpp>

#include "data/correspondence_view.h"
#include "utils/logger.h"

namespace ir {

cv::Mat DrawMatches::render(const RegistrationContext& ctx, const Options& opt) {
    const auto& images = ctx.images;

    if (images.first.empty() || images.second.empty()) {
        IR_LOG_WARN("DrawMatches::render - source images empty, returning empty Mat.");
        return {};
    }

    const CorrespondenceSource correspondenceSource = correspondenceSourceFromContext(ctx);
    const CorrespondenceView view =
        correspondenceSource == CorrespondenceSource::NONE ? buildBestCorrespondenceView(ctx)
                                                           : buildCorrespondenceView(ctx, correspondenceSource);
    if (view.empty()) {
        IR_LOG_WARN("DrawMatches::render - no correspondence view available.");
        return {};
    }

    // DrawMatches 只负责画匹配阶段的数据层级：raw 表示匹配器原始输出，filtered 表示过滤链结果。
    // 几何内点由 DrawInliers 单独渲染，避免匹配图和内点图语义混用。
    const std::vector<cv::DMatch>& matches =
        opt.draw_raw_matches ? view.raw : view.filtered;
    if (matches.empty()) {
        IR_LOG_WARN("DrawMatches::render - no matches available for current draw mode.");
        return {};
    }

    // 匹配过多时按距离择优抽样，兼顾可读性与代表性。
    std::vector<cv::DMatch> draw = matches;
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
                    view.first_keypoints,
                    images.second,
                    view.second_keypoints,
                    draw,
                    canvas,
                    opt.match_color,
                    opt.single_point,
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    return canvas;
}

} // namespace ir

