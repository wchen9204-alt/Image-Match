#include "utils/visualization/draw_matches.h"

#include <algorithm>
#include <vector>

#include <opencv2/features2d.hpp>

#include "data/correspondence_view.h"
#include "utils/logger.h"

namespace ir {

/// 从共享对应点快照渲染原始或过滤后的匹配图。
cv::Mat DrawMatches::render(const RegistrationContext& ctx, const Options& opt) {
    const auto& images = ctx.images;

    if (images.first.empty() || images.second.empty()) {
        IR_LOG_WARN("DrawMatches::render - source images empty, returning empty Mat.");
        return {};
    }

    // 可视化只读取流水线已建立的快照，避免每张图重新复制对应点容器。
    const CorrespondenceView view = cachedCorrespondenceView(ctx);
    if (view.empty() || !view.first_keypoints_storage || !view.second_keypoints_storage) {
        IR_LOG_WARN("DrawMatches::render - no drawable correspondence view available.");
        return {};
    }

    // DrawMatches 只负责画匹配阶段的数据层级：raw 表示匹配器原始输出，filtered 表示过滤链结果。
    // 几何内点由 DrawInliers 单独渲染，避免匹配图和内点图语义混用。
    const std::span<const cv::DMatch> matches =
        opt.draw_raw_matches ? view.raw : view.filtered;
    if (matches.empty()) {
        IR_LOG_WARN("DrawMatches::render - no matches available for current draw mode.");
        return {};
    }

    // 绘制时可能按距离截断，因此仅为当前 PNG 建立一份局部排序副本。
    std::vector<cv::DMatch> draw(matches.begin(), matches.end());
    if (opt.max_matches > 0 && static_cast<int>(draw.size()) > opt.max_matches) {
        std::partial_sort(draw.begin(),
                          draw.begin() + opt.max_matches,
                          draw.end(),
                          [](const cv::DMatch& a, const cv::DMatch& b) {
                              return a.distance < b.distance;
                          });
        draw.resize(opt.max_matches);
    }

    // OpenCV 接口要求 vector 引用；storage 指针与 view 中的 span 指向同一份数据。
    cv::Mat canvas;
    cv::drawMatches(images.first,
                    *view.first_keypoints_storage,
                    images.second,
                    *view.second_keypoints_storage,
                    draw,
                    canvas,
                    opt.match_color,
                    opt.single_point,
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    return canvas;
}

} // namespace ir