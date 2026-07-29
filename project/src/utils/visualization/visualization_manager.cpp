#include "utils/visualization/visualization_manager.h"

#include <filesystem>

#include <opencv2/imgcodecs.hpp>

#include "data/correspondence_view.h"
#include "utils/file_utils.h"
#include "utils/logger.h"
#include "utils/visualization/draw_diff.h"
#include "utils/visualization/draw_inliers.h"
#include "utils/visualization/draw_matches.h"
#include "utils/visualization/draw_overlay.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

// 统一封装落盘动作，避免各输出分支重复处理目录创建与空图保护。
bool writeImage(const fs::path& path, const cv::Mat& img) {
    if (img.empty()) {
        return false;
    }
    file_utils::ensureDirectory(path.parent_path());
    return cv::imwrite(path.string(), img);
}

} // namespace

bool VisualizationManager::saveAll(const RegistrationContext& ctx,
                                   const fs::path& output_root,
                                   const std::string& stem) const {
    return saveAll(ctx, output_root, stem, Options{});
}

bool VisualizationManager::saveAll(const RegistrationContext& ctx,
                                   const fs::path& output_root,
                                   const std::string& stem,
                                   const Options& opt) const {
    // 输出根目录和文件名前缀缺一不可，否则无法维持批处理结果的稳定组织结构。
    if (output_root.empty() || stem.empty()) {
        return false;
    }

    if (opt.draw_matches) {
        // 匹配图保留全局对应关系，适合作为第一层人工诊断材料。
        DrawMatches::Options match_opt;
        match_opt.max_matches = opt.max_matches;
        const cv::Mat img = DrawMatches::render(ctx, match_opt);
        const fs::path path = output_root / "all_match" / (stem + "_all_match.png");
        if (writeImage(path, img)) {
            IR_LOG_DEBUG("Saved ", path.string());
        }
    }

    const CorrespondenceView view = cachedCorrespondenceView(ctx);

    if (opt.draw_inliers && !view.inliers.empty()) {
        // 内点图只展示几何模型接受的匹配，用于检查鲁棒估计质量。
        DrawInliers::Options inlier_opt;
        inlier_opt.max_inliers = opt.max_inliers;
        const cv::Mat img = DrawInliers::render(ctx, inlier_opt);
        const fs::path path = output_root / "inlier_match" / (stem + "_inlier_match.png");
        if (writeImage(path, img)) {
            IR_LOG_DEBUG("Saved ", path.string());
        }
    }

    if (opt.save_warped && !ctx.warped_image.empty()) {
        // 变换后图像是后续叠加图、差异图和图像指标计算的直接输入。
        const fs::path path = output_root / "warped" / (stem + "_warped.png");
        if (writeImage(path, ctx.warped_image)) {
            IR_LOG_DEBUG("Saved ", path.string());
        }
    }

    if (opt.draw_overlay && !ctx.warped_image.empty()) {
        // 叠加图适合快速观察结构是否对齐，但不强调误差大小。
        DrawOverlay::Options overlay_opt;
        overlay_opt.alpha = 0.5;
        const cv::Mat img = DrawOverlay::render(ctx, overlay_opt);
        const fs::path path = output_root / "overlay" / (stem + "_overlay.png");
        if (writeImage(path, img)) {
            IR_LOG_DEBUG("Saved ", path.string());
        }
    }

    if (opt.draw_diff && !ctx.warped_image.empty()) {
        // 差异图更强调局部残差分布，适合作为配准误差的直觉补充。
        DrawDiff::Options diff_opt;
        diff_opt.heatmap = true;
        const cv::Mat img = DrawDiff::render(ctx, diff_opt);
        const fs::path path = output_root / "diff" / (stem + "_diff.png");
        if (writeImage(path, img)) {
            IR_LOG_DEBUG("Saved ", path.string());
        }
    }

    return true;
}

} // namespace ir


