#pragma once

#include <filesystem>
#include <string>

#include "core/context.h"

namespace ir {

/// 汇总输出可视化结果的管理器。
///
/// 主要供 Benchmark 或批处理脚本使用，将匹配、内点、叠加图、差异图
/// 和变换后图像统一保存到结果目录中。
class VisualizationManager {
public:
    /// 可视化开关与绘制上限。
    struct Options {
        bool draw_matches = true;
        bool draw_inliers = true;
        bool draw_overlay = true;
        bool draw_diff = true;
        bool save_warped = true;
        int max_matches = 100;
        int max_inliers = 200;
    };

    /// 保存默认可视化输出。
    bool saveAll(const RegistrationContext& ctx,
                 const std::filesystem::path& output_root,
                 const std::string& stem) const;

    /// 按指定选项保存所有可视化输出。
    bool saveAll(const RegistrationContext& ctx,
                 const std::filesystem::path& output_root,
                 const std::string& stem,
                 const Options& opt) const;
};

} // namespace ir
