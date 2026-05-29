#pragma once

#include <filesystem>
#include <string>

#include "core/context.h"

namespace ir {

// ---------------------------------------------------------------------------
// VisualizationManager：统一保存配准后的匹配、内点、叠加和差异图。
//
// 主要供 Benchmark / batch_eval 使用。
// ---------------------------------------------------------------------------
class VisualizationManager {
public:
    struct Options {
        bool draw_matches  = true;
        bool draw_inliers  = true;
        bool draw_overlay  = true;
        bool draw_diff     = true;
        bool save_warped   = true;
        int  max_matches   = 100;
        int  max_inliers   = 200;
    };

    // 输出文件目录：
    //   <output_root>/matches/<stem>_matches.png
    //   <output_root>/inliers/<stem>_inliers.png
    //   <output_root>/overlay/<stem>_overlay.png
    //   <output_root>/diff/<stem>_diff.png
    //   <output_root>/warped/<stem>_warped.png
    bool saveAll(const RegistrationContext& ctx,
                 const std::filesystem::path& output_root,
                 const std::string& stem) const;

    bool saveAll(const RegistrationContext& ctx,
                 const std::filesystem::path& output_root,
                 const std::string& stem,
                 const Options& opt) const;
};

} // namespace ir
