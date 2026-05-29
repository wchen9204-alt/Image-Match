#pragma once

#include <chrono>
#include <string>

namespace ir {

// ---------------------------------------------------------------------------
// RegistrationResult：一次配准运行的轻量摘要结果。
// ---------------------------------------------------------------------------
struct RegistrationResult {
    bool        success         = false;
    std::string message;

    // 数量统计。
    int num_keypoints_first   = 0;
    int num_keypoints_second  = 0;
    int num_raw_matches       = 0;
    int num_filtered_matches  = 0;
    int num_inliers           = 0;

    double inlier_ratio       = 0.0;
    double mean_reproj_error  = 0.0;

    // 阶段耗时，单位毫秒。
    double t_load_ms          = 0.0;
    double t_extract_ms       = 0.0;
    double t_match_ms         = 0.0;
    double t_filter_ms        = 0.0;
    double t_geometry_ms      = 0.0;
    double t_warp_ms          = 0.0;
    double t_total_ms         = 0.0;
};

} // namespace ir
