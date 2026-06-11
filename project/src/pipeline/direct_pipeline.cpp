#include "pipeline/direct_pipeline.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/config.h"
#include "core/factory.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/visualization/draw_matches.h"
#include "utils/yaml_utils.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

/// 将 warped 与 target 的绝对差渲染成伪彩色热力图，便于肉眼观察局部配准误差。
cv::Mat renderWarpDifference(const cv::Mat& warped, const cv::Mat& target) {
    if (warped.empty() || target.empty() || warped.size() != target.size()) {
        return {};
    }

    cv::Mat warpedGray;
    cv::Mat targetGray;
    if (warped.channels() == 1) {
        warpedGray = warped;
    } else {
        cv::cvtColor(warped, warpedGray, cv::COLOR_BGR2GRAY);
    }
    if (target.channels() == 1) {
        targetGray = target;
    } else {
        cv::cvtColor(target, targetGray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat diff;
    cv::absdiff(warpedGray, targetGray, diff);

    cv::Mat diff8;
    if (diff.depth() == CV_8U) {
        diff8 = diff;
    } else {
        cv::normalize(diff, diff8, 0, 255, cv::NORM_MINMAX, CV_8U);
    }

    cv::Mat colored;
    cv::applyColorMap(diff8, colored, cv::COLORMAP_TURBO);
    return colored;
}

void removeStaleDirectOutput(const fs::path& out) {
    std::error_code ec;
    if (fs::exists(out, ec)) {
        fs::remove(out, ec);
        if (ec) {
            IR_LOG_WARN("Failed to remove stale direct visualization: ", out.string());
        } else {
            IR_LOG_INFO("Removed stale direct visualization: ", out.string());
        }
    }
}

} // namespace

void DirectPipeline::resetStages() {
    _aligner.reset();
}

bool DirectPipeline::configureStages(const PipelineConfig& cfg) {
    if (cfg.direct_path.empty()) {
        IR_LOG_ERROR("DirectPipeline: missing direct config path.");
        return false;
    }

    _aligner = Factory::createDirectAligner(Config::load(cfg.direct_path));
    IR_LOG_INFO("DirectPipeline stages configured: aligner=", _aligner->name());
    return true;
}

bool DirectPipeline::runExtraction(RegistrationContext& ctx) {
    ctx.result.num_keypoints_first = 0;
    ctx.result.num_keypoints_second = 0;
    return true;
}

bool DirectPipeline::runAssociation(RegistrationContext& ctx) {
    ctx.result.num_raw_matches = 0;
    ctx.result.num_filtered_matches = 0;
    return true;
}

bool DirectPipeline::runEstimation(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_geometry_ms);

    if (!_aligner) {
        ctx.geometry_data.message = "no direct aligner configured";
        IR_LOG_ERROR("DirectPipeline::runEstimation: no direct aligner configured.");
        return false;
    }

    ctx.correspondence_source = "DIRECT";
    const bool ok = _aligner->align(ctx);

    // 直接法不再伪装成 keypoint match，摘要统计直接从 direct_data/geometry_data 读取。
    const int pairCount = static_cast<int>(
        std::max(ctx.direct_data.matches.size(),
                 std::min(ctx.direct_data.points1.size(), ctx.direct_data.points2.size())));
    ctx.result.num_raw_matches = pairCount;
    ctx.result.num_filtered_matches = pairCount;
    ctx.result.num_inliers = ctx.geometry_data.num_inliers;
    ctx.result.inlier_ratio = ctx.geometry_data.inlier_ratio;
    if (ctx.result.num_inliers == 0 && ctx.direct_data.valid) {
        // ECC/相位相关这类非点对方法可能只有 score，没有常规内点数，用 score 填入摘要置信度。
        ctx.result.num_inliers = pairCount;
        ctx.result.inlier_ratio = ctx.direct_data.score;
    }
    return ok;
}

std::string DirectPipeline::buildOutputStem(const RegistrationContext& ctx) const {
    return ctx.image1_path.stem().string() + "_" + ctx.image2_path.stem().string() + "_" +
           (_aligner ? _aligner->name() : std::string("DIRECT"));
}

bool DirectPipeline::saveOutputs(RegistrationContext& ctx) {
    if (_config.output_dir.empty()) {
        return true;
    }

    const fs::path directDir = _config.output_dir / "direct";
    std::error_code ec;
    // direct 子目录仅存放直接法专属可视化，通用 warped/blend 仍由 BasePipeline 统一输出。
    fs::create_directories(directDir, ec);

    const std::string stem = buildOutputStem(ctx);
    const bool hasPointPairs = !ctx.direct_data.points1.empty() && !ctx.direct_data.points2.empty();
    const fs::path matchesOut = directDir / (stem + "_matches.png");
    if (_config.draw_matches && hasPointPairs) {
        DrawMatches::Options matchOpt;
        matchOpt.draw_inliers_only = !ctx.direct_data.inlier_mask.empty();
        matchOpt.max_matches = _config.max_matches_drawn;
        const cv::Mat matchesVis = DrawMatches::render(ctx, matchOpt);
        if (!matchesVis.empty()) {
            cv::imwrite(matchesOut.string(), matchesVis);
            IR_LOG_INFO("Wrote direct matches visualization: ", matchesOut.string());
        }
    } else {
        // 配置关闭或当前直接法不产出点对时，清理同名旧图，避免输出目录误导复盘。
        removeStaleDirectOutput(matchesOut);
    }

    // 稠密光流只作为内部估计数据保留，不再输出伪彩色 flow 图，避免 direct 目录混入非配准质量图。
    removeStaleDirectOutput(directDir / (stem + "_flow.png"));

    if (_config.warp && !ctx.warped_image.empty() && !ctx.images.second.empty()) {
        const cv::Mat diffVis = renderWarpDifference(ctx.warped_image, ctx.images.second);
        if (!diffVis.empty()) {
            const fs::path out = directDir / (stem + "_warp_diff.png");
            cv::imwrite(out.string(), diffVis);
            IR_LOG_INFO("Wrote direct warp difference visualization: ", out.string());
        }
    }

    return BasePipeline::saveOutputs(ctx);
}

} // namespace ir
