#include "pipeline/direct_pipeline.h"

#include <algorithm>
#include <filesystem>
#include <string>

#include "core/config.h"
#include "core/factory.h"
#include "utils/logger.h"
#include "utils/timer.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

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

    // Direct methods report summary stats from direct_data/geometry_data.
    const int pairCount = static_cast<int>(
        std::max(ctx.direct_data.matches.size(),
                 std::min(ctx.direct_data.points1.size(), ctx.direct_data.points2.size())));
    ctx.result.num_raw_matches = pairCount;
    ctx.result.num_filtered_matches = pairCount;
    ctx.result.num_inliers = ctx.geometry_data.num_inliers;
    ctx.result.inlier_ratio = ctx.geometry_data.inlier_ratio;
    if (ctx.result.num_inliers == 0 && ctx.direct_data.valid) {
        // Methods without point pairs may only provide a score; reuse it as confidence.
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
    if (_aligner) {
        const std::string stem = buildOutputStem(ctx);
        removeStaleDirectOutput(directDir / (stem + "_matches.png"));
        removeStaleDirectOutput(directDir / (stem + "_flow.png"));
        removeStaleDirectOutput(directDir / (stem + "_warp_diff.png"));
    }

    return BasePipeline::saveOutputs(ctx);
}

} // namespace ir
