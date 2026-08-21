#include "pipeline/direct_feature_initializer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include <opencv2/imgproc.hpp>

#include "core/factory.h"
#include "evaluator/quality/warp_quality_evaluator.h"
#include "pipeline/base_pipeline_helpers.h"
#include "utils/logger.h"
#include "utils/string_utils.h"

namespace ir {

namespace {

using base_pipeline_helpers::buildForegroundMask;
using base_pipeline_helpers::computeInlierSpatialCoverage;

struct ContextSnapshot {
    KeypointData keypoint_data;
    KeypointMatchData keypoint_match_data;
    GeometryData geometry_data;
    std::string correspondence_source;
};

void restoreContext(RegistrationContext& ctx, const ContextSnapshot& snapshot) {
    ctx.keypoint_data = snapshot.keypoint_data;
    ctx.keypoint_match_data = snapshot.keypoint_match_data;
    ctx.geometry_data = snapshot.geometry_data;
    ctx.correspondence_source = snapshot.correspondence_source;
}

bool matrixFromGeometry(const GeometryData& geometry, cv::Mat& matrix) {
    matrix.release();
    if (geometry.type == GeometryType::HOMOGRAPHY && !geometry.H.empty()) {
        geometry.H.convertTo(matrix, CV_64F);
        return matrix.rows >= 3 && matrix.cols >= 3;
    }
    if (!geometry.A.empty()) {
        geometry.A.convertTo(matrix, CV_64F);
        return matrix.rows >= 2 && matrix.cols >= 3;
    }
    if (!geometry.H.empty()) {
        geometry.H.convertTo(matrix, CV_64F);
        return matrix.rows >= 3 && matrix.cols >= 3;
    }
    return false;
}

bool finiteMetric(double value) {
    return std::isfinite(value) && value >= 0.0;
}

bool captureGeometryEstimate(const GeometryData& geometry,
                             FeatureInitializerData& data,
                             std::string& reason) {
    if (!geometry.valid) {
        reason = geometry.message.empty() ? "geometry estimator returned invalid result"
                                          : geometry.message;
        return false;
    }

    cv::Mat matrix;
    if (!matrixFromGeometry(geometry, matrix)) {
        reason = "initializer geometry has no usable transform matrix";
        return false;
    }

    data.type = geometry.type;
    data.num_inliers = geometry.num_inliers;
    data.inlier_ratio = geometry.inlier_ratio;
    data.A = geometry.A.clone();
    data.H = geometry.H.clone();
    return true;
}

bool validateGeometryGate(const PipelineConfig& cfg,
                          const GeometryData& geometry,
                          std::string& reason) {
    const auto& acceptance = cfg.feature_initializer.acceptance;
    if (geometry.num_inliers < acceptance.min_inliers) {
        reason = "initializer inliers below threshold";
        return false;
    }
    if (acceptance.min_inlier_ratio >= 0.0 &&
        geometry.inlier_ratio < acceptance.min_inlier_ratio) {
        reason = "initializer inlier ratio below threshold";
        return false;
    }
    return true;
}

bool validateSpatialCoverage(const PipelineConfig& cfg,
                             RegistrationContext& ctx,
                             const cv::Mat& sourceMask,
                             const cv::Mat& targetMask,
                             FeatureInitializerData& data,
                             std::string& reason) {
    double sourceCoverage = -1.0;
    double targetCoverage = -1.0;
    data.inlier_spatial_coverage =
        computeInlierSpatialCoverage(ctx.keypoint_data.first.keypoints,
                                     ctx.keypoint_data.second.keypoints,
                                     ctx.keypoint_match_data.inlier_matches,
                                     sourceMask,
                                     targetMask,
                                     sourceCoverage,
                                     targetCoverage);

    const auto& acceptance = cfg.feature_initializer.acceptance;
    if (acceptance.min_inlier_spatial_coverage >= 0.0 &&
        (!finiteMetric(data.inlier_spatial_coverage) ||
         data.inlier_spatial_coverage < acceptance.min_inlier_spatial_coverage)) {
        reason = "initializer inlier spatial coverage below threshold";
        return false;
    }
    return true;
}

bool validateCandidate(const PipelineConfig& cfg,
                       RegistrationContext& ctx,
                       const std::string& candidateName,
                       FeatureInitializerData& data,
                       std::string& reason) {
    if (!validateGeometryGate(cfg, ctx.geometry_data, reason)) {
        return false;
    }

    cv::Mat matrix;
    if (!matrixFromGeometry(ctx.geometry_data, matrix)) {
        reason = "initializer geometry has no usable transform matrix";
        return false;
    }

    cv::Mat sourceMask;
    cv::Mat targetMask;
    const int thresholdValue =
        std::clamp(cfg.feature_initializer.validation.overlap.foreground_threshold, 0, 255);
    if (!buildForegroundMask(ctx.images.first, thresholdValue, sourceMask) ||
        !buildForegroundMask(ctx.images.second, thresholdValue, targetMask)) {
        reason = "failed to build initializer foreground masks";
        return false;
    }

    if (!validateSpatialCoverage(cfg, ctx, sourceMask, targetMask, data, reason)) {
        return false;
    }

    warp_quality::WarpQualityResult quality;
    if (!warp_quality::evaluateWarpQuality(warp_quality::makeInitializerWarpQualityOptions(cfg),
                                           ctx.images.first,
                                           ctx.images.second,
                                           matrix,
                                           cv::Mat{},
                                           quality)) {
        reason = "initializer " + quality.message;
        return false;
    }

    data.warp_overlap_containment = quality.overlap_containment;
    data.warp_source_coverage = quality.source_coverage;
    data.warp_target_coverage = quality.target_coverage;
    data.warp_height_diff_valid_count = quality.height_diff_valid_count;
    data.warp_height_diff_overlap_ratio = quality.height_diff_overlap_ratio;
    data.warp_height_diff_mean = quality.height_diff_mean;
    data.warp_height_diff_p50 = quality.height_diff_p50;
    data.warp_height_diff_p75 = quality.height_diff_p75;
    data.warp_height_diff_p90 = quality.height_diff_p90;
    data.warp_height_diff_p95 = quality.height_diff_p95;
    data.warp_height_diff_max = quality.height_diff_max;
    return true;
}

} // namespace

void DirectFeatureInitializer::reset() {
    _enabled = false;
    _has_candidate = false;
    _candidate = {};
    _matcher.reset();
    _filters.clear();
    _geometry.reset();
    _config = PipelineConfig{};
}

bool DirectFeatureInitializer::configure(const PipelineConfig& cfg) {
    reset();
    _config = cfg;
    _enabled = cfg.feature_initializer.enabled;
    if (!_enabled) {
        return true;
    }

    // 1. 点特征初始化只读取单个候选；若未配置或不可用，则直接关闭 initializer。
    const auto& candidateCfg = cfg.feature_initializer.candidate;
    if (!candidateCfg.keypoint_path.empty()) {
        _candidate.name = string_utils::toUpperAscii(candidateCfg.name);
        _candidate.extractor =
            Factory::createKeypointExtractor(Config::load(candidateCfg.keypoint_path));
        if (_candidate.name.empty() && _candidate.extractor) {
            _candidate.name = _candidate.extractor->name();
        }
        if (!_candidate.extractor) {
            IR_LOG_WARN("DirectFeatureInitializer disabled: failed to create keypoint candidate ",
                        candidateCfg.keypoint_path.string());
            _enabled = false;
            return true;
        }
        _has_candidate = true;
    }

    if (!_has_candidate) {
        IR_LOG_WARN("DirectFeatureInitializer disabled: no keypoint candidate configured.");
        _enabled = false;
        return true;
    }

    // 2. matcher 和 geometry 是单候选共享链路的一部分。
    if (!cfg.feature_initializer.matcher_path.empty()) {
        _matcher = Factory::createMatcher(Config::load(cfg.feature_initializer.matcher_path));
    }
    if (!cfg.feature_initializer.geometry_path.empty()) {
        _geometry =
            Factory::createGeometryEstimator(Config::load(cfg.feature_initializer.geometry_path));
    }
    for (const auto& filterPath : cfg.feature_initializer.filter_paths) {
        _filters.push_back(Factory::createFilter(Config::load(filterPath)));
    }

    if (!_matcher || !_geometry) {
        IR_LOG_WARN("DirectFeatureInitializer disabled: matcher or geometry is not configured.");
        _enabled = false;
        return true;
    }

    IR_LOG_INFO("DirectFeatureInitializer configured: candidate=",
                _candidate.name,
                ", filters=",
                static_cast<int>(_filters.size()),
                ", geometry=",
                _geometry->name());
    return true;
}

bool DirectFeatureInitializer::run(RegistrationContext& ctx) {
    auto& output = ctx.feature_initializer_data;
    output.clear();
    output.attempted = _enabled;
    if (!_enabled || !_has_candidate) {
        output.message = "feature initializer disabled";
        return false;
    }

    const ContextSnapshot snapshot{
        ctx.keypoint_data,
        ctx.keypoint_match_data,
        ctx.geometry_data,
        ctx.correspondence_source
    };

    FeatureInitializerData best;
    best.attempted = true;
    // 仅保存“有合法矩阵但未通过质量门控”的候选；不会参与最终结果选择。
    FeatureInitializerData estimatedOnly;
    std::vector<std::string> rejectMessages;

    // 1. 候选从干净的点特征上下文开始，避免残留匹配污染初始化判断。
    ctx.keypoint_data.clear();
    ctx.keypoint_match_data.clear();
    ctx.geometry_data.clear();
    ctx.correspondence_source = "KEYPOINT";

    std::string rejectReason;
    bool stageOk = _candidate.extractor && _candidate.extractor->extract(ctx);
    if (stageOk && _matcher) {
        stageOk = _matcher->match(ctx);
    }
    if (stageOk) {
        ctx.keypoint_match_data.seedFilteredMatchesFromRaw();
        for (const auto& filter : _filters) {
            if (filter && !filter->apply(ctx)) {
                IR_LOG_WARN("DirectFeatureInitializer filter '",
                            filter->name(),
                            "' returned false for candidate ",
                            _candidate.name);
            }
        }
        if (ctx.keypoint_match_data.filtered_matches.empty() &&
            !ctx.keypoint_match_data.raw_matches.empty()) {
            ctx.keypoint_match_data.restoreFilteredMatchesFromRaw();
            IR_LOG_WARN("DirectFeatureInitializer restored raw matches for candidate ", _candidate.name);
        }
    }
    if (stageOk && _geometry) {
        stageOk = _geometry->estimate(ctx);
    }
    if (!stageOk) {
        rejectReason = ctx.geometry_data.message.empty() ? "feature initializer stage failed"
                                                         : ctx.geometry_data.message;
    }

    FeatureInitializerData candidateData;
    candidateData.attempted = true;
    candidateData.method = _candidate.name;
    candidateData.num_keypoints_first = static_cast<int>(ctx.keypoint_data.first.keypoints.size());
    candidateData.num_keypoints_second = static_cast<int>(ctx.keypoint_data.second.keypoints.size());
    candidateData.num_raw_matches = static_cast<int>(ctx.keypoint_match_data.raw_matches.size());
    candidateData.num_filtered_matches = static_cast<int>(ctx.keypoint_match_data.filtered_matches.size());

    std::string matrixReason;
    const bool hasEstimatedMatrix =
        captureGeometryEstimate(ctx.geometry_data, candidateData, matrixReason);
    if (stageOk && !hasEstimatedMatrix) {
        rejectReason = matrixReason;
    }

    if (stageOk && hasEstimatedMatrix &&
        validateCandidate(_config, ctx, _candidate.name, candidateData, rejectReason)) {
        candidateData.accepted = true;
        candidateData.seed_available = true;
        candidateData.message = "accepted";
        best = candidateData;
        IR_LOG_INFO("DirectFeatureInitializer accepted ",
                    _candidate.name,
                    ": inliers=",
                    candidateData.num_inliers,
                    ", ratio=",
                    candidateData.inlier_ratio,
                    ", spatial=",
                    candidateData.inlier_spatial_coverage,
                    ", height_diff_count=",
                    candidateData.warp_height_diff_valid_count,
                    ", height_diff_overlap_ratio=",
                    candidateData.warp_height_diff_overlap_ratio,
                    ", height_diff_mean=",
                    candidateData.warp_height_diff_mean,
                    ", height_diff_p50=",
                    candidateData.warp_height_diff_p50,
                    ", height_diff_p75=",
                    candidateData.warp_height_diff_p75,
                    ", height_diff_p90=",
                    candidateData.warp_height_diff_p90,
                    ", height_diff_p95=",
                    candidateData.warp_height_diff_p95,
                    ", height_diff_max=",
                     candidateData.warp_height_diff_max);
    } else {
        rejectMessages.push_back(_candidate.name + ":" + rejectReason);
        IR_LOG_INFO("DirectFeatureInitializer rejected ", _candidate.name, ": ", rejectReason);
        estimatedOnly = candidateData;
        estimatedOnly.message = "visualization-only: " + rejectReason;
        if (hasEstimatedMatrix &&
            _config.feature_initializer.seed_mode ==
                FeatureInitializerSeedMode::ESTIMATED_WHEN_AVAILABLE) {
            estimatedOnly.seed_available = true;
            estimatedOnly.message = "seed-only: " + rejectReason;
            IR_LOG_INFO("DirectFeatureInitializer keeps rejected estimate as direct seed: ",
                        _candidate.name);
        }
    }

    restoreContext(ctx, snapshot);

    output = best.accepted ? best
                           : estimatedOnly;
    output.attempted = true;
    if (best.accepted) {
        output.message = "accepted";
        return true;
    }
    std::ostringstream oss;
    oss << "no feature initializer candidate accepted";
    if (!rejectMessages.empty()) {
        oss << " (";
        for (size_t i = 0; i < rejectMessages.size(); ++i) {
            if (i > 0) {
                oss << "; ";
            }
            oss << rejectMessages[i];
        }
        oss << ")";
    }
    output.message = oss.str();
    return false;
}

} // namespace ir
