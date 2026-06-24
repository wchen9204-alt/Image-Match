#include "pipeline/structure_pipeline.h"

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "core/config.h"
#include "core/factory.h"
#include "data/correspondence_view.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/visualization/structure/draw_structure_matches.h"
#include "utils/yaml_utils.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

bool restoreContourAssociatorGeometry(RegistrationContext& ctx,
                                       const cv::Point2d& assocTranslation,
                                       const cv::Mat& assocAffine,
                                       const std::vector<cv::DMatch>& assocInliers,
                                       double assocScore) {
    if (ctx.structure_data.type != StructureType::CONTOUR || assocInliers.empty()) {
        return false;
    }

    auto& gd = ctx.geometry_data;
    auto& md = ctx.structure_match_data;
    gd.type = GeometryType::AFFINE;
    if (!assocAffine.empty() && assocAffine.rows == 2 && assocAffine.cols == 3) {
        assocAffine.convertTo(gd.A, CV_64F);
    } else {
        gd.A = (cv::Mat_<double>(2, 3) << 1.0,
                0.0,
                assocTranslation.x,
                0.0,
                1.0,
                assocTranslation.y);
    }
    gd.valid = true;
    gd.num_inliers = static_cast<int>(assocInliers.size());
    gd.inlier_ratio =
        md.line_matches.empty()
            ? 0.0
            : static_cast<double>(assocInliers.size()) / static_cast<double>(md.line_matches.size());
    gd.message = "using contour associator geometry";

    md.affine = gd.A.clone();
    md.translation = {gd.A.at<double>(0, 2), gd.A.at<double>(1, 2)};
    md.inlier_line_matches = assocInliers;
    md.score = assocScore;
    return true;
}

void promoteStructureInliersFromGeometryMask(RegistrationContext& ctx) {
    auto& md = ctx.structure_match_data;
    const auto& gd = ctx.geometry_data;
    const CorrespondenceView view = buildStructureCorrespondenceView(ctx);

    std::vector<int> pointInlierCounts(md.line_matches.size(), 0);
    for (size_t i = 0; i < view.filtered.size() && i < gd.inlier_mask.size(); ++i) {
        if (!gd.inlier_mask[i]) {
            continue;
        }
        const int structureMatchIndex = view.filtered[i].imgIdx;
        if (structureMatchIndex >= 0 &&
            structureMatchIndex < static_cast<int>(pointInlierCounts.size())) {
            ++pointInlierCounts[static_cast<size_t>(structureMatchIndex)];
        }
    }

    md.inlier_line_matches.clear();
    const int requiredPoints = ctx.structure_data.type == StructureType::LINE ? 2 : 1;
    for (size_t i = 0; i < pointInlierCounts.size(); ++i) {
        if (pointInlierCounts[i] >= requiredPoints) {
            md.inlier_line_matches.push_back(md.line_matches[i]);
        }
    }
    md.score = md.line_matches.empty()
                   ? 0.0
                   : static_cast<double>(md.inlier_line_matches.size()) /
                         static_cast<double>(md.line_matches.size());
}

} // namespace

void StructurePipeline::resetStages() {
    _extractor.reset();
    _associator.reset();
    _geometry.reset();
    _filters.clear();
}

bool StructurePipeline::configureStages(const PipelineConfig& cfg) {
    // 1. 检查 structure YAML 路径，结构流水线依赖同一份配置创建提取器和关联器。
    if (cfg.structure_path.empty()) {
        IR_LOG_ERROR("StructurePipeline: missing structure config path.");
        return false;
    }
    // 2. 读取结构配置，并创建结构提取器、结构匹配器与通用几何估计器。
    const YAML::Node structureCfg = Config::load(cfg.structure_path);
    _extractor = Factory::createStructureExtractor(structureCfg);
    _associator = Factory::createStructureAssociator(structureCfg);
    if (!cfg.geometry_path.empty()) {
        const YAML::Node geometryCfg = Config::load(cfg.geometry_path);
        _geometry = Factory::createGeometryEstimator(geometryCfg);
    } else {
        IR_LOG_WARN("StructurePipeline: missing geometry config path; falling back to "
                    "association transform when available.");
    }

    // 3. 从 pipeline YAML 的 filters: 列表加载过滤链（与 KeypointPipeline 一致）。
    _filters.clear();
    for (const auto& fp : cfg.filter_paths) {
        _filters.push_back(Factory::createFilter(Config::load(fp)));
    }

    // 4. 输出当前结构方法组合，方便批量实验时核对配置是否生效。
    IR_LOG_INFO("StructurePipeline stages configured: extractor=",
                _extractor->name(),
                ", associator=",
                _associator->name(),
                ", geometry=",
                (_geometry ? _geometry->name() : std::string("NONE")),
                ", filters=",
                _filters.size());
    return true;
}

bool StructurePipeline::runExtraction(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_extract_ms);

    // 1. 检查结构提取器是否已经由 configureStages 创建。
    if (!_extractor) {
        IR_LOG_ERROR("StructurePipeline::runExtraction: no structure extractor configured.");
        return false;
    }

    // 2. 执行结构响应提取，结果写入 ctx.structure_data。
    const bool ok = _extractor->extract(ctx);

    // 3. 按当前结构类型统计 source / target 的结构数量。
    ctx.result.num_structures_first =
        ctx.structure_data.first.primitiveCount(ctx.structure_data.type);
    ctx.result.num_structures_second =
        ctx.structure_data.second.primitiveCount(ctx.structure_data.type);
    return ok;
}

bool StructurePipeline::runAssociation(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_match_ms);

    // 1. 检查结构匹配器和结构响应数据是否可用。
    if (!_associator) {
        IR_LOG_ERROR("StructurePipeline::runAssociation: no structure associator configured.");
        return false;
    }
    if (ctx.structure_data.empty()) {
        IR_LOG_ERROR("StructurePipeline::runAssociation: structure data is empty.");
        return false;
    }

    // 2. 执行结构匹配，关联器负责写入 raw_matches_knn / filtered_matches 等。
    const bool ok = _associator->associate(ctx);
    if (!ok) {
        return false;
    }

    // 3. 对有线匹配数据的关联器，执行过滤链。
    if (!_filters.empty() && !ctx.structure_match_data.raw_matches_knn.empty()) {
        if (!runFilters(ctx)) {
            IR_LOG_WARN("StructurePipeline::runAssociation: filter chain rejected all matches.");
            return false;
        }
    }

    // 4. 同步匹配计数到运行摘要。
    if (!ctx.structure_match_data.line_matches.empty() ||
        !ctx.structure_match_data.inlier_line_matches.empty()) {
        ctx.result.num_raw_matches =
            static_cast<int>(ctx.structure_match_data.raw_matches_knn.size());
        ctx.result.num_filtered_matches =
            static_cast<int>(ctx.structure_match_data.line_matches.size());
    } else {
        ctx.result.num_raw_matches = ok ? 1 : 0;
        ctx.result.num_filtered_matches = ok ? 1 : 0;
    }
    return ok;
}

bool StructurePipeline::runFilters(RegistrationContext& ctx) {
    auto& md = ctx.structure_match_data;

    // 1. 关联器已种子 filtered_matches（top-1），直接交给过滤链逐级处理。
    //    RatioTest 等需要 KNN 的过滤器会从 raw_matches_knn 重新计算。
    for (const auto& f : _filters) {
        if (!f) {
            continue;
        }
        if (!f->apply(ctx)) {
            IR_LOG_WARN("StructurePipeline filter ", f->name(), " returned false");
        }
    }

    // 2. 过滤完成后同步 filtered_matches → line_matches。
    md.line_matches = md.filtered_matches;
    md.inlier_line_matches = md.line_matches;
    md.valid = !md.line_matches.empty();

    IR_LOG_INFO("StructurePipeline filters done: ",
                md.line_matches.size(),
                " / ",
                md.raw_matches_knn.size(),
                " line matches after filtering");
    return md.valid;
}

bool StructurePipeline::runEstimation(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_geometry_ms);

    // 1. 初始化几何结果。
    auto& gd = ctx.geometry_data;
    gd.clear();
    ctx.correspondence_source = "STRUCTURE";

    // 2. 检查结构匹配是否给出了有效结果，无效时保留失败原因。
    if (!ctx.structure_match_data.valid) {
        gd.message = ctx.structure_match_data.message.empty()
                         ? "structure match result is invalid"
                         : ctx.structure_match_data.message;
        IR_LOG_WARN("StructurePipeline::runEstimation: ", gd.message);
        return false;
    }

    // 3. 有结构匹配时，几何估计器直接读取 CorrespondenceView，不再临时伪装成 keypoint match。
    if (_geometry && !ctx.structure_match_data.line_matches.empty()) {
        if (buildStructureCorrespondenceView(ctx).empty()) {
            gd.message = "no valid structure correspondences for geometry estimation";
            IR_LOG_WARN("StructurePipeline::runEstimation: ", gd.message);
            return false;
        }

        const cv::Point2d assocTranslation = ctx.structure_match_data.translation;
        const cv::Mat assocAffine = ctx.structure_match_data.affine.clone();
        const std::vector<cv::DMatch> assocInliers =
            ctx.structure_match_data.inlier_line_matches;
        const double assocScore = ctx.structure_match_data.score;

        const bool ok = _geometry->estimate(ctx);

        if (ok) {
            promoteStructureInliersFromGeometryMask(ctx);

            const bool geometryDowngradesContourInliers =
                ctx.structure_data.type == StructureType::CONTOUR && !assocInliers.empty() &&
                ctx.structure_match_data.inlier_line_matches.size() < assocInliers.size();
            if (geometryDowngradesContourInliers) {
                IR_LOG_WARN("StructurePipeline contour geometry downgraded inliers from ",
                            assocInliers.size(),
                            " to ",
                            ctx.structure_match_data.inlier_line_matches.size(),
                            "; keeping contour associator inliers");
                ctx.structure_match_data.inlier_line_matches = assocInliers;
                ctx.structure_match_data.score = assocScore;
                ctx.structure_match_data.affine = assocAffine;
                ctx.structure_match_data.translation = assocTranslation;
            } else {
                ctx.structure_match_data.affine =
                    (ctx.geometry_data.A.empty() ? cv::Mat{} : ctx.geometry_data.A.clone());
                if (!ctx.geometry_data.A.empty()) {
                    ctx.structure_match_data.translation = {ctx.geometry_data.A.at<double>(0, 2),
                                                            ctx.geometry_data.A.at<double>(1, 2)};
                } else if (!ctx.geometry_data.H.empty()) {
                    ctx.structure_match_data.translation =
                        {ctx.geometry_data.H.at<double>(0, 2) /
                             ctx.geometry_data.H.at<double>(2, 2),
                         ctx.geometry_data.H.at<double>(1, 2) /
                             ctx.geometry_data.H.at<double>(2, 2)};
                }
            }
        } else {
            ctx.structure_match_data.affine = assocAffine;
            ctx.structure_match_data.translation = assocTranslation;
            ctx.structure_match_data.inlier_line_matches = assocInliers;
            ctx.structure_match_data.score = assocScore;
            if (restoreContourAssociatorGeometry(
                    ctx, assocTranslation, assocAffine, assocInliers, assocScore)) {
                IR_LOG_WARN("StructurePipeline contour geometry estimator failed; using "
                            "contour associator affine instead");
                ctx.result.num_inliers =
                    static_cast<int>(ctx.structure_match_data.inlier_line_matches.size());
                ctx.result.inlier_ratio = ctx.structure_match_data.score;
                return true;
            }
        }

        // 保留关联器/预筛后的匹配数量，不在几何阶段用内点数覆盖，
        // 这样 summary 里可以区分“筛后匹配数”和“最终几何内点数”。
        ctx.result.num_inliers =
            static_cast<int>(ctx.structure_match_data.inlier_line_matches.size());
        ctx.result.inlier_ratio = ctx.structure_match_data.score;
        if (!ok) {
            ctx.structure_match_data.message = ctx.geometry_data.message;
            return false;
        }

        IR_LOG_INFO("StructurePipeline geometry estimated by ",
                    _geometry->name(),
                    ", correspondence_inliers=",
                    ctx.geometry_data.num_inliers,
                    ", line_inliers=",
                    ctx.structure_match_data.inlier_line_matches.size(),
                    ", score=",
                    ctx.structure_match_data.score);
        return true;
    }

    // 4. 无几何估计器或无线段匹配时的回退路径：
    //    响应图关联器（PhaseCorrelate / Chamfer / Hausdorff / ICP）直接给出平移，
    //    线段关联器在无 geometry 配置时也以平均平移作为兜底。
    gd.type = GeometryType::AFFINE;
    const cv::Point2d shift = ctx.structure_match_data.translation;
    const cv::Mat& affine = ctx.structure_match_data.affine;
    if (!affine.empty() && affine.rows == 2 && affine.cols == 3) {
        affine.convertTo(gd.A, CV_64F);
    } else {
        gd.A = (cv::Mat_<double>(2, 3) << 1.0, 0.0, shift.x, 0.0, 1.0, shift.y);
    }
    gd.valid = true;

    // 5. 同步内点数和得分到通用运行摘要。
    gd.num_inliers = ctx.structure_match_data.inlier_line_matches.empty()
                         ? ctx.result.num_structures_first
                         : static_cast<int>(ctx.structure_match_data.inlier_line_matches.size());
    gd.inlier_ratio = ctx.structure_match_data.score;
    ctx.result.num_inliers = gd.num_inliers;
    ctx.result.inlier_ratio = gd.inlier_ratio;

    IR_LOG_INFO("StructurePipeline fallback affine dx=",
                gd.A.at<double>(0, 2),
                ", dy=",
                gd.A.at<double>(1, 2),
                ", score=",
                ctx.structure_match_data.score);
    return true;
}

std::string StructurePipeline::buildOutputStem(const RegistrationContext& ctx) const {
    const std::string sampleStem = ctx.image1_path.stem().string() + "_" +
                                   ctx.image2_path.stem().string();
    return sampleStem + "_" + (_extractor ? _extractor->name() : std::string("STRUCTURE")) +
           "_" + (_associator ? _associator->name() : std::string("MATCH"));
}

bool StructurePipeline::saveOutputs(RegistrationContext& ctx) {
    if (_config.output_dir.empty()) {
        return true;
    }

    // 1. 创建结构响应图和结构匹配图的输出目录。
    const fs::path structureDir = _config.output_dir / "structures";
    const fs::path matchesDir = _config.output_dir / "matches";
    std::error_code ec;
    fs::create_directories(structureDir, ec);
    fs::create_directories(matchesDir, ec);

    const std::string stem = buildOutputStem(ctx);
    const std::string sampleStem = ctx.image1_path.stem().string() + "_" +
                                   ctx.image2_path.stem().string();
    const std::string structureStem =
        sampleStem + "_" + (_extractor ? _extractor->outputLabel() : std::string("STRUCTURE"));

    // 2. 保存结构提取器生成的 source / target 响应图。
    if (!ctx.structure_data.first.response.empty()) {
        const fs::path out = structureDir / (structureStem + "_source_structure.png");
        cv::imwrite(out.string(), ctx.structure_data.first.response);
        IR_LOG_INFO("Wrote source structure visualization: ", out.string());
    }
    if (!ctx.structure_data.second.response.empty()) {
        const fs::path out = structureDir / (structureStem + "_target_structure.png");
        cv::imwrite(out.string(), ctx.structure_data.second.response);
        IR_LOG_INFO("Wrote target structure visualization: ", out.string());
    }

    // 3. 按配置保存结构匹配连线图，便于和点特征 matches 输出对照。
    if (_config.draw_matches) {
        cv::Mat vis;
        const bool preferInliers =
            _config.draw_inliers_only && !ctx.structure_match_data.inlier_line_matches.empty();
        const std::vector<cv::DMatch>& preferredMatches =
            preferInliers ? ctx.structure_match_data.inlier_line_matches
                          : ctx.structure_match_data.line_matches;

        if (ctx.structure_data.type == StructureType::LINE && !preferredMatches.empty()) {
            vis = renderLineSegmentMatches(ctx, preferredMatches, _config.max_matches_drawn);
        } else if (ctx.structure_data.type == StructureType::CONTOUR &&
                   !preferredMatches.empty()) {
            vis = renderContourMatches(ctx, preferredMatches, _config.max_matches_drawn);
        }
        if (vis.empty()) {
            vis = renderStructureMatches(ctx, _config.max_matches_drawn);
        }
        if (!vis.empty()) {
            const fs::path out = matchesDir / (stem + "_structure_matches.png");
            if (cv::imwrite(out.string(), vis)) {
                IR_LOG_INFO("Wrote structure matches visualization: ", out.string());
            } else {
                IR_LOG_WARN("Failed to write structure matches visualization: ", out.string());
            }
        } else {
            IR_LOG_WARN("Structure matches visualization skipped: no drawable correspondences.");
        }
    }

    // 4. 委托基类保存 originals / warped / blend 等通用输出。
    return BasePipeline::saveOutputs(ctx);
}

} // namespace ir

