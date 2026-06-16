#include "pipeline/learning_pipeline.h"

#include <filesystem>
#include <string>

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>

#include "core/config.h"
#include "core/factory.h"
#include "learning/python_learning_matcher.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/visualization/draw_inliers.h"
#include "utils/visualization/draw_matches.h"

namespace fs = std::filesystem;

namespace ir {

/// 清空学习流水线持有的阶段组件，保证重新 configure 时不会复用旧模型或旧几何估计器。
void LearningPipeline::resetStages() {
    _matcher.reset();
    _geometry.reset();
}

/// 按 pipeline 配置创建学习匹配器和几何估计器。
bool LearningPipeline::configureStages(const PipelineConfig& cfg) {
    // 1. learning 和 geometry 是学习流水线的两个必需阶段，缺任一阶段都无法继续。
    if (cfg.learning_path.empty()) {
        IR_LOG_ERROR("LearningPipeline: missing learning config path.");
        return false;
    }
    if (cfg.geometry_path.empty()) {
        IR_LOG_ERROR("LearningPipeline: missing geometry config path.");
        return false;
    }

    // 2. 学习匹配器由独立 YAML 配置驱动，当前实现通过 Python bridge 调用外部深度模型。
    const YAML::Node learningCfg = Config::load(cfg.learning_path);
    _matcher = std::make_shared<PythonLearningMatcher>(learningCfg, cfg.learning_path.parent_path());

    // 3. 几何估计器复用现有工厂，保持 Homography/Affine/Rigid/Similarity 可配置。
    _geometry = Factory::createGeometryEstimator(Config::load(cfg.geometry_path));

    IR_LOG_INFO("LearningPipeline stages configured: matcher=",
                _matcher->name(),
                ", geometry=",
                _geometry->name());
    return true;
}

/// 学习方法没有独立的 C++ 提取阶段；匹配点由后续 Python matcher 一次性产出。
bool LearningPipeline::runExtraction(RegistrationContext& ctx) {
    ctx.result.num_keypoints_first = 0;
    ctx.result.num_keypoints_second = 0;
    return true;
}

/// 调用深度学习匹配器，并把 Python 输出的点对统计写入运行结果。
bool LearningPipeline::runAssociation(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_match_ms);
    // 1. 防御未配置状态，避免空 matcher 在批处理失败样本中继续下沉。
    if (!_matcher) {
        IR_LOG_ERROR("LearningPipeline::runAssociation: no learning matcher configured.");
        return false;
    }

    // 2. PythonLearningMatcher 会负责执行脚本、读取 JSON，并转换为平台统一点对容器。
    if (!_matcher->match(ctx)) {
        return false;
    }

    // 3. 将学习点对数量同步到通用结果字段，便于 summary/CSV 复用已有列。
    int rawCount = 0;
    for (const auto& row : ctx.keypoint_match_data.raw_matches_by_query) {
        rawCount += static_cast<int>(row.size());
    }
    ctx.result.num_keypoints_first = static_cast<int>(ctx.keypoint_data.first.keypoints.size());
    ctx.result.num_keypoints_second = static_cast<int>(ctx.keypoint_data.second.keypoints.size());
    ctx.result.num_raw_matches = rawCount;
    ctx.result.num_filtered_matches = static_cast<int>(ctx.keypoint_match_data.filtered_matches.size());
    return true;
}

/// 使用学习方法产出的对应点视图执行几何估计。
bool LearningPipeline::runEstimation(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_geometry_ms);
    // 1. 几何估计器缺失时直接失败，并写入 geometry_data 供摘要说明原因。
    if (!_geometry) {
        ctx.geometry_data.message = "no geometry estimator configured";
        IR_LOG_ERROR("LearningPipeline::runEstimation: no geometry estimator configured.");
        return false;
    }

    // 2. 显式标记来源为 LEARNING，避免深度学习点对被误识别为传统 keypoint 来源。
    ctx.correspondence_source = "LEARNING";
    const bool ok = _geometry->estimate(ctx);

    // 3. 将几何阶段结果同步到顶层运行结果，保持输出摘要与其它 pipeline 一致。
    ctx.result.num_inliers = ctx.geometry_data.num_inliers;
    ctx.result.inlier_ratio = ctx.geometry_data.inlier_ratio;
    return ok;
}

/// 生成学习方法输出文件 stem，包含样本名、学习 matcher 名称和几何模型类型。
std::string LearningPipeline::buildOutputStem(const RegistrationContext& ctx) const {
    return ctx.image1_path.stem().string() + "_" + ctx.image2_path.stem().string() + "_" +
           (_matcher ? _matcher->name() : std::string("LEARNING")) + "_" +
           (_geometry ? toString(_geometry->type()) : std::string("UNK"));
}

/// 保存学习方法的匹配/内点可视化，并复用 BasePipeline 保存原图、warp、blend 和摘要。
bool LearningPipeline::saveOutputs(RegistrationContext& ctx) {
    if (_config.output_dir.empty()) {
        return true;
    }

    // 1. 学习方法的专属可视化放入 learning 子目录，避免和原图/warp 输出混在一起。
    const fs::path learningDir = _config.output_dir / "learning";
    std::error_code ec;
    fs::create_directories(learningDir, ec);

    const std::string stem = buildOutputStem(ctx);
    if (_config.draw_matches) {
        // 2. 保存全部学习匹配，便于观察模型原始输出的空间分布。
        DrawMatches::Options allOpt;
        allOpt.draw_raw_matches = true;
        allOpt.max_matches = _config.max_matches_drawn;
        const cv::Mat allVis = DrawMatches::render(ctx, allOpt);
        if (!allVis.empty()) {
            const fs::path out = learningDir / (stem + "_all_match.png");
            cv::imwrite(out.string(), allVis);
            IR_LOG_INFO("Wrote learning matches visualization: ", out.string());
        }

        // 3. 保存几何估计接受的内点，便于诊断匹配质量和模型筛选效果。
        DrawInliers::Options inlierOpt;
        inlierOpt.max_inliers = _config.max_matches_drawn;
        const cv::Mat inlierVis = DrawInliers::render(ctx, inlierOpt);
        if (!inlierVis.empty()) {
            const fs::path out = learningDir / (stem + "_inlier_match.png");
            cv::imwrite(out.string(), inlierVis);
            IR_LOG_INFO("Wrote learning inlier visualization: ", out.string());
        }
    }

    // 4. 其余通用输出交给 BasePipeline，保持不同方法族的目录和摘要格式一致。
    return BasePipeline::saveOutputs(ctx);
}

} // namespace ir


