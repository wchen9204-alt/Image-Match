#include "pipeline/structure_pipeline.h"

#include <filesystem>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/config.h"
#include "core/factory.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/yaml_utils.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

// 将结构响应图转换为相位相关估计可用的浮点图，并用轻微模糊降低稀疏噪声影响。
bool preparePhaseImage(const cv::Mat& mask, cv::Mat& out, int blurKernel) {
    // 1. 空响应图无法参与相位相关估计，直接判定失败。
    if (mask.empty()) {
        return false;
    }

    // 2. 保证输入为单通道灰度响应图。
    cv::Mat src;
    if (mask.channels() == 1) {
        src = mask;
    } else {
        cv::cvtColor(mask, src, cv::COLOR_BGR2GRAY);
    }

    // 3. 全黑结构图没有可对齐信息，也直接判定失败。
    if (cv::countNonZero(src) == 0) {
        return false;
    }

    // 4. 转为归一化浮点图，并按配置做轻微平滑。
    src.convertTo(out, CV_32F, 1.0 / 255.0);
    if (blurKernel >= 3) {
        if (blurKernel % 2 == 0) {
            ++blurKernel;
        }
        cv::GaussianBlur(out, out, cv::Size(blurKernel, blurKernel), 0.0);
    }
    return true;
}

} // namespace

void StructurePipeline::resetStages() {
    // 1. 释放结构提取器，避免下一次 configure 复用旧配置。
    _extractor.reset();

    // 2. 恢复结构估计参数默认值。
    _responseThreshold = 0.01;
    _phaseBlurKernel = 5;
}

bool StructurePipeline::configureStages(const PipelineConfig& cfg) {
    // 1. 结构法必须提供 structure 子配置，否则无法确定使用边缘、直线还是轮廓。
    if (cfg.structure_path.empty()) {
        IR_LOG_ERROR("StructurePipeline: missing structure config path.");
        return false;
    }

    // 2. 加载 structure YAML，并根据 type 创建具体结构提取器。
    const YAML::Node structureCfg = Config::load(cfg.structure_path);
    _extractor = Factory::createStructureExtractor(structureCfg);

    // 3. 读取结构响应图估计阶段的通用参数。
    const YAML::Node estimation = structureCfg["estimation"];
    _responseThreshold = yaml_utils::getDouble(estimation, "responseThreshold", 0.01);
    _phaseBlurKernel = yaml_utils::getInt(estimation, "phaseBlurKernel", 5);

    IR_LOG_INFO("StructurePipeline stages configured: extractor=",
                _extractor->name(),
                ", responseThreshold=",
                _responseThreshold,
                ", phaseBlurKernel=",
                _phaseBlurKernel);
    return true;
}

bool StructurePipeline::runExtraction(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_extract_ms);

    // 1. 检查结构提取器是否已由 configureStages 成功装配。
    if (!_extractor) {
        IR_LOG_ERROR("StructurePipeline::runExtraction: no structure extractor configured.");
        return false;
    }

    // 2. 调用具体结构提取器，结果写入 ctx.structure_data。
    const bool ok = _extractor->extract(ctx);

    // 3. 按结构类型统计数量，供终端摘要和汇总表使用。
    ctx.result.num_structures_first =
        ctx.structure_data.first.primitiveCount(ctx.structure_data.type);
    ctx.result.num_structures_second =
        ctx.structure_data.second.primitiveCount(ctx.structure_data.type);
    return ok;
}

bool StructurePipeline::runAssociation(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_match_ms);

    // 1. 结构响应图为空时，后续估计没有任何输入依据。
    if (ctx.structure_data.empty()) {
        IR_LOG_ERROR("StructurePipeline::runAssociation: structure data is empty.");
        return false;
    }

    // 2. 当前最小可运行版本直接使用稠密结构响应图做估计，因此这里不再单独构造匹配器。
    ctx.result.num_raw_matches = 0;
    ctx.result.num_filtered_matches = 0;
    return true;
}

bool StructurePipeline::runEstimation(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_geometry_ms);

    // 1. 将两张结构响应图转换为相位相关估计需要的浮点图。
    cv::Mat src;
    cv::Mat dst;
    if (!preparePhaseImage(ctx.structure_data.first.mask, src, _phaseBlurKernel) ||
        !preparePhaseImage(ctx.structure_data.second.mask, dst, _phaseBlurKernel)) {
        ctx.geometry_data.message = "structure masks are empty";
        IR_LOG_ERROR("StructurePipeline::runEstimation: structure masks are empty.");
        return false;
    }

    // 2. 当前平移估计要求两张结构图尺寸一致。
    if (src.size() != dst.size()) {
        ctx.geometry_data.message = "structure masks have different sizes";
        IR_LOG_ERROR("StructurePipeline::runEstimation: structure masks have different sizes.");
        return false;
    }

    // 3. 使用相位相关估计平移量和响应强度。
    double response = 0.0;
    const cv::Point2d shift = cv::phaseCorrelate(src, dst, cv::noArray(), &response);

    // 4. 将平移写成 2x3 仿射矩阵，复用后续通用 warp 流程。
    auto& gd = ctx.geometry_data;
    gd.clear();
    gd.type = GeometryType::AFFINE;
    gd.A = (cv::Mat_<double>(2, 3) << 1.0, 0.0, shift.x, 0.0, 1.0, shift.y);
    gd.num_inliers = ctx.result.num_structures_first;
    gd.inlier_ratio = response;
    gd.valid = response >= _responseThreshold;

    // 5. 同步总结果统计字段，并在响应不足时记录失败原因。
    ctx.result.num_inliers = gd.num_inliers;
    ctx.result.inlier_ratio = gd.inlier_ratio;
    if (!gd.valid) {
        gd.message = "phase correlation response below threshold: " + std::to_string(response);
        IR_LOG_WARN("StructurePipeline rejected transform: ", gd.message);
    }

    IR_LOG_INFO("StructurePipeline estimated translation dx=",
                shift.x,
                ", dy=",
                shift.y,
                ", response=",
                response);
    return gd.valid;
}

std::string StructurePipeline::buildOutputStem(const RegistrationContext& ctx) const {
    // 1. 使用输入图像名作为基础前缀，便于追溯样本来源。
    const std::string sampleStem = ctx.image1_path.stem().string() + "_" +
                                   ctx.image2_path.stem().string();

    // 2. 追加结构类型和当前估计策略，避免不同结构法输出互相覆盖。
    return sampleStem + "_" + (_extractor ? _extractor->name() : std::string("STRUCTURE")) +
           "_TRANSLATION";
}

bool StructurePipeline::saveOutputs(RegistrationContext& ctx) {
    // 1. 没有配置输出目录时直接跳过可视化落盘。
    if (_config.output_dir.empty())
        return true;

    // 2. 准备结构法专属输出目录。
    const fs::path structureDir = _config.output_dir / "structures";
    std::error_code ec;
    fs::create_directories(structureDir, ec);

    // 3. 保存两张图像的结构响应图，便于检查提取质量。
    const std::string stem = buildOutputStem(ctx);
    if (!ctx.structure_data.first.mask.empty()) {
        const fs::path out = structureDir / (stem + "_source_structure.png");
        cv::imwrite(out.string(), ctx.structure_data.first.mask);
        IR_LOG_INFO("Wrote source structure visualization: ", out.string());
    }
    if (!ctx.structure_data.second.mask.empty()) {
        const fs::path out = structureDir / (stem + "_target_structure.png");
        cv::imwrite(out.string(), ctx.structure_data.second.mask);
        IR_LOG_INFO("Wrote target structure visualization: ", out.string());
    }

    // 4. 委托基类保存 warp 和 blend 等通用输出。
    return BasePipeline::saveOutputs(ctx);
}

} // namespace ir
