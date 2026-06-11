#include "registration_app.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "core/config.h"
#include "core/context.h"
#include "dataset/dataset_loader.h"
#include "interfaces/i_pipeline.h"
#include "pipeline/direct_pipeline.h"
#include "pipeline/keypoint_pipeline.h"
#include "pipeline/learning_pipeline.h"
#include "pipeline/structure_pipeline.h"
#include "utils/file_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

namespace summary_text {

// 统一格式化毫秒耗时，保证终端摘要的小数位一致。
std::string formatMilliseconds(double v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << v << " ms";
    return oss.str();
}

// 统一输出各阶段耗时，避免两类方法维护两套终端格式。
void appendTiming(std::ostringstream& oss, const RegistrationResult& r) {
    oss << "  -- timings --\n";
    oss << "  load          : " << formatMilliseconds(r.t_load_ms) << "\n";
    oss << "  extract       : " << formatMilliseconds(r.t_extract_ms) << "\n";
    oss << "  match         : " << formatMilliseconds(r.t_match_ms) << "\n";
    oss << "  filter        : " << formatMilliseconds(r.t_filter_ms) << "\n";
    oss << "  geometry      : " << formatMilliseconds(r.t_geometry_ms) << "\n";
    oss << "  warp          : " << formatMilliseconds(r.t_warp_ms) << "\n";
    oss << "  TOTAL         : " << formatMilliseconds(r.t_total_ms) << "\n";
}

void appendEvaluation(std::ostringstream& oss, const EvaluationData& evaluation) {
    if (evaluation.metrics.empty()) {
        return;
    }

    oss << "  -- metrics --\n";
    for (const auto& metric : evaluation.metrics) {
        oss << "  " << metric.name << " : ";
        if (metric.valid) {
            oss << std::fixed << std::setprecision(6) << metric.value;
        } else {
            oss << "N/A";
        }
        if (!metric.note.empty()) {
            oss << " (" << metric.note << ")";
        }
        oss << "\n";
    }
}

// 直接法诊断项由各算法自行写入，摘要层只负责统一展示，避免为每个新方法增加特判。
void appendDirectDiagnostics(std::ostringstream& oss, const DirectData& direct) {
    if (direct.diagnostics.empty()) {
        return;
    }

    for (const auto& item : direct.diagnostics) {
        if (!std::isfinite(item.value)) {
            continue;
        }
        const std::string label = item.label.empty() ? item.key : item.label;
        oss << "  " << label << " : " << std::fixed << std::setprecision(3)
            << item.value << "\n";
    }
}

} // namespace summary_text

namespace json_output {

std::string escapeString(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out.push_back(c);
            break;
        }
    }
    return out;
}

// 将直接法诊断项写成 JSON 对象；key 保持机器可读，便于后续 FMT 等方法复用同一出口。
void appendDirectDiagnostics(std::ostringstream& oss, const DirectData& direct) {
    oss << "    \"direct_diagnostics\": {";
    bool wroteAny = false;
    for (const auto& item : direct.diagnostics) {
        if (item.key.empty() || !std::isfinite(item.value)) {
            continue;
        }
        oss << (wroteAny ? ",\n" : "\n");
        oss << "      \"" << escapeString(item.key) << "\": " << item.value;
        wroteAny = true;
    }
    if (wroteAny) {
        oss << "\n    }\n";
    } else {
        oss << "}\n";
    }
}

} // namespace json_output

double matAtAsDouble(const cv::Mat& mat, int row, int col) {
    if (mat.empty() || row < 0 || col < 0 || row >= mat.rows || col >= mat.cols) {
        return 0.0;
    }

    cv::Mat value64;
    mat(cv::Rect(col, row, 1, 1)).convertTo(value64, CV_64F);
    return value64.at<double>(0, 0);
}

std::vector<std::string> readPatternCandidates(const YAML::Node& node,
                                               const std::string& listKey,
                                               const std::string& scalarKey,
                                               const std::vector<std::string>& fallback) {
    std::vector<std::string> values = yaml_utils::getVec<std::string>(node, listKey, {});
    if (!values.empty()) {
        return values;
    }

    const std::string scalar = yaml_utils::getString(node, scalarKey);
    if (!scalar.empty()) {
        return {scalar};
    }
    return fallback;
}

void loadDatasetNamingOptions(const YAML::Node& ds, DatasetLoader::Options& options) {
    // 读取数据集命名候选列表；缺省时回退到默认 source/moving 与 target/reference。
    options.pattern_sources =
        readPatternCandidates(ds, "pattern_sources", "pattern_source", {"source", "moving"});
    options.pattern_targets =
        readPatternCandidates(ds, "pattern_targets", "pattern_target", {"target", "reference"});
}

// 点特征法摘要聚焦 keypoint/descriptor 匹配链路。
std::string buildKeypointSummaryText(const RegistrationContext& ctx) {
    const auto& r = ctx.result;
    std::ostringstream oss;
    oss << "\n================ Keypoint registration summary ================\n";
    oss << "  status        : " << (r.success ? "OK" : "FAILED") << "\n";
    oss << "  message       : " << r.message << "\n";
    oss << "  keypoints     : " << r.num_keypoints_first << " / " << r.num_keypoints_second
        << "\n";
    oss << "  raw matches   : " << r.num_raw_matches << "\n";
    oss << "  filtered      : " << r.num_filtered_matches << "\n";
    oss << "  inliers       : " << r.num_inliers << " (" << std::fixed << std::setprecision(3)
        << r.inlier_ratio << ")\n";
    if (r.warp_overlap_iou >= 0.0) {
        oss << "  warp IoU      : " << std::fixed << std::setprecision(3)
            << r.warp_overlap_iou << "\n";
    }
    if (r.warp_photometric_error >= 0.0) {
        oss << "  warp NMAD      : " << std::fixed << std::setprecision(4)
            << r.warp_photometric_error << "\n";
    }
    // 终端摘要统一追加耗时和评价指标，避免各方法族各写一套格式。
    summary_text::appendTiming(oss, r);
    summary_text::appendEvaluation(oss, ctx.evaluation);
    oss << "==============================================================\n";
    return oss.str();
}

// 深度学习匹配法复用点对几何链路，但摘要标题单独标识来源。
std::string buildLearningSummaryText(const RegistrationContext& ctx) {
    const auto& r = ctx.result;
    std::ostringstream oss;
    oss << "\n================ Learning registration summary ================\n";
    oss << "  status        : " << (r.success ? "OK" : "FAILED") << "\n";
    oss << "  message       : " << r.message << "\n";
    oss << "  keypoints     : " << r.num_keypoints_first << " / " << r.num_keypoints_second
        << "\n";
    oss << "  raw matches   : " << r.num_raw_matches << "\n";
    oss << "  filtered      : " << r.num_filtered_matches << "\n";
    oss << "  inliers       : " << r.num_inliers << " (" << std::fixed << std::setprecision(3)
        << r.inlier_ratio << ")\n";
    if (r.warp_overlap_iou >= 0.0) {
        oss << "  warp IoU      : " << std::fixed << std::setprecision(3)
            << r.warp_overlap_iou << "\n";
    }
    if (r.warp_photometric_error >= 0.0) {
        oss << "  warp NMAD      : " << std::fixed << std::setprecision(4)
            << r.warp_photometric_error << "\n";
    }
    summary_text::appendTiming(oss, r);
    summary_text::appendEvaluation(oss, ctx.evaluation);
    oss << "==============================================================\n";
    return oss.str();
}

// 结构法摘要聚焦边缘/直线/轮廓数量，以及当前结构配准的估计结果。
std::string buildStructureSummaryText(const RegistrationContext& ctx) {
    const auto& r = ctx.result;
    const auto& gd = ctx.geometry_data;
    const auto& smd = ctx.structure_match_data;
    std::ostringstream oss;
    oss << "\n================ Structure registration summary ================\n";
    oss << "  status        : " << (r.success ? "OK" : "FAILED") << "\n";
    oss << "  message       : " << r.message << "\n";
    oss << "  structure type: " << toString(ctx.structure_data.type) << "\n";
    oss << "  structures    : " << r.num_structures_first << " / " << r.num_structures_second
        << "\n";
    if (gd.valid && !gd.A.empty() && gd.A.rows >= 2 && gd.A.cols >= 3) {
        oss << "  translation   : dx=" << std::fixed << std::setprecision(3)
            << gd.A.at<double>(0, 2) << ", dy=" << gd.A.at<double>(1, 2) << "\n";
    } else if (!smd.affine.empty() && smd.affine.rows >= 2 && smd.affine.cols >= 3) {
        oss << "  translation   : dx=" << std::fixed << std::setprecision(3)
            << smd.affine.at<double>(0, 2) << ", dy=" << smd.affine.at<double>(1, 2) << "\n";
    } else {
        oss << "  translation   : dx=" << std::fixed << std::setprecision(3)
            << smd.translation.x << ", dy=" << smd.translation.y << "\n";
    }
    oss << "  response      : " << std::fixed << std::setprecision(3) << r.inlier_ratio << "\n";
    if (r.warp_overlap_iou >= 0.0) {
        oss << "  warp IoU      : " << std::fixed << std::setprecision(3)
            << r.warp_overlap_iou << "\n";
    }
    if (r.warp_photometric_error >= 0.0) {
        oss << "  warp NMAD      : " << std::fixed << std::setprecision(4)
            << r.warp_photometric_error << "\n";
    }
    // 终端摘要统一追加耗时和评价指标，避免各方法族各写一套格式。
    summary_text::appendTiming(oss, r);
    summary_text::appendEvaluation(oss, ctx.evaluation);
    oss << "================================================================\n";
    return oss.str();
}

// 直接法摘要聚焦算法、全局几何结果和光度质量，不再套用点特征标题。
std::string buildDirectSummaryText(const RegistrationContext& ctx) {
    const auto& r = ctx.result;
    const auto& gd = ctx.geometry_data;
    const auto& dd = ctx.direct_data;
    std::ostringstream oss;
    oss << "\n================ Direct registration summary ================\n";
    oss << "  status        : " << (r.success ? "OK" : "FAILED") << "\n";
    oss << "  message       : " << r.message << "\n";
    oss << "  method        : " << (dd.method.empty() ? "DIRECT" : dd.method) << "\n";
    oss << "  geometry      : " << toString(gd.type) << "\n";

    if (gd.valid && !gd.A.empty() && gd.A.rows >= 2 && gd.A.cols >= 3) {
        const double a00 = matAtAsDouble(gd.A, 0, 0);
        const double a10 = matAtAsDouble(gd.A, 1, 0);
        const double tx = matAtAsDouble(gd.A, 0, 2);
        const double ty = matAtAsDouble(gd.A, 1, 2);
        const double rotationDeg = std::atan2(a10, a00) * 180.0 / 3.14159265358979323846;
        oss << "  translation   : dx=" << std::fixed << std::setprecision(3) << tx
            << ", dy=" << ty << "\n";
        if (gd.type == GeometryType::RIGID || gd.type == GeometryType::SIMILARITY ||
            gd.type == GeometryType::AFFINE) {
            oss << "  rotation      : " << std::fixed << std::setprecision(3)
                << rotationDeg << " deg\n";
        }
    }

    oss << "  correspondences: " << r.num_raw_matches << "\n";
    oss << "  inliers       : " << r.num_inliers << " (" << std::fixed << std::setprecision(3)
        << r.inlier_ratio << ")\n";
    summary_text::appendDirectDiagnostics(oss, dd);
    if (dd.photometric_error >= 0.0) {
        oss << "  photometric MSE: " << std::fixed << std::setprecision(6)
            << dd.photometric_error << "\n";
    }
    if (r.warp_overlap_iou >= 0.0) {
        oss << "  warp IoU      : " << std::fixed << std::setprecision(3)
            << r.warp_overlap_iou << "\n";
    }
    if (r.warp_photometric_error >= 0.0) {
        oss << "  warp NMAD      : " << std::fixed << std::setprecision(4)
            << r.warp_photometric_error << "\n";
    }
    // 终端摘要统一追加耗时和评价指标，避免各方法族各写一套格式。
    summary_text::appendTiming(oss, r);
    summary_text::appendEvaluation(oss, ctx.evaluation);
    oss << "=============================================================\n";
    return oss.str();
}

// 根据方法族选择摘要格式。
std::string buildSummaryText(const RegistrationContext& ctx, MethodFamily family) {
    if (family == MethodFamily::STRUCTURE)
        return buildStructureSummaryText(ctx);
    if (family == MethodFamily::DIRECT)
        return buildDirectSummaryText(ctx);
    if (family == MethodFamily::LEARNING)
        return buildLearningSummaryText(ctx);
    return buildKeypointSummaryText(ctx);
}

void printSummary(const RegistrationContext& ctx, MethodFamily family) {
    std::cout << buildSummaryText(ctx, family);
}

// 根据 method_family 显式字段选择 pipeline 类型，直接法后续在此扩展。
std::shared_ptr<IPipeline> createPipelineForConfig(const PipelineConfig& cfg) {
    switch (cfg.method_family) {
    case MethodFamily::STRUCTURE:
        return std::make_shared<StructurePipeline>();
    case MethodFamily::DIRECT:
        return std::make_shared<DirectPipeline>();
    case MethodFamily::LEARNING:
        return std::make_shared<LearningPipeline>();
    case MethodFamily::KEYPOINT:
    default:
        return std::make_shared<KeypointPipeline>();
    }
}

std::string sampleStemFromPaths(const fs::path& image1, const fs::path& image2) {
    return image1.stem().string() + "__" + image2.stem().string();
}

fs::path normalizeOutputBaseRoot(fs::path root) {
    if (root.empty()) {
        return root;
    }

    const std::string leaf = root.filename().string();
    if (leaf == "single" || leaf == "batch") {
        root = root.parent_path();
    }
    return root;
}

std::string buildSummaryJson(const RegistrationContext& ctx,
                             const PipelineConfig& cfg,
                             const std::string& sample_name) {
    const auto& r = ctx.result;
    const auto family = cfg.methodFamily();
    const bool isStructure = (family == MethodFamily::STRUCTURE);
    const bool isDirect = (family == MethodFamily::DIRECT);
    std::ostringstream oss;
    oss << "{\n";
    // JSON 字符串统一经过本文件私有转义，避免路径/消息中的特殊字符破坏语法。
    oss << "  \"pipeline_name\": \"" << json_output::escapeString(cfg.name) << "\",\n";
    oss << "  \"method_family\": \"" << methodFamilyDir(family) << "\",\n";
    oss << "  \"sample_name\": \"" << json_output::escapeString(sample_name) << "\",\n";
    oss << "  \"status\": \"" << (r.success ? "OK" : "FAILED") << "\",\n";
    oss << "  \"message\": \"" << json_output::escapeString(r.message) << "\",\n";
    oss << "  \"image1_path\": \"" << json_output::escapeString(ctx.image1_path.string()) << "\",\n";
    oss << "  \"image2_path\": \"" << json_output::escapeString(ctx.image2_path.string()) << "\",\n";
    oss << "  \"counts\": {\n";
    if (isStructure) {
        oss << "    \"num_structures_first\": " << r.num_structures_first << ",\n";
        oss << "    \"num_structures_second\": " << r.num_structures_second << ",\n";
    } else {
        oss << "    \"num_keypoints_first\": " << r.num_keypoints_first << ",\n";
        oss << "    \"num_keypoints_second\": " << r.num_keypoints_second << ",\n";
    }
    oss << "    \"num_raw_matches\": " << r.num_raw_matches << ",\n";
    oss << "    \"num_filtered_matches\": " << r.num_filtered_matches << ",\n";
    oss << "    \"num_inliers\": " << r.num_inliers << "\n";
    oss << "  },\n";
    oss << "  \"quality\": {\n";
    oss << "    \"inlier_ratio\": " << r.inlier_ratio << ",\n";
    oss << "    \"mean_reproj_error\": " << r.mean_reproj_error << ",\n";
    oss << "    \"warp_overlap_iou\": " << r.warp_overlap_iou;
    if (isDirect) {
        oss << ",\n";
        json_output::appendDirectDiagnostics(oss, ctx.direct_data);
    } else {
        oss << "\n";
    }
    oss << "  },\n";
    if (isStructure && ctx.geometry_data.valid &&
        !ctx.geometry_data.A.empty() && ctx.geometry_data.A.rows >= 2 &&
        ctx.geometry_data.A.cols >= 3) {
        oss << "  \"translation\": {\n";
        oss << "    \"dx\": " << ctx.geometry_data.A.at<double>(0, 2) << ",\n";
        oss << "    \"dy\": " << ctx.geometry_data.A.at<double>(1, 2) << "\n";
        oss << "  },\n";
    } else if (isStructure) {
        oss << "  \"translation\": {\n";
        oss << "    \"dx\": " << ctx.structure_match_data.translation.x << ",\n";
        oss << "    \"dy\": " << ctx.structure_match_data.translation.y << "\n";
        oss << "  },\n";
    }
    oss << "  \"timings_ms\": {\n";
    oss << "    \"load\": " << r.t_load_ms << ",\n";
    oss << "    \"extract\": " << r.t_extract_ms << ",\n";
    oss << "    \"match\": " << r.t_match_ms << ",\n";
    oss << "    \"filter\": " << r.t_filter_ms << ",\n";
    oss << "    \"geometry\": " << r.t_geometry_ms << ",\n";
    oss << "    \"warp\": " << r.t_warp_ms << ",\n";
    oss << "    \"total\": " << r.t_total_ms << "\n";
    oss << "  },\n";
    oss << "  \"metrics\": {\n";
    for (size_t i = 0; i < ctx.evaluation.metrics.size(); ++i) {
        const auto& metric = ctx.evaluation.metrics[i];
        oss << "    \"" << json_output::escapeString(metric.name) << "\": ";
        if (metric.valid) {
            oss << metric.value;
        } else {
            oss << "null";
        }
        if (i + 1 < ctx.evaluation.metrics.size()) {
            oss << ",";
        }
        oss << "\n";
    }
    oss << "  }\n";
    oss << "}\n";
    return oss.str();
}

void writeRunSummaryFiles(const RegistrationContext& ctx,
                          const PipelineConfig& cfg,
                          const std::string& sample_name) {
    if (ctx.output_dir.empty()) {
        return;
    }

    const auto family = cfg.methodFamily();
    const std::string summary_text = buildSummaryText(ctx, family);
    const std::string summary_json = buildSummaryJson(ctx, cfg, sample_name);

    const fs::path txt_path = ctx.output_dir / "run_summary.txt";
    const fs::path json_path = ctx.output_dir / "run_summary.json";
    if (file_utils::writeWholeFile(txt_path, summary_text)) {
        IR_LOG_INFO("Wrote run summary text: ", txt_path.string());
    }
    if (file_utils::writeWholeFile(json_path, summary_json)) {
        IR_LOG_INFO("Wrote run summary json: ", json_path.string());
    }
}

std::vector<std::string> collectMetricColumns(const std::vector<EvaluationData>& evaluations) {
    std::vector<std::string> columns;
    for (const auto& evaluation : evaluations) {
        for (const auto& metric : evaluation.metrics) {
            if (std::find(columns.begin(), columns.end(), metric.name) == columns.end()) {
                columns.push_back(metric.name);
            }
        }
    }
    return columns;
}

} // namespace

void RegistrationApp::printUsage(const std::string& exe) {
    std::cout << "Usage:\n"
              << "  " << exe << " <pipeline.yaml> [image1] [image2] [output_dir]\n\n"
              << "  " << exe << " <batch.yaml>\n\n"
              << "Examples:\n"
              << "  " << exe << " configs/pipeline/keypoint/sift_pipeline.yaml\n"
              << "  " << exe << " configs/pipeline/batch/batch.yaml\n"
              << "  " << exe
              << " configs/pipeline/keypoint/orb_pipeline.yaml a.jpg b.jpg outputs\n"
              << std::endl;
}

int RegistrationApp::runSingle(const Args& args) {
    // 1. 先校验 pipeline YAML 路径，避免后续阶段在缺少配置时继续下沉。
    if (args.pipeline_yaml.empty()) {
        std::cerr << "Pipeline YAML path is empty.\n";
        return 2;
    }
    if (!fs::exists(args.pipeline_yaml)) {
        std::cerr << "Pipeline YAML not found: " << args.pipeline_yaml.string() << "\n";
        return 2;
    }

    // 2. 加载 pipeline 配置；具体算法组件都由子 YAML 决定。
    PipelineConfig cfg;
    try {
        cfg = Config::loadPipeline(args.pipeline_yaml);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load pipeline YAML: " << e.what() << "\n";
        return 3;
    }

    // 3. 命令行显式传入的图像和输出目录优先级高于 YAML。
    if (!args.image1.empty())
        cfg.image1_path = fs::weakly_canonical(args.image1);
    if (!args.image2.empty())
        cfg.image2_path = fs::weakly_canonical(args.image2);

    // 4. 单次运行必须具备一对输入图像。
    if (cfg.image1_path.empty() || cfg.image2_path.empty()) {
        std::cerr << "Missing image1 / image2. Provide them either in the YAML "
                     "(io.image1 / io.image2) or as positional arguments.\n";
        return 4;
    }
    if (!fs::exists(cfg.image1_path)) {
        std::cerr << "image1 not found: " << cfg.image1_path.string() << "\n";
        return 5;
    }
    if (!fs::exists(cfg.image2_path)) {
        std::cerr << "image2 not found: " << cfg.image2_path.string() << "\n";
        return 5;
    }

    // 5. 最终输出目录统一由应用层生成，避免单次/批处理/不同 pipeline 各自长一套。
    const fs::path single_root =
        normalizeOutputBaseRoot(!args.output_dir.empty() ? fs::weakly_canonical(args.output_dir)
                                                         : cfg.output_dir);
    const std::string sample_name = sampleStemFromPaths(cfg.image1_path, cfg.image2_path);
    cfg.output_dir = buildOutputDir(OutputMode::SINGLE, single_root, cfg, sample_name);

    if (!cfg.output_dir.empty()) {
        std::error_code ec;
        fs::create_directories(cfg.output_dir, ec);
    }

    // 6. 根据配置选择 keypoint 流水线或结构流水线。
    auto pipeline = createPipelineForConfig(cfg);
    if (!pipeline->configure(cfg)) {
        std::cerr << "Pipeline configure failed.\n";
        return 6;
    }

    // 7. 上下文承担阶段间数据交换职责，也是最终统计结果的统一出口。
    RegistrationContext ctx;

    // 8. 单次运行沿用统一流水线编排，具体算法由配置决定。
    const bool ok = pipeline->run(ctx);
    writeRunSummaryFiles(ctx, cfg, sample_name);
    writeSummaryCsv(cfg.output_dir / "summary.csv",
                    cfg.methodFamily(),
                    std::vector<std::string>{sample_name},
                    std::vector<RegistrationResult>{ctx.result},
                    std::vector<EvaluationData>{ctx.evaluation});
    pipeline->showWindows(ctx);

    // 9. 根据方法族输出对应摘要。
    printSummary(ctx, cfg.methodFamily());
    return ok ? 0 : 1;
}

bool RegistrationApp::isBatchYaml(const YAML::Node& node) {
    return node && node.IsMap() && node["pipeline"] && node["dataset"];
}

bool RegistrationApp::isCompareYaml(const YAML::Node& node) {
    return node && node.IsMap() && node["base_pipeline"] && node["combinations"];
}

int RegistrationApp::runCompare(const std::filesystem::path& compare_yaml) {
    const YAML::Node cfg = Config::load(compare_yaml);
    const auto baseDir = compare_yaml.parent_path();

    // 1. 解析基础 pipeline。compare_line 复用它跑结构法；compare_direct 可在组合项中覆盖。
    const auto pipeline_yaml = Config::resolvePath(baseDir,
        yaml_utils::getString(cfg, "base_pipeline"));

    // 2. 数据集
    DatasetLoader::Options datasetOpts;
    const YAML::Node ds = cfg["dataset"];
    datasetOpts.root = Config::resolvePath(baseDir, yaml_utils::getString(ds, "root"));
    loadDatasetNamingOptions(ds, datasetOpts);
    datasetOpts.include = yaml_utils::getVec<std::string>(ds, "include", {});

    // 3. 输出根目录
    const auto outputRoot = Config::resolvePath(baseDir,
        yaml_utils::getString(cfg["output"], "root", "../../outputs/compare"));
    std::error_code ec;
    fs::create_directories(outputRoot / "tmp", ec);

    // 4. 遍历组合
    const YAML::Node combos = cfg["combinations"];
    std::vector<std::string> labels;
    std::vector<int> successCounts;
    std::vector<int> totalCounts;
    std::vector<double> avgIous;
    std::vector<double> avgPsnrs;

    auto runCombination = [&](const std::string& label, const PipelineConfig& pipelineCfg) {
        DatasetLoader loader(datasetOpts);
        const auto samples = loader.load();
        if (samples.empty()) {
            std::cerr << "Compare: no samples for " << label << "\n";
            return;
        }

        int okCount = 0;
        double iouSum = 0.0, psnrSum = 0.0;
        int iouValid = 0, psnrValid = 0;

        for (const auto& sample : samples) {
            auto pCfg = pipelineCfg;
            pCfg.image1_path = sample.source_path;
            pCfg.image2_path = sample.target_path;
            pCfg.output_dir = pCfg.output_dir / sample.name;

            auto pipeline = createPipelineForConfig(pCfg);
            if (!pipeline->configure(pCfg)) continue;

            RegistrationContext ctx;
            if (pipeline->run(ctx)) ++okCount;

            if (ctx.result.warp_overlap_iou >= 0.0) {
                iouSum += ctx.result.warp_overlap_iou;
                ++iouValid;
            }
            // 从 evaluation 中取 PSNR。
            if (const auto* m = ctx.evaluation.find("PSNR"); m && m->valid) {
                psnrSum += m->value;
                ++psnrValid;
            }
        }

        labels.push_back(label);
        successCounts.push_back(okCount);
        totalCounts.push_back(static_cast<int>(samples.size()));
        avgIous.push_back(iouValid > 0 ? iouSum / iouValid : -1.0);
        avgPsnrs.push_back(psnrValid > 0 ? psnrSum / psnrValid : -1.0);

        std::cout << "  " << label << ": " << okCount << " / " << samples.size()
                  << " succeeded\n";
    };

    bool compareDirect = static_cast<bool>(cfg["base_direct"]);
    for (const auto& combo : combos) {
        if (combo["direct"] || combo["pipeline"]) {
            compareDirect = true;
            break;
        }
    }

    if (compareDirect) {
        const std::string baseDirect = yaml_utils::getString(cfg, "base_direct");

        for (const auto& combo : combos) {
            const std::string pipelineEntry = yaml_utils::getString(combo, "pipeline");
            const auto comboPipelineYaml = pipelineEntry.empty()
                ? pipeline_yaml
                : Config::resolvePath(baseDir, pipelineEntry);

            const std::string directEntry = yaml_utils::getString(combo, "direct", baseDirect);
            const auto directYaml = directEntry.empty()
                ? std::filesystem::path{}
                : Config::resolvePath(baseDir, directEntry);

            PipelineConfig pipelineCfg = Config::loadPipeline(comboPipelineYaml);
            if (!directYaml.empty()) {
                pipelineCfg.direct_path = directYaml;
            }
            if (pipelineCfg.methodFamily() != MethodFamily::DIRECT) {
                std::cerr << "Compare: skip non-direct pipeline "
                          << comboPipelineYaml.string() << "\n";
                continue;
            }

            const std::string directStem = !pipelineCfg.direct_path.empty()
                ? pipelineCfg.direct_path.stem().string()
                : std::string{"configured_direct"};
            std::string label = yaml_utils::getString(combo, "label");
            if (label.empty()) {
                label = comboPipelineYaml.stem().string() + "+" + directStem;
            }

            pipelineCfg.name = label;
            pipelineCfg.output_dir = outputRoot / label;
            pipelineCfg.draw_matches = false;
            pipelineCfg.warp = true;
            pipelineCfg.show_source_window = false;
            pipelineCfg.show_target_window = false;
            pipelineCfg.show_warped_window = false;

            runCombination(label, pipelineCfg);
        }
    } else {
        const auto structure_yaml = Config::resolvePath(baseDir,
            yaml_utils::getString(cfg, "base_structure"));
        YAML::Node structureNode = Config::load(structure_yaml);

        for (const auto& combo : combos) {
            const std::string extractor = yaml_utils::getString(combo, "extractor", "LSD");
            const std::string descriptor = yaml_utils::getString(combo, "descriptor", "LBD");
            const std::string label = extractor + "+" + descriptor;

            // 修改结构 YAML 节点
            structureNode["extractor"]["method"] = extractor;
            structureNode["association"]["params"]["line_descriptor"]["descriptor"] = descriptor;
            structureNode["association"]["params"]["line_descriptor"]["geometric_filter"] = true;

            // 写临时文件
            const auto tmpYaml = outputRoot / "tmp" / (label + ".yaml");
            std::ofstream fout(tmpYaml.string());
            fout << structureNode;
            fout.close();

            // 加载 pipeline 配置并执行批量
            PipelineConfig pipelineCfg = Config::loadPipeline(pipeline_yaml);
            pipelineCfg.structure_path = tmpYaml;
            pipelineCfg.output_dir = outputRoot / label;
            pipelineCfg.draw_matches = false;
            pipelineCfg.warp = true;
            pipelineCfg.show_source_window = false;
            pipelineCfg.show_target_window = false;
            pipelineCfg.show_warped_window = false;

            runCombination(label, pipelineCfg);
        }
    }

    // 5. 写对比总表
    const auto csvPath = outputRoot / "comparison.csv";
    std::ostringstream oss;
    oss << "method,succeeded,total,success_rate,avg_iou,avg_psnr\n";
    for (size_t i = 0; i < labels.size(); ++i) {
        oss << labels[i] << ","
            << successCounts[i] << ","
            << totalCounts[i] << ","
            << (totalCounts[i] > 0
                    ? static_cast<double>(successCounts[i]) / totalCounts[i]
                    : 0.0) << ","
            << avgIous[i] << ","
            << avgPsnrs[i] << "\n";
    }
    file_utils::writeWholeFile(csvPath, oss.str());
    std::cout << "\nWrote comparison table: " << csvPath.string() << "\n";
    return 0;
}

RegistrationApp::BatchConfig RegistrationApp::loadBatchConfig(const std::filesystem::path& yaml_path) {
    // 批处理配置以 batch.yaml 所在目录为基准解析相对路径，便于配置整体迁移。
    const YAML::Node node = Config::load(yaml_path);
    BatchConfig cfg;
    cfg.name = yaml_utils::getString(node, "name", yaml_path.stem().string());

    const auto batch_dir = yaml_path.parent_path();
    cfg.pipeline_yaml = Config::resolvePath(batch_dir, yaml_utils::getString(node, "pipeline"));

    const YAML::Node dataset = node["dataset"];
    cfg.dataset.root = Config::resolvePath(batch_dir, yaml_utils::getString(dataset, "root"));
    loadDatasetNamingOptions(dataset, cfg.dataset);
    cfg.dataset.include = yaml_utils::getVec<std::string>(dataset, "include", {});

    const YAML::Node output = node["output"];
    cfg.output_root = Config::resolvePath(
        batch_dir, yaml_utils::getString(output, "root", "../../outputs"));
    cfg.save_visuals = yaml_utils::getBool(output, "save_visuals", true);
    cfg.summary_csv = yaml_utils::getBool(output, "summary_csv", true);
    return cfg;
}

std::filesystem::path RegistrationApp::resolveBatchOutputRoot(const BatchConfig& batch,
                                                              const PipelineConfig& pipeline_cfg) {
    if (batch.output_root.empty())
        return {};

    // 批处理输出根目录只保留真正的根路径，具体层级由应用层统一补齐。
    return normalizeOutputBaseRoot(batch.output_root);
}

std::filesystem::path RegistrationApp::buildOutputDir(OutputMode mode,
                                                      const std::filesystem::path& base_root,
                                                      const PipelineConfig& cfg,
                                                      const std::string& sample_name) {
    if (base_root.empty()) {
        return {};
    }

    const std::string mode_dir = mode == OutputMode::BATCH ? "batch" : "single";
    return base_root / mode_dir / methodFamilyDir(cfg.methodFamily()) / cfg.name / sample_name;
}

void RegistrationApp::writeSummaryCsv(const std::filesystem::path& csv_path,
                                      MethodFamily family,
                                      const std::vector<std::string>& sample_names,
                                      const std::vector<RegistrationResult>& results,
                                      const std::vector<EvaluationData>& evaluations) {
    // 汇总表保留两类方法的核心数量字段、耗时和评测指标，便于后续横向比较。
    const std::vector<std::string> metric_columns = collectMetricColumns(evaluations);
    std::ostringstream oss;
    oss << "sample_name,success,message,";
    if (family == MethodFamily::STRUCTURE) {
        oss << "num_structures_first,num_structures_second,";
    } else {
        oss << "num_keypoints_first,num_keypoints_second,";
    }
    oss << "num_raw_matches,num_filtered_matches,num_inliers,"
        << "inlier_ratio,mean_reproj_error,warp_overlap_iou,warp_photometric_error,"
        << "t_load_ms,t_extract_ms,t_match_ms,t_filter_ms,t_geometry_ms,t_warp_ms,t_total_ms";
    for (const auto& metric_name : metric_columns) {
        oss << "," << metric_name;
    }
    oss << "\n";

    for (size_t i = 0; i < sample_names.size() && i < results.size(); ++i) {
        const auto& r = results[i];
        oss << file_utils::csvEscape(sample_names[i]) << "," << (r.success ? "1" : "0") << ","
            << file_utils::csvEscape(r.message) << ",";
        if (family == MethodFamily::STRUCTURE) {
            oss << r.num_structures_first << "," << r.num_structures_second << ",";
        } else {
            oss << r.num_keypoints_first << "," << r.num_keypoints_second << ",";
        }
        oss << r.num_raw_matches << "," << r.num_filtered_matches << "," << r.num_inliers << ","
            << r.inlier_ratio << "," << r.mean_reproj_error << "," << r.warp_overlap_iou << ","
            << r.warp_photometric_error << ","
            << r.t_load_ms << "," << r.t_extract_ms << "," << r.t_match_ms << ","
            << r.t_filter_ms << "," << r.t_geometry_ms << "," << r.t_warp_ms << ","
            << r.t_total_ms;

        if (i < evaluations.size()) {
            for (const auto& metric_name : metric_columns) {
                const MetricResult* metric = evaluations[i].find(metric_name);
                oss << ",";
                if (metric && metric->valid) {
                    oss << metric->value;
                }
            }
        } else {
            for (size_t j = 0; j < metric_columns.size(); ++j) {
                oss << ",";
            }
        }
        oss << "\n";
    }

    file_utils::writeWholeFile(csv_path, oss.str());
}

int RegistrationApp::runBatch(const std::filesystem::path& batch_yaml) {
    // 1. 先加载批处理配置，再派生单样本运行时的基础 pipeline 配置。
    const BatchConfig batch = loadBatchConfig(batch_yaml);
    const PipelineConfig base_cfg = Config::loadPipeline(batch.pipeline_yaml);
    const std::filesystem::path output_root = resolveBatchOutputRoot(batch, base_cfg);
    const std::filesystem::path pipeline_root =
        output_root / "batch" / methodFamilyDir(base_cfg.methodFamily()) / base_cfg.name;

    // 2. 数据集扫描与输出根目录准备是批处理前置条件。
    DatasetLoader loader(batch.dataset);
    const std::vector<Sample> samples = loader.load();
    if (samples.empty()) {
        std::cerr << "No dataset samples found for batch config: " << batch_yaml.string() << "\n";
        return 7;
    }

    std::error_code ec;
    std::filesystem::create_directories(pipeline_root, ec);

    std::vector<std::string> sample_names;
    std::vector<std::string> succeeded_names;
    std::vector<RegistrationResult> results;
    std::vector<EvaluationData> evaluations;
    sample_names.reserve(samples.size());
    succeeded_names.reserve(samples.size());
    results.reserve(samples.size());
    evaluations.reserve(samples.size());

    int ok_count = 0;
    for (const auto& sample : samples) {
        // 3. 每个样本复用同一算法配置，只替换输入图像与输出目录。
        PipelineConfig cfg = base_cfg;
        cfg.image1_path = sample.source_path;
        cfg.image2_path = sample.target_path;
        cfg.output_dir = buildOutputDir(OutputMode::BATCH, output_root, cfg, sample.name);
        if (!batch.save_visuals) {
            cfg.draw_matches = false;
            cfg.warp = false;
        }
        cfg.show_source_window = false;
        cfg.show_target_window = false;
        cfg.show_warped_window = false;

        auto pipeline = createPipelineForConfig(cfg);
        if (!pipeline->configure(cfg)) {
            // 配置失败单独记为该样本失败，避免影响整批任务继续执行。
            RegistrationResult failed;
            failed.success = false;
            failed.message = "pipeline configure failed";
            sample_names.push_back(sample.name);
            results.push_back(failed);
            evaluations.push_back(EvaluationData{});
            continue;
        }

        RegistrationContext ctx;
        const bool ok = pipeline->run(ctx);
        writeRunSummaryFiles(ctx, cfg, sample.name);
        printSummary(ctx, cfg.methodFamily());
        sample_names.push_back(sample.name);
        results.push_back(ctx.result);
        evaluations.push_back(ctx.evaluation);
        if (ok) {
            ++ok_count;
            succeeded_names.push_back(sample.name);
        }
    }

    // 4. 批处理汇总结果写到方法族/pipeline 这一层，便于直接比较同一算法的所有样本。
    if (batch.summary_csv) {
        const auto csv_path = pipeline_root / "summary.csv";
        writeSummaryCsv(csv_path, base_cfg.methodFamily(), sample_names, results, evaluations);
        std::cout << "Wrote summary CSV: " << csv_path.string() << "\n";
    }

    std::cout << "\nBatch summary: " << ok_count << " / " << samples.size()
              << " samples succeeded.\n";
    if (!succeeded_names.empty()) {
        std::cout << "Successful samples: ";
        for (size_t i = 0; i < succeeded_names.size(); ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << succeeded_names[i];
        }
        std::cout << "\n";
    }
    return ok_count == static_cast<int>(samples.size()) ? 0 : 1;
}

int RegistrationApp::run(const Args& args) {
    // 入口先判断 YAML 类型，再路由到单样本或批处理分支。
    if (args.pipeline_yaml.empty()) {
        std::cerr << "Pipeline YAML path is empty.\n";
        return 2;
    }
    if (!fs::exists(args.pipeline_yaml)) {
        std::cerr << "YAML not found: " << args.pipeline_yaml.string() << "\n";
        return 2;
    }

    try {
        const YAML::Node node = Config::load(args.pipeline_yaml);
        if (isCompareYaml(node)) {
            return RegistrationApp::runCompare(args.pipeline_yaml);
        }
        if (isBatchYaml(node)) {
            return RegistrationApp::runBatch(args.pipeline_yaml);
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to read YAML: " << e.what() << "\n";
        return 3;
    }

    return runSingle(args);
}

int RegistrationApp::run(int argc, char** argv) {
    // 位置参数保持最小约定，避免命令行层承担复杂解析逻辑。
    if (argc < 2) {
        printUsage(argc > 0 ? argv[0] : "registration_app");
        return 1;
    }

    Args args;
    args.pipeline_yaml = argv[1];
    if (argc >= 3)
        args.image1 = argv[2];
    if (argc >= 4)
        args.image2 = argv[3];
    if (argc >= 5)
        args.output_dir = argv[4];

    return run(args);
}

} // namespace ir
