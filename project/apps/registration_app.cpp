#include "registration_app.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "core/config.h"
#include "core/context.h"
#include "dataset/dataset_loader.h"
#include "interfaces/i_pipeline.h"
#include "pipeline/feature_pipeline.h"
#include "pipeline/structure_pipeline.h"
#include "utils/file_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

// 统一格式化毫秒耗时，保证终端摘要的小数位一致。
std::string fmtMs(double v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << v << " ms";
    return oss.str();
}

// 统一输出各阶段耗时，避免点特征法和结构法重复维护同一段格式。
void printTimingSummary(const RegistrationResult& r) {
    std::cout << "  -- timings --\n";
    std::cout << "  load          : " << fmtMs(r.t_load_ms) << "\n";
    std::cout << "  extract       : " << fmtMs(r.t_extract_ms) << "\n";
    std::cout << "  match         : " << fmtMs(r.t_match_ms) << "\n";
    std::cout << "  filter        : " << fmtMs(r.t_filter_ms) << "\n";
    std::cout << "  geometry      : " << fmtMs(r.t_geometry_ms) << "\n";
    std::cout << "  warp          : " << fmtMs(r.t_warp_ms) << "\n";
    std::cout << "  TOTAL         : " << fmtMs(r.t_total_ms) << "\n";
}

// 点特征法摘要聚焦 keypoint/descriptor 匹配链路。
void printFeatureSummary(const RegistrationContext& ctx) {
    const auto& r = ctx.result;
    std::cout << "\n================ Feature registration summary ================\n";
    std::cout << "  status        : " << (r.success ? "OK" : "FAILED") << "\n";
    std::cout << "  message       : " << r.message << "\n";
    std::cout << "  keypoints     : " << r.num_keypoints_first << " / " << r.num_keypoints_second
              << "\n";
    std::cout << "  raw matches   : " << r.num_raw_matches << "\n";
    std::cout << "  filtered      : " << r.num_filtered_matches << "\n";
    std::cout << "  inliers       : " << r.num_inliers << " (" << std::fixed
              << std::setprecision(3) << r.inlier_ratio << ")\n";
    printTimingSummary(r);
    std::cout << "=============================================================\n";
}

// 结构法摘要聚焦边缘/直线/轮廓数量，以及当前结构配准的估计结果。
void printStructureSummary(const RegistrationContext& ctx) {
    const auto& r = ctx.result;
    const auto& gd = ctx.geometry_data;
    std::cout << "\n================ Structure registration summary ================\n";
    std::cout << "  status        : " << (r.success ? "OK" : "FAILED") << "\n";
    std::cout << "  message       : " << r.message << "\n";
    std::cout << "  structure type: " << toString(ctx.structure_data.type) << "\n";
    std::cout << "  structures    : " << r.num_structures_first << " / "
              << r.num_structures_second << "\n";
    if (!gd.A.empty() && gd.A.rows >= 2 && gd.A.cols >= 3) {
        std::cout << "  translation   : dx=" << std::fixed << std::setprecision(3)
                  << gd.A.at<double>(0, 2) << ", dy=" << gd.A.at<double>(1, 2) << "\n";
    }
    std::cout << "  response      : " << std::fixed << std::setprecision(3) << r.inlier_ratio
              << "\n";
    printTimingSummary(r);
    std::cout << "================================================================\n";
}

// 根据 pipeline 类型选择摘要格式，避免两类方法的统计字段混在一起。
void printSummary(const RegistrationContext& ctx, bool structurePipeline) {
    if (structurePipeline) {
        printStructureSummary(ctx);
        return;
    }
    printFeatureSummary(ctx);
}

// 当前用是否配置 structure 子项来区分点特征法和结构法。
std::shared_ptr<IPipeline> createPipelineForConfig(const PipelineConfig& cfg) {
    if (!cfg.structure_path.empty()) {
        return std::make_shared<StructurePipeline>();
    }
    return std::make_shared<FeaturePipeline>();
}

} // namespace

void RegistrationApp::printUsage(const std::string& exe) {
    std::cout << "Usage:\n"
              << "  " << exe << " <pipeline.yaml> [image1] [image2] [output_dir]\n\n"
              << "  " << exe << " <batch.yaml>\n\n"
              << "Examples:\n"
              << "  " << exe << " configs/pipeline/sift_pipeline.yaml\n"
              << "  " << exe << " configs/pipeline/batch_pipeline.yaml\n"
              << "  " << exe << " configs/pipeline/orb_pipeline.yaml a.jpg b.jpg outputs\n"
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
    if (!args.output_dir.empty())
        cfg.output_dir = fs::weakly_canonical(args.output_dir);

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

    // 5. 输出目录提前创建，避免流程成功后因为落盘失败而丢失诊断图。
    if (!cfg.output_dir.empty()) {
        std::error_code ec;
        fs::create_directories(cfg.output_dir, ec);
    }

    // 6. 根据配置选择点特征流水线或结构特征流水线。
    auto pipeline = createPipelineForConfig(cfg);
    if (!pipeline->configure(cfg)) {
        std::cerr << "Pipeline configure failed.\n";
        return 6;
    }

    // 7. 上下文承担阶段间数据交换职责，也是最终统计结果的统一出口。
    RegistrationContext ctx;

    // 8. 单次运行沿用统一流水线编排，具体算法由配置决定。
    const bool ok = pipeline->run(ctx);

    // 9. 根据 pipeline 类型输出不同摘要，避免点特征和结构特征字段混杂。
    printSummary(ctx, !cfg.structure_path.empty());
    return ok ? 0 : 1;
}

bool RegistrationApp::isBatchYaml(const YAML::Node& node) {
    return node && node.IsMap() && node["pipeline"] && node["dataset"];
}

RegistrationApp::BatchConfig
RegistrationApp::loadBatchConfig(const std::filesystem::path& yaml_path) {
    // 批处理配置以 batch.yaml 所在目录为基准解析相对路径，便于配置整体迁移。
    const YAML::Node node = Config::load(yaml_path);
    BatchConfig cfg;
    cfg.name = yaml_utils::getString(node, "name", yaml_path.stem().string());

    const auto batch_dir = yaml_path.parent_path();
    cfg.pipeline_yaml = Config::resolvePath(batch_dir, yaml_utils::getString(node, "pipeline"));

    const YAML::Node dataset = node["dataset"];
    cfg.dataset.root = Config::resolvePath(batch_dir, yaml_utils::getString(dataset, "root"));
    cfg.dataset.pattern_source = yaml_utils::getString(dataset, "pattern_source", "source");
    cfg.dataset.pattern_target = yaml_utils::getString(dataset, "pattern_target", "target");
    cfg.dataset.include = yaml_utils::getVec<std::string>(dataset, "include", {});

    const YAML::Node output = node["output"];
    cfg.output_root = Config::resolvePath(
        batch_dir, yaml_utils::getString(output, "root", "../../outputs/batch"));
    cfg.save_visuals = yaml_utils::getBool(output, "save_visuals", true);
    cfg.summary_csv = yaml_utils::getBool(output, "summary_csv", true);
    return cfg;
}

std::filesystem::path RegistrationApp::resolveBatchOutputRoot(const BatchConfig& batch,
                                                              const PipelineConfig& pipeline_cfg) {
    if (batch.output_root.empty())
        return {};

    // `current` 表示结果自动落到当前流水线名称对应目录下，便于批量切换算法。
    std::filesystem::path out = batch.output_root;
    if (out.filename() == "current") {
        const std::string pipeline_name =
            !pipeline_cfg.name.empty() ? pipeline_cfg.name : batch.pipeline_yaml.stem().string();
        out = out.parent_path() / pipeline_name;
    }
    return out;
}

void RegistrationApp::writeSummaryCsv(const std::filesystem::path& csv_path,
                                      const std::vector<std::string>& sample_names,
                                      const std::vector<RegistrationResult>& results) {
    // 汇总表保留两类方法的核心数量字段，便于后续横向比较。
    std::ostringstream oss;
    oss << "sample_name,success,message,"
        << "num_keypoints_first,num_keypoints_second,"
        << "num_structures_first,num_structures_second,"
        << "num_raw_matches,num_filtered_matches,num_inliers,"
        << "inlier_ratio,t_total_ms\n";

    for (size_t i = 0; i < sample_names.size() && i < results.size(); ++i) {
        const auto& r = results[i];
        oss << file_utils::csvEscape(sample_names[i]) << "," << (r.success ? "1" : "0") << ","
            << file_utils::csvEscape(r.message) << "," << r.num_keypoints_first << ","
            << r.num_keypoints_second << "," << r.num_structures_first << ","
            << r.num_structures_second << "," << r.num_raw_matches << ","
            << r.num_filtered_matches << "," << r.num_inliers << "," << r.inlier_ratio << ","
            << r.t_total_ms << "\n";
    }

    file_utils::writeWholeFile(csv_path, oss.str());
}

int RegistrationApp::runBatch(const std::filesystem::path& batch_yaml) {
    // 1. 先加载批处理配置，再派生单样本运行时的基础 pipeline 配置。
    const BatchConfig batch = loadBatchConfig(batch_yaml);
    const PipelineConfig base_cfg = Config::loadPipeline(batch.pipeline_yaml);
    const std::filesystem::path output_root = resolveBatchOutputRoot(batch, base_cfg);

    // 2. 数据集扫描与输出根目录准备是批处理前置条件。
    DatasetLoader loader(batch.dataset);
    const std::vector<Sample> samples = loader.load();
    if (samples.empty()) {
        std::cerr << "No dataset samples found for batch config: " << batch_yaml.string() << "\n";
        return 7;
    }

    std::error_code ec;
    std::filesystem::create_directories(output_root, ec);

    std::vector<std::string> sample_names;
    std::vector<std::string> succeeded_names;
    std::vector<RegistrationResult> results;
    sample_names.reserve(samples.size());
    succeeded_names.reserve(samples.size());
    results.reserve(samples.size());

    int ok_count = 0;
    for (const auto& sample : samples) {
        // 3. 每个样本复用同一算法配置，只替换输入图像与输出目录。
        PipelineConfig cfg = base_cfg;
        cfg.image1_path = sample.source_path;
        cfg.image2_path = sample.target_path;
        cfg.output_dir = output_root / sample.name;
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
            continue;
        }

        RegistrationContext ctx;
        const bool ok = pipeline->run(ctx);
        printSummary(ctx, !cfg.structure_path.empty());
        sample_names.push_back(sample.name);
        results.push_back(ctx.result);
        if (ok) {
            ++ok_count;
            succeeded_names.push_back(sample.name);
        }
    }

    // 4. 批处理汇总结果独立落盘，便于后续比较不同样本表现。
    if (batch.summary_csv) {
        const auto csv_path = output_root / "summary.csv";
        writeSummaryCsv(csv_path, sample_names, results);
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
