#include "registration_app.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "core/config.h"
#include "core/context.h"
#include "dataset/dataset_loader.h"
#include "pipeline/feature_pipeline.h"
#include "utils/file_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

std::string fmtMs(double v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << v << " ms";
    return oss.str();
}

void printSummary(const RegistrationContext& ctx) {
    const auto& r = ctx.result;
    std::cout << "\n================ Registration summary ================\n";
    std::cout << "  status        : " << (r.success ? "OK" : "FAILED") << "\n";
    std::cout << "  message       : " << r.message << "\n";
    std::cout << "  keypoints     : " << r.num_keypoints_first << " / "
                                       << r.num_keypoints_second << "\n";
    std::cout << "  raw matches   : " << r.num_raw_matches << "\n";
    std::cout << "  filtered      : " << r.num_filtered_matches << "\n";
    std::cout << "  inliers       : " << r.num_inliers
                                       << " (" << std::fixed << std::setprecision(3)
                                       << r.inlier_ratio << ")\n";
    std::cout << "  -- timings --\n";
    std::cout << "  load          : " << fmtMs(r.t_load_ms)     << "\n";
    std::cout << "  extract       : " << fmtMs(r.t_extract_ms)  << "\n";
    std::cout << "  match         : " << fmtMs(r.t_match_ms)    << "\n";
    std::cout << "  filter        : " << fmtMs(r.t_filter_ms)   << "\n";
    std::cout << "  geometry      : " << fmtMs(r.t_geometry_ms) << "\n";
    std::cout << "  warp          : " << fmtMs(r.t_warp_ms)     << "\n";
    std::cout << "  TOTAL         : " << fmtMs(r.t_total_ms)    << "\n";
    std::cout << "======================================================\n";
}

} // namespace

void RegistrationApp::printUsage(const std::string& exe) {
    std::cout
        << "Usage:\n"
        << "  " << exe << " <pipeline.yaml> [image1] [image2] [output_dir]\n\n"
        << "  " << exe << " <batch.yaml>\n\n"
        << "Examples:\n"
        << "  " << exe << " configs/pipeline/sift_pipeline.yaml\n"
        << "  " << exe << " configs/pipeline/batch_pipeline.yaml\n"
        << "  " << exe << " configs/pipeline/orb_pipeline.yaml a.jpg b.jpg outputs\n"
        << std::endl;
}

int RegistrationApp::runSingle(const Args& args) {
    // 1. 检查 pipeline YAML 文件是否传入并存在。
    if (args.pipeline_yaml.empty()) {
        std::cerr << "Pipeline YAML path is empty.\n";
        return 2;
    }
    if (!fs::exists(args.pipeline_yaml)) {
        std::cerr << "Pipeline YAML not found: "
                  << args.pipeline_yaml.string() << "\n";
        return 2;
    }

    // 2. 加载 pipeline 配置，这是后续创建 SIFT/ORB 等组件的入口。
    PipelineConfig cfg;
    try {
        cfg = Config::loadPipeline(args.pipeline_yaml);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load pipeline YAML: " << e.what() << "\n";
        return 3;
    }

    // 3. 如果命令行传入图片或输出目录，则覆盖 YAML 中的默认配置。
    if (!args.image1.empty()) cfg.image1_path = fs::weakly_canonical(args.image1);
    if (!args.image2.empty()) cfg.image2_path = fs::weakly_canonical(args.image2);
    if (!args.output_dir.empty()) cfg.output_dir = fs::weakly_canonical(args.output_dir);

    // 4. 检查必须的输入图片路径。
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

    // 5. 创建输出目录，后续匹配图和配准结果会写到这里。
    if (!cfg.output_dir.empty()) {
        std::error_code ec;
        fs::create_directories(cfg.output_dir, ec);
    }

    // 6. 创建并配置配准管道，configure 会根据 YAML 创建具体算法组件。
    FeaturePipeline pipeline;
    if (!pipeline.configure(cfg)) {
        std::cerr << "Pipeline configure failed.\n";
        return 6;
    }

    // 7. 创建上下文对象，用于保存图片、特征、匹配、几何结果等中间数据。
    RegistrationContext ctx;

    // 8. 执行完整 pipeline；SIFT 特征提取等核心步骤会在这里触发。
    const bool ok = pipeline.run(ctx);

    // 9. 打印最终统计结果。
    printSummary(ctx);
    return ok ? 0 : 1;
}

bool RegistrationApp::isBatchYaml(const YAML::Node& node) {
    return node && node.IsMap() && node["pipeline"] && node["dataset"];
}

RegistrationApp::BatchConfig
RegistrationApp::loadBatchConfig(const std::filesystem::path& yaml_path) {
    const YAML::Node node = Config::load(yaml_path);
    BatchConfig cfg;
    cfg.name = yaml_utils::getString(node, "name", yaml_path.stem().string());

    const auto batch_dir = yaml_path.parent_path();
    cfg.pipeline_yaml = Config::resolvePath(
        batch_dir, yaml_utils::getString(node, "pipeline"));

    const YAML::Node dataset = node["dataset"];
    cfg.dataset.root = Config::resolvePath(
        batch_dir, yaml_utils::getString(dataset, "root"));
    cfg.dataset.pattern_source =
        yaml_utils::getString(dataset, "pattern_source", "source");
    cfg.dataset.pattern_target =
        yaml_utils::getString(dataset, "pattern_target", "target");
    cfg.dataset.include =
        yaml_utils::getVec<std::string>(dataset, "include", {});

    const YAML::Node output = node["output"];
    cfg.output_root = Config::resolvePath(
        batch_dir, yaml_utils::getString(output, "root", "../../outputs/batch"));
    cfg.save_visuals = yaml_utils::getBool(output, "save_visuals", true);
    cfg.summary_csv  = yaml_utils::getBool(output, "summary_csv", true);
    return cfg;
}

std::filesystem::path RegistrationApp::resolveBatchOutputRoot(
    const BatchConfig& batch,
    const PipelineConfig& pipeline_cfg) {
    if (batch.output_root.empty()) return {};

    std::filesystem::path out = batch.output_root;
    if (out.filename() == "current") {
        const std::string pipeline_name =
            !pipeline_cfg.name.empty()
                ? pipeline_cfg.name
                : batch.pipeline_yaml.stem().string();
        out = out.parent_path() / pipeline_name;
    }
    return out;
}

void RegistrationApp::writeSummaryCsv(
    const std::filesystem::path& csv_path,
    const std::vector<std::string>& sample_names,
    const std::vector<RegistrationResult>& results) {
    std::ostringstream oss;
    oss << "sample_name,success,message,"
        << "num_keypoints_first,num_keypoints_second,"
        << "num_raw_matches,num_filtered_matches,num_inliers,"
        << "inlier_ratio,t_total_ms\n";

    for (size_t i = 0; i < sample_names.size() && i < results.size(); ++i) {
        const auto& r = results[i];
        oss << file_utils::csvEscape(sample_names[i]) << ","
            << (r.success ? "1" : "0") << ","
            << file_utils::csvEscape(r.message) << ","
            << r.num_keypoints_first << ","
            << r.num_keypoints_second << ","
            << r.num_raw_matches << ","
            << r.num_filtered_matches << ","
            << r.num_inliers << ","
            << r.inlier_ratio << ","
            << r.t_total_ms << "\n";
    }

    file_utils::writeWholeFile(csv_path, oss.str());
}

int RegistrationApp::runBatch(const std::filesystem::path& batch_yaml) {
    const BatchConfig batch = loadBatchConfig(batch_yaml);
    const PipelineConfig base_cfg = Config::loadPipeline(batch.pipeline_yaml);
    const std::filesystem::path output_root =
        resolveBatchOutputRoot(batch, base_cfg);

    DatasetLoader loader(batch.dataset);
    const std::vector<Sample> samples = loader.load();
    if (samples.empty()) {
        std::cerr << "No dataset samples found for batch config: "
                  << batch_yaml.string() << "\n";
        return 7;
    }

    std::error_code ec;
    std::filesystem::create_directories(output_root, ec);

    std::vector<std::string> sample_names;
    std::vector<RegistrationResult> results;
    sample_names.reserve(samples.size());
    results.reserve(samples.size());

    int ok_count = 0;
    for (const auto& sample : samples) {
        PipelineConfig cfg = base_cfg;
        cfg.image1_path = sample.source_path;
        cfg.image2_path = sample.target_path;
        cfg.output_dir  = output_root / sample.name;
        if (!batch.save_visuals) {
            cfg.draw_matches = false;
            cfg.warp = false;
        }
        cfg.show_source_window = false;
        cfg.show_target_window = false;
        cfg.show_warped_window = false;

        FeaturePipeline pipeline;
        if (!pipeline.configure(cfg)) {
            RegistrationResult failed;
            failed.success = false;
            failed.message = "pipeline configure failed";
            sample_names.push_back(sample.name);
            results.push_back(failed);
            continue;
        }

        RegistrationContext ctx;
        const bool ok = pipeline.run(ctx);
        printSummary(ctx);
        sample_names.push_back(sample.name);
        results.push_back(ctx.result);
        if (ok) ++ok_count;
    }

    if (batch.summary_csv) {
        const auto csv_path = output_root / "summary.csv";
        writeSummaryCsv(csv_path, sample_names, results);
        std::cout << "Wrote summary CSV: " << csv_path.string() << "\n";
    }

    std::cout << "\nBatch summary: " << ok_count << " / " << samples.size()
              << " samples succeeded.\n";
    return ok_count == static_cast<int>(samples.size()) ? 0 : 1;
}

int RegistrationApp::run(const Args& args) {
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
    if (argc < 2) {
        printUsage(argc > 0 ? argv[0] : "registration_app");
        return 1;
    }

    Args args;
    args.pipeline_yaml = argv[1];
    if (argc >= 3) args.image1     = argv[2];
    if (argc >= 4) args.image2     = argv[3];
    if (argc >= 5) args.output_dir = argv[4];

    return run(args);
}

} // namespace ir
