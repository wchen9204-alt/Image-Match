#include "registration_app.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "core/config.h"
#include "core/context.h"
#include "dataset/dataset_loader.h"
#include "registration_app_helpers.h"
#include "summary_csv_writer.h"
#include "utils/file_utils.h"
#include "utils/yaml_utils.h"

namespace fs = std::filesystem;

namespace ir {

namespace app_helpers = registration_app_helpers;

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
        app_helpers::normalizeOutputBaseRoot(!args.output_dir.empty()
                                                 ? fs::weakly_canonical(args.output_dir)
                                                 : cfg.output_dir);
    const std::string sample_name =
        app_helpers::sampleStemFromPaths(cfg.image1_path, cfg.image2_path);
    cfg.output_dir = buildOutputDir(OutputMode::SINGLE, single_root, cfg, sample_name);

    if (!cfg.output_dir.empty()) {
        std::error_code ec;
        fs::create_directories(cfg.output_dir, ec);
    }

    // 6. 根据配置选择 keypoint 流水线或结构流水线。
    auto pipeline = app_helpers::createPipelineForConfig(cfg);
    if (!pipeline->configure(cfg)) {
        std::cerr << "Pipeline configure failed.\n";
        return 6;
    }

    // 7. 上下文承担阶段间数据交换职责，也是最终统计结果的统一出口。
    RegistrationContext ctx;

    // 8. 单次运行沿用统一流水线编排，具体算法由配置决定。
    const bool ok = pipeline->run(ctx);
    app_helpers::writeRunSummaryFiles(ctx, cfg, sample_name);
    writeSummaryCsv(cfg.output_dir / "summary.csv",
                    cfg.methodFamily(),
                    std::vector<std::string>{sample_name},
                    std::vector<RegistrationResult>{ctx.result},
                    std::vector<EvaluationData>{ctx.evaluation});
    pipeline->showWindows(ctx);

    // 9. 根据方法族输出对应摘要。
    app_helpers::printSummary(ctx, cfg.methodFamily());
    return ok ? 0 : 1;
}

RegistrationApp::RunMode RegistrationApp::detectRunMode(const YAML::Node& node) {
    if (node && node.IsMap() && node["base_pipeline"] && node["combinations"]) {
        return RunMode::COMPARE;
    }
    if (node && node.IsMap() && node["pipeline"] && node["dataset"]) {
        return RunMode::BATCH;
    }
    return RunMode::SINGLE;
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
    app_helpers::loadDatasetNamingOptions(ds, datasetOpts);
    datasetOpts.include = yaml_utils::getVec<std::string>(ds, "include", {});

    // 3. 输出根目录
    const YAML::Node output = cfg["output"];
    const auto outputRoot = Config::resolvePath(baseDir,
        yaml_utils::getString(output, "root", "../../outputs/compare"));
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

            auto pipeline = app_helpers::createPipelineForConfig(pCfg);
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
    bool compareStructure = static_cast<bool>(cfg["base_structure"]);
    for (const auto& combo : combos) {
        if (combo["direct"]) {
            compareDirect = true;
            break;
        }
        if (combo["extractor"] || combo["descriptor"]) {
            compareStructure = true;
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

            app_helpers::applyCompareOverrides(pipelineCfg, cfg, combo, baseDir, outputRoot, label);

            runCombination(label, pipelineCfg);
        }
    } else if (compareStructure) {
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
            app_helpers::applyCompareOverrides(pipelineCfg, cfg, combo, baseDir, outputRoot, label);

            runCombination(label, pipelineCfg);
        }
    } else {
        for (const auto& combo : combos) {
            const std::string pipelineEntry = yaml_utils::getString(combo, "pipeline");
            const auto comboPipelineYaml = pipelineEntry.empty()
                ? pipeline_yaml
                : Config::resolvePath(baseDir, pipelineEntry);

            PipelineConfig pipelineCfg = Config::loadPipeline(comboPipelineYaml);
            if (pipelineCfg.methodFamily() != MethodFamily::KEYPOINT) {
                std::cerr << "Compare: skip non-keypoint pipeline "
                          << comboPipelineYaml.string() << "\n";
                continue;
            }

            std::string label = yaml_utils::getString(combo, "label");
            if (label.empty()) {
                label = comboPipelineYaml.stem().string();
            }

            app_helpers::applyCompareOverrides(pipelineCfg, cfg, combo, baseDir, outputRoot, label);

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
    app_helpers::loadDatasetNamingOptions(dataset, cfg.dataset);
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
    return app_helpers::normalizeOutputBaseRoot(batch.output_root);
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
    summary_csv::write(csv_path, family, sample_names, results, evaluations);
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

        auto pipeline = app_helpers::createPipelineForConfig(cfg);
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
        app_helpers::writeRunSummaryFiles(ctx, cfg, sample.name);
        app_helpers::printSummary(ctx, cfg.methodFamily());
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
    // 入口先识别运行模式，再统一分发到单样本、批处理或方法对比分支。
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
        switch (detectRunMode(node)) {
        case RunMode::COMPARE:
            return RegistrationApp::runCompare(args.pipeline_yaml);
        case RunMode::BATCH:
            return RegistrationApp::runBatch(args.pipeline_yaml);
        case RunMode::SINGLE:
        default:
            return runSingle(args);
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to read YAML: " << e.what() << "\n";
        return 3;
    }
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
