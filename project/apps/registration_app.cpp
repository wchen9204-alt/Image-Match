#include "registration_app.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>


#include "core/config.h"
#include "core/context.h"
#include "core/registration.h"
#include "dataset/dataset_loader.h"
#include "registration_app_helpers.h"
#include "summary_csv_writer.h"
#include "utils/file_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace fs = std::filesystem;

namespace ir {

namespace app_helpers = registration_app_helpers;

namespace {

fs::path findGlobalConfig(const fs::path& pipeline_yaml, const fs::path& filename) {
    fs::path current = pipeline_yaml.parent_path();
    while (!current.empty()) {
        if (current.filename() == "configs") {
            return current / filename;
        }
        const fs::path parent = current.parent_path();
        if (parent == current) {
            break;
        }
        current = parent;
    }
    return fs::path{"project/configs"} / filename;
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
        IR_LOG_ERROR("Pipeline YAML path is empty.");
        return 2;
    }
    if (!fs::exists(args.pipeline_yaml)) {
        IR_LOG_ERROR("Pipeline YAML not found: ", args.pipeline_yaml.string());
        return 2;
    }

    // 2. 加载 pipeline 配置；具体算法组件都由子 YAML 决定。
    PipelineConfig cfg;
    try {
        cfg = Config::loadPipeline(args.pipeline_yaml);
    } catch (const std::exception& e) {
        IR_LOG_ERROR("Failed to load pipeline YAML: ", e.what());
        return 3;
    }

    // 3. 命令行显式传入的图像和输出目录优先级高于 YAML。
    if (!args.image1.empty())
        cfg.image1_path = fs::weakly_canonical(args.image1);
    if (!args.image2.empty())
        cfg.image2_path = fs::weakly_canonical(args.image2);

    // 4. 单次运行必须具备一对输入图像。
    if (cfg.image1_path.empty() || cfg.image2_path.empty()) {
        IR_LOG_ERROR("Missing image1 / image2. Provide them either in the YAML "
                     "(io.image1 / io.image2) or as positional arguments.");
        return 4;
    }
    if (!fs::exists(cfg.image1_path)) {
        IR_LOG_ERROR("image1 not found: ", cfg.image1_path.string());
        return 5;
    }
    if (!fs::exists(cfg.image2_path)) {
        IR_LOG_ERROR("image2 not found: ", cfg.image2_path.string());
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
    Registration pipeline;
    if (!pipeline.configure(cfg)) {
        IR_LOG_ERROR("Pipeline configure failed.");
        return 6;
    }

    // 7. 上下文承担阶段间数据交换职责，也是最终统计结果的统一出口。
    RegistrationContext ctx;

    // 8. 单次运行沿用统一流水线编排，具体算法由配置决定。
    const bool ok = pipeline.run(ctx);
    app_helpers::writeRunSummaryFiles(ctx, cfg, sample_name);
    writeSummaryCsv(cfg.output_dir / "summary.csv",
                    cfg.methodFamily(),
                    std::vector<std::string>{sample_name},
                    std::vector<RegistrationResult>{ctx.result},
                    std::vector<EvaluationData>{ctx.evaluation});
    if (pipeline.pipeline()) {
        pipeline.pipeline()->showWindows(ctx);
    }

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

    // 4. 遍历组合
    const YAML::Node combos = cfg["combinations"];
    std::vector<std::string> labels;
    std::vector<int> successCounts;
    std::vector<int> totalCounts;
    std::vector<double> avgContainments;
    std::vector<double> avgTotalTimes;
    std::vector<std::string> failedCases;

    auto joinFailedCases = [](const std::vector<std::string>& names) {
        std::ostringstream joined;
        for (size_t i = 0; i < names.size(); ++i) {
            if (i > 0) {
                joined << "; ";
            }
            joined << names[i];
        }
        return joined.str();
    };

    auto runCombination = [&](const std::string& label, const PipelineConfig& pipelineCfg) {
        DatasetLoader loader(datasetOpts);
        const auto samples = loader.load();
        if (samples.empty()) {
            IR_LOG_ERROR("Compare: no samples for ", label);
            return;
        }

        int okCount = 0;
        double containmentSum = 0.0;
        int containmentValid = 0;
        // 所有实际运行样本的总耗时累计，口径与 summary.csv 的 AVERAGE 行一致。
        double totalTimeSumMs = 0.0;
        std::vector<std::string> failedSampleNames;
        std::vector<std::string> sampleNames;
        std::vector<RegistrationResult> results;
        std::vector<EvaluationData> evaluations;
        sampleNames.reserve(samples.size());
        results.reserve(samples.size());
        evaluations.reserve(samples.size());

        std::error_code ec;
        fs::create_directories(pipelineCfg.output_dir, ec);

        // 1. 对比组合的算法配置在样本循环外只构建一次。
        Registration pipeline;
        const bool pipelineConfigured = pipeline.configure(pipelineCfg);
        if (!pipelineConfigured) {
            IR_LOG_ERROR("Compare: pipeline configure failed for ", label);
        }

        for (const auto& sample : samples) {
            if (!pipelineConfigured) {
                RegistrationResult failed;
                failed.success = false;
                failed.message = "pipeline configure failed";
                failedSampleNames.push_back(sample.name);
                sampleNames.push_back(sample.name);
                results.push_back(failed);
                evaluations.push_back(EvaluationData{});
                continue;
            }

            // 2. 每次运行只传入当前样本的输入路径和独立输出目录。
            const PipelineRunOptions runOptions{
                sample.source_path,
                sample.target_path,
                pipelineCfg.output_dir / sample.name};
            RegistrationContext ctx;
            if (pipeline.run(ctx, runOptions)) {
                ++okCount;
            } else {
                failedSampleNames.push_back(sample.name);
            }
            app_helpers::writeRunSummaryFiles(ctx, pipelineCfg, sample.name);
            sampleNames.push_back(sample.name);
            results.push_back(ctx.result);
            evaluations.push_back(ctx.evaluation);
            totalTimeSumMs += ctx.result.t_total_ms;

            if (ctx.result.warp_overlap_containment >= 0.0) {
                containmentSum += ctx.result.warp_overlap_containment;
                ++containmentValid;
            }
        }

        labels.push_back(label);
        successCounts.push_back(okCount);
        totalCounts.push_back(static_cast<int>(samples.size()));
        avgContainments.push_back(containmentValid > 0 ? containmentSum / containmentValid : -1.0);
        avgTotalTimes.push_back(results.empty() ? 0.0
                                                : totalTimeSumMs / static_cast<double>(results.size()));
        failedCases.push_back(joinFailedCases(failedSampleNames));

        if (yaml_utils::getBool(output, "summary_csv", true)) {
            writeSummaryCsv(pipelineCfg.output_dir / "summary.csv",
                            pipelineCfg.methodFamily(),
                            sampleNames,
                            results,
                            evaluations);
        }

        IR_LOG_INFO(label, ": ", okCount, " / ", samples.size(), " succeeded");
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
                IR_LOG_WARN("Compare: skip non-direct pipeline ", comboPipelineYaml.string());
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
        std::error_code ec;
        fs::create_directories(outputRoot / "tmp", ec);

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
                IR_LOG_WARN("Compare: skip non-keypoint pipeline ", comboPipelineYaml.string());
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
    oss << "方法,成功数,总数,成功率,平均局部包含率,平均总耗时(ms),失败用例\n";
    for (size_t i = 0; i < labels.size(); ++i) {
        oss << file_utils::csvEscape(labels[i]) << ","
            << successCounts[i] << ","
            << totalCounts[i] << ","
            << (totalCounts[i] > 0
                    ? static_cast<double>(successCounts[i]) / totalCounts[i]
                    : 0.0) << ","
            << avgContainments[i] << ","
            << std::fixed << std::setprecision(3) << avgTotalTimes[i] << std::defaultfloat << ","
            << file_utils::csvEscape(failedCases[i]) << "\n";
    }
    file_utils::writeWholeFile(csvPath, oss.str());
    IR_LOG_INFO("Wrote comparison table: ", csvPath.string());
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
    cfg.visualization = node["visualization"];
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
    const auto pipeline_root = base_root / mode_dir / methodFamilyDir(cfg.methodFamily()) / cfg.name;
    // 单次运行只有一个当前样本，样本名层级只会制造冗余并掩盖本次运行结果；
    // 批处理仍按样本分目录，避免不同样本互相覆盖。
    return mode == OutputMode::BATCH ? pipeline_root / sample_name : pipeline_root;
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
        IR_LOG_ERROR("No dataset samples found for batch config: ", batch_yaml.string());
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

    // 3. 批处理只使用自身 YAML 的可视化设置，不继承单算法 pipeline 的输出偏好。
    PipelineConfig configured_cfg = base_cfg;
    Config::resetVisualization(configured_cfg);
    Config::applyVisualizationOverrides(configured_cfg, batch.visualization);
    configured_cfg.save_visuals = batch.save_visuals;
    configured_cfg.show_source_window = false;
    configured_cfg.show_target_window = false;
    configured_cfg.show_warped_window = false;

    Registration pipeline;
    const bool pipelineConfigured = pipeline.configure(configured_cfg);
    if (!pipelineConfigured) {
        IR_LOG_ERROR("Batch: pipeline configure failed for ", configured_cfg.name);
    }

    int ok_count = 0;
    for (size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
        const auto& sample = samples[sample_index];
        if (!pipelineConfigured) {
            // 配置失败单独记为该样本失败，避免影响整批结果汇总。
            RegistrationResult failed;
            failed.success = false;
            failed.message = "pipeline configure failed";
            sample_names.push_back(sample.name);
            results.push_back(failed);
            evaluations.push_back(EvaluationData{});
            continue;
        }

        // 4. 当前样本仅传入输入图像和独立输出目录，不再重建算法组件。
        const PipelineRunOptions runOptions{
            sample.source_path,
            sample.target_path,
            buildOutputDir(OutputMode::BATCH, output_root, configured_cfg, sample.name)};
        RegistrationContext ctx;
        const bool ok = pipeline.run(ctx, runOptions);
        app_helpers::writeRunSummaryFiles(ctx, configured_cfg, sample.name);
        app_helpers::printSummary(ctx, configured_cfg.methodFamily());
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
        IR_LOG_INFO("Wrote summary CSV: ", csv_path.string());
    }

    IR_LOG_INFO("Batch summary: ", ok_count, " / ", samples.size(), " samples succeeded.");
    if (!succeeded_names.empty()) {
        std::ostringstream successful;
        for (size_t i = 0; i < succeeded_names.size(); ++i) {
            if (i > 0) {
                successful << ", ";
            }
            successful << succeeded_names[i];
        }
        IR_LOG_INFO("Successful samples: ", successful.str());
    }
    return ok_count == static_cast<int>(samples.size()) ? 0 : 1;
}

int RegistrationApp::run(const Args& args) {
    // 入口先识别运行模式，再统一分发到单样本、批处理或方法对比分支。
    if (args.pipeline_yaml.empty()) {
        IR_LOG_ERROR("Pipeline YAML path is empty.");
        return 2;
    }
    if (!fs::exists(args.pipeline_yaml)) {
        IR_LOG_ERROR("YAML not found: ", args.pipeline_yaml.string());
        return 2;
    }

    try {
        std::string logging_error;
        const fs::path logging_config = findGlobalConfig(args.pipeline_yaml, "logging.yaml");
        if (!Logger::instance().loadConfig(logging_config, &logging_error)) {
            IR_LOG_WARN("Failed to load logging config ", logging_config.string(),
                        "; using built-in defaults: ", logging_error);
        }
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
        IR_LOG_ERROR("Failed to read YAML: ", e.what());
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
