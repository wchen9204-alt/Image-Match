#pragma once

#include <vector>

#include <filesystem>
#include <string>

#include <yaml-cpp/yaml.h>

#include "core/config.h"
#include "core/result.h"
#include "dataset/dataset_loader.h"

namespace ir {

// ---------------------------------------------------------------------------
// RegistrationApp：基于 FeaturePipeline 的命令行入口。
//
// 用法：
//     registration_app <pipeline.yaml> [image1] [image2] [output_dir]
//
// 后三个参数可选；未传入时使用 pipeline YAML 中的配置。
// ---------------------------------------------------------------------------
class RegistrationApp {
public:
    struct Args {
        std::filesystem::path pipeline_yaml;
        std::filesystem::path image1;
        std::filesystem::path image2;
        std::filesystem::path output_dir;
    };

    static int run(const Args& args);
    static int run(int argc, char** argv);

    static void printUsage(const std::string& exe);

private:
    struct BatchConfig {
        std::string name;
        std::filesystem::path pipeline_yaml;
        DatasetLoader::Options dataset;
        std::filesystem::path output_root;
        bool save_visuals = true;
        bool summary_csv  = true;
    };

    static int runSingle(const Args& args);
    static int runBatch(const std::filesystem::path& batch_yaml);
    static bool isBatchYaml(const YAML::Node& node);
    static BatchConfig loadBatchConfig(const std::filesystem::path& yaml_path);
    static std::filesystem::path resolveBatchOutputRoot(
        const BatchConfig& batch,
        const PipelineConfig& pipeline_cfg);
    static void writeSummaryCsv(
        const std::filesystem::path& csv_path,
        const std::vector<std::string>& sample_names,
        const std::vector<RegistrationResult>& results);
};

} // namespace ir
