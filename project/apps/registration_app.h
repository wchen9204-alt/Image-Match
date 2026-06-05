#pragma once

#include <vector>

#include <filesystem>
#include <string>

#include <yaml-cpp/yaml.h>

#include "core/config.h"
#include "core/result.h"
#include "data/evaluation_data.h"
#include "dataset/dataset_loader.h"

namespace ir {

/// 命令行应用入口。
///
/// 负责解析单次配准与批处理两类运行模式，组织 YAML 加载、参数覆盖、
/// 输出目录规划以及汇总结果写出。
class RegistrationApp {
public:
    /// 单次运行的命令行参数。
    struct Args {
        std::filesystem::path pipeline_yaml;
        std::filesystem::path image1;
        std::filesystem::path image2;
        std::filesystem::path output_dir;
    };

    /// 根据已解析参数执行应用逻辑。
    static int run(const Args& args);

    /// 从原始命令行参数构造 `Args` 并执行。
    static int run(int argc, char** argv);

    /// 输出命令行用法说明。
    static void printUsage(const std::string& exe);

private:
    /// 输出模式，区分单次运行与批处理运行。
    enum class OutputMode {
        SINGLE,
        BATCH
    };

    /// 批量评测模式使用的配置快照。
    struct BatchConfig {
        std::string name;
        std::filesystem::path pipeline_yaml;
        DatasetLoader::Options dataset;
        std::filesystem::path output_root;
        bool save_visuals = true;
        bool summary_csv = true;
    };

    /// 执行单个样本的配准流程。
    static int runSingle(const Args& args);

    /// 执行一个 batch.yaml 中定义的批量任务。
    static int runBatch(const std::filesystem::path& batch_yaml);

    /// 判断 YAML 是否符合批处理配置结构。
    static bool isBatchYaml(const YAML::Node& node);

    /// 判断 YAML 是否符合对比配置结构。
    static bool isCompareYaml(const YAML::Node& node);

    /// 执行对比任务，遍历 (检测器 × 描述子) 组合输出总表。
    static int runCompare(const std::filesystem::path& compare_yaml);

    /// 从 batch.yaml 中解析批处理配置。
    static BatchConfig loadBatchConfig(const std::filesystem::path& yaml_path);

    /// 解析批处理输出根目录。
    static std::filesystem::path resolveBatchOutputRoot(const BatchConfig& batch,
                                                        const PipelineConfig& pipeline_cfg);

    /// 生成统一的输出目录层级：single|batch / keypoint|structure / pipeline / sample。
    static std::filesystem::path buildOutputDir(OutputMode mode,
                                                const std::filesystem::path& base_root,
                                                const PipelineConfig& cfg,
                                                const std::string& sample_name);

    /// 将逐样本结果写出为 CSV，便于离线统计与复盘。
    static void writeSummaryCsv(const std::filesystem::path& csv_path,
                                MethodFamily family,
                                const std::vector<std::string>& sample_names,
                                const std::vector<RegistrationResult>& results,
                                const std::vector<EvaluationData>& evaluations);
};

} // namespace ir
