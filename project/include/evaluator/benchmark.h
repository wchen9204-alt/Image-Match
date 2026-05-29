#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "core/registration.h"
#include "dataset/dataset_loader.h"
#include "evaluator/evaluator.h"
#include "evaluator/statistics.h"

namespace ir {

/// 批量基准测试入口。
///
/// 负责对多个 pipeline 和多个数据样本执行评测，并输出 CSV 和汇总报告。
class Benchmark {
public:
    /// 基准测试配置。
    struct Config {
        DatasetLoader::Options             dataset;
        std::vector<std::filesystem::path> pipeline_yamls;
        std::filesystem::path              metrics_yaml;
        std::filesystem::path              output_root;
        std::filesystem::path              csv_dir;
        std::filesystem::path              reports_dir;
        std::filesystem::path              benchmark_dir;
        bool                               save_visuals = false;
    };

    Benchmark() = default;

    /// 从 benchmark.yaml 读取配置。
    static Config loadConfig(const std::filesystem::path& yaml_path);

    /// 执行一次完整的批量评测。
    int run(const Config& cfg);

private:
    /// 写出单个 pipeline 的逐样本 CSV。
    void writePerPipelineCsv(const std::filesystem::path& path,
                             const std::vector<Sample>& samples,
                             const std::vector<EvaluationData>& evals,
                             const std::vector<RegistrationResult>& results) const;

    /// 写出汇总 CSV 报告。
    void writeSummary(const std::filesystem::path& path,
                      const std::vector<std::shared_ptr<Registration>>& regs,
                      const std::vector<Statistics>& stats) const;
};

} // namespace ir
