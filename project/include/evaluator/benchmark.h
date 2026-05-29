#pragma once

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "core/pipeline_manager.h"
#include "core/registration.h"
#include "dataset/dataset_loader.h"
#include "evaluator/evaluator.h"
#include "evaluator/statistics.h"

namespace ir {

// ---------------------------------------------------------------------------
// Benchmark：对多个 pipeline 和数据集样本执行批量评测。
// 输出：
//   - <output>/csv/<pipeline>.csv  : 每个样本一行
//   - <output>/reports/summary.csv : 每个 pipeline 的汇总统计
// ---------------------------------------------------------------------------
class Benchmark {
public:
    struct Config {
        DatasetLoader::Options             dataset;
        std::vector<std::filesystem::path> pipeline_yamls;
        std::filesystem::path              metrics_yaml;
        std::filesystem::path              output_root;     // 通常为 "outputs"。
        std::filesystem::path              csv_dir;         // 相对于 output_root。
        std::filesystem::path              reports_dir;
        std::filesystem::path              benchmark_dir;
        bool                               save_visuals = false;
    };

    Benchmark() = default;

    // 从 benchmark.yaml 加载配置，并以该文件所在目录解析相对路径。
    static Config loadConfig(const std::filesystem::path& yaml_path);

    int run(const Config& cfg);

private:
    void writePerPipelineCsv(const std::filesystem::path& path,
                             const std::vector<Sample>& samples,
                             const std::vector<EvaluationData>& evals,
                             const std::vector<RegistrationResult>& results) const;

    void writeSummary(const std::filesystem::path& path,
                      const std::vector<std::shared_ptr<Registration>>& regs,
                      const std::vector<Statistics>& stats) const;
};

} // namespace ir
