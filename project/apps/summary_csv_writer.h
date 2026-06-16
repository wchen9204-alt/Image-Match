#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/result.h"
#include "data/evaluation_data.h"

namespace ir::summary_csv {

/// 按方法族输出批处理结果 CSV；该文件优先服务人工查看与报告生成。
void write(const std::filesystem::path& csv_path,
           MethodFamily family,
           const std::vector<std::string>& sample_names,
           const std::vector<RegistrationResult>& results,
           const std::vector<EvaluationData>& evaluations);

} // namespace ir::summary_csv

