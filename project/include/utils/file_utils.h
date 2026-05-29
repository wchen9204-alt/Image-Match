#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace ir {

// ---------------------------------------------------------------------------
// 文件和路径辅助函数。
// ---------------------------------------------------------------------------
namespace file_utils {

bool ensureDirectory(const std::filesystem::path& dir);

bool fileExists(const std::filesystem::path& path);

std::string readWholeFile(const std::filesystem::path& path);

bool writeWholeFile(const std::filesystem::path& path, const std::string& content);

// 对单个 CSV 字段做安全转义。
std::string csvEscape(const std::string& s);

// 根据样本名和 pipeline 名生成安全文件名前缀。
std::string makeStem(const std::string& sample_name,
                     const std::string& pipeline_name);

// 列出 root 下一级子目录，并按名称排序。
std::vector<std::filesystem::path> listSubdirectories(const std::filesystem::path& root);

} // namespace file_utils
} // namespace ir
