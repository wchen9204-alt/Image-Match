#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace ir {

/// 文件和路径相关的辅助函数。
namespace file_utils {

/// 确保目录存在，不存在时递归创建。
bool ensureDirectory(const std::filesystem::path& dir);

/// 判断路径是否为现有文件。
bool fileExists(const std::filesystem::path& path);

/// 读取整个文件内容到字符串。
std::string readWholeFile(const std::filesystem::path& path);

/// 将字符串完整写入文件。
bool writeWholeFile(const std::filesystem::path& path, const std::string& content);

/// 对单个 CSV 字段做安全转义。
std::string csvEscape(const std::string& s);

/// 根据样本名和 pipeline 名生成安全的文件名主体。
std::string makeStem(const std::string& sample_name,
                     const std::string& pipeline_name);

/// 枚举 root 下的一层子目录，并按名称排序。
std::vector<std::filesystem::path> listSubdirectories(const std::filesystem::path& root);

} // namespace file_utils
} // namespace ir
