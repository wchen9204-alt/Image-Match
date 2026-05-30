#include "utils/file_utils.h"

#include <algorithm>
#include <sstream>
#include <system_error>

namespace fs = std::filesystem;

namespace ir {
namespace file_utils {

bool ensureDirectory(const fs::path& dir) {
    // 目录创建使用 error_code 分支，避免把“目录已存在”等情况升级为异常。
    if (dir.empty())
        return false;
    std::error_code ec;
    fs::create_directories(dir, ec);
    return !ec;
}

bool fileExists(const fs::path& path) {
    std::error_code ec;
    return !path.empty() && fs::exists(path, ec) && !ec;
}

std::string readWholeFile(const fs::path& path) {
    // 文本整读主要服务于配置、报告和 CSV 这类中小型文件。
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return {};
    std::ostringstream oss;
    oss << f.rdbuf();
    return oss.str();
}

bool writeWholeFile(const fs::path& path, const std::string& content) {
    // 写文件前自动补目录，减少调用方重复处理输出路径准备逻辑。
    if (path.has_parent_path())
        ensureDirectory(path.parent_path());
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
        return false;
    f.write(content.data(), static_cast<std::streamsize>(content.size()));
    return f.good();
}

std::string csvEscape(const std::string& s) {
    // 仅在必要时加引号，兼顾 CSV 合法性与结果文件可读性。
    bool needs_quote = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            needs_quote = true;
            break;
        }
    }
    if (!needs_quote)
        return s;

    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"')
            out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

std::string makeStem(const std::string& sample, const std::string& pipeline) {
    // 文件名前缀只保留安全字符，避免跨平台路径兼容问题。
    std::string out;
    out.reserve(sample.size() + pipeline.size() + 1);
    auto safe = [&out](const std::string& s) {
        for (char c : s) {
            const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                            (c >= '0' && c <= '9') || c == '_' || c == '-';
            out.push_back(ok ? c : '_');
        }
    };
    safe(sample);
    out.push_back('_');
    safe(pipeline);
    return out;
}

std::vector<fs::path> listSubdirectories(const fs::path& root) {
    // 子目录列表统一排序，保证批处理扫描结果具有稳定顺序。
    std::vector<fs::path> out;
    std::error_code ec;
    if (!fs::exists(root, ec) || ec)
        return out;
    for (const auto& e : fs::directory_iterator(root, ec)) {
        if (ec)
            break;
        if (e.is_directory())
            out.push_back(e.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace file_utils
} // namespace ir
