#include "dataset/dataset_loader.h"

#include <algorithm>
#include <cctype>
#include <fstream>

#include "utils/file_utils.h"
#include "utils/logger.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

// 自然排序比较器：把连续数字段按数值比较，保证 test2 排在 test10 前面。
bool naturalLess(const std::string& a, const std::string& b) {
    size_t i = 0;
    size_t j = 0;
    while (i < a.size() && j < b.size()) {
        const unsigned char ca = static_cast<unsigned char>(a[i]);
        const unsigned char cb = static_cast<unsigned char>(b[j]);

        if (std::isdigit(ca) && std::isdigit(cb)) {
            // 先跳过前导零，再比较数字有效长度和值；长度更短代表数值更小。
            size_t ai = i;
            size_t bj = j;
            while (ai < a.size() && a[ai] == '0') {
                ++ai;
            }
            while (bj < b.size() && b[bj] == '0') {
                ++bj;
            }

            size_t ae = ai;
            size_t be = bj;
            while (ae < a.size() && std::isdigit(static_cast<unsigned char>(a[ae]))) {
                ++ae;
            }
            while (be < b.size() && std::isdigit(static_cast<unsigned char>(b[be]))) {
                ++be;
            }

            const size_t aDigits = ae - ai;
            const size_t bDigits = be - bj;
            if (aDigits != bDigits) {
                return aDigits < bDigits;
            }
            for (size_t k = 0; k < aDigits; ++k) {
                if (a[ai + k] != b[bj + k]) {
                    return a[ai + k] < b[bj + k];
                }
            }

            // 数值完全相同（如 test01/test1）时，用原数字段长度保持稳定次序。
            const size_t aFullDigits = ae - i;
            const size_t bFullDigits = be - j;
            if (aFullDigits != bFullDigits) {
                return aFullDigits < bFullDigits;
            }

            i = ae;
            j = be;
            continue;
        }

        // 非数字段按大小写不敏感比较；完全相同再回退到原字符保证确定性。
        const char la = static_cast<char>(std::tolower(ca));
        const char lb = static_cast<char>(std::tolower(cb));
        if (la != lb) {
            return la < lb;
        }
        if (a[i] != b[j]) {
            return a[i] < b[j];
        }
        ++i;
        ++j;
    }
    return a.size() < b.size();
}

} // namespace

DatasetLoader::DatasetLoader(const Options& opt) : _opt(opt) {}

bool DatasetLoader::resolveImage(const fs::path& dir,
                                 const std::vector<std::string>& stems,
                                 fs::path& out) const {
    // 同一数据集可能混用 source/target 或 moving/reference 命名，按候选关键词和扩展名顺序查找。
    for (const auto& stem : stems) {
        if (stem.empty()) {
            continue;
        }
        for (const auto& ext : _opt.extensions) {
            const fs::path p = dir / (stem + ext);
            if (file_utils::fileExists(p)) {
                out = p;
                return true;
            }
        }
    }
    return false;
}

std::vector<std::string> DatasetLoader::sourcePatterns() const {
    return _opt.pattern_sources;
}

std::vector<std::string> DatasetLoader::targetPatterns() const {
    return _opt.pattern_targets;
}

bool DatasetLoader::tryLoadGroundTruth(const fs::path& dir, cv::Mat& H_gt) const {
    // 真值矩阵按约定文件名读取，缺失时保持样本可用但不附带评测真值。
    const fs::path p = dir / "H_gt.txt";
    if (!file_utils::fileExists(p))
        return false;

    std::ifstream f(p);
    if (!f)
        return false;

    std::vector<double> v;
    v.reserve(9);
    double x;
    while (f >> x)
        v.push_back(x);
    if (v.size() != 9) {
        IR_LOG_WARN("H_gt.txt at ", p.string(), " expected 9 values, got ", v.size());
        return false;
    }
    cv::Mat H(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i)
        H.at<double>(i / 3, i % 3) = v[i];
    H_gt = H;
    return true;
}

std::vector<Sample> DatasetLoader::load() const {
    std::vector<Sample> out;

    // 1. 根目录校验失败时直接返回空集，避免目录遍历产生误导性结果。
    if (_opt.root.empty()) {
        IR_LOG_ERROR("DatasetLoader: empty root.");
        return out;
    }
    if (!fs::exists(_opt.root)) {
        IR_LOG_ERROR("DatasetLoader: root not found: ", _opt.root.string());
        return out;
    }

    // 2. 支持显式 include 列表与自动扫描两种工作模式，满足可控复现与全量遍历。
    std::vector<fs::path> dirs;
    if (!_opt.include.empty()) {
        for (const auto& name : _opt.include) {
            const fs::path d = _opt.root / name;
            if (fs::is_directory(d))
                dirs.push_back(d);
            else
                IR_LOG_WARN("DatasetLoader: include entry not a directory: ", d.string());
        }
    } else {
        // Only direct child directories are samples: root/sample.
        for (const auto& entry : file_utils::listSubdirectories(_opt.root)) {
            fs::path source;
            fs::path target;
            if (resolveImage(entry, sourcePatterns(), source) &&
                resolveImage(entry, targetPatterns(), target)) {
                dirs.push_back(entry);
            }
        }
    }
    // 3. 每个子目录解析为一个样本，缺少任一输入图像时跳过该样本。
    const std::vector<std::string> sourceStems = sourcePatterns();
    const std::vector<std::string> targetStems = targetPatterns();
    for (const auto& d : dirs) {
        Sample s;
        s.name = d.filename().string();
        if (!resolveImage(d, sourceStems, s.source_path)) {
            IR_LOG_WARN("DatasetLoader: missing source for ", d.string());
            continue;
        }
        if (!resolveImage(d, targetStems, s.target_path)) {
            IR_LOG_WARN("DatasetLoader: missing target for ", d.string());
            continue;
        }
        tryLoadGroundTruth(d, s.H_gt);
        out.push_back(std::move(s));
    }

    // 4. 最终按样本名自然排序，保证 test2 排在 test10 前面。
    std::sort(out.begin(), out.end(), [](const Sample& a, const Sample& b) {
        return naturalLess(a.name, b.name);
    });

    IR_LOG_INFO("DatasetLoader: loaded ", out.size(), " samples from ", _opt.root.string());
    return out;
}

} // namespace ir

