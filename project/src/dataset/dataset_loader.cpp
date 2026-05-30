#include "dataset/dataset_loader.h"

#include <algorithm>
#include <fstream>

#include "utils/file_utils.h"
#include "utils/logger.h"

namespace fs = std::filesystem;

namespace ir {

DatasetLoader::DatasetLoader(const Options& opt) : _opt(opt) {}

bool DatasetLoader::resolveImage(const fs::path& dir,
                                 const std::string& stem,
                                 fs::path& out) const {
    // 同一数据集可能混用多种扩展名，按配置顺序尝试匹配第一个存在的文件。
    for (const auto& ext : _opt.extensions) {
        const fs::path p = dir / (stem + ext);
        if (file_utils::fileExists(p)) {
            out = p;
            return true;
        }
    }
    return false;
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
        dirs = file_utils::listSubdirectories(_opt.root);
    }

    // 3. 每个子目录解析为一个样本，缺少任一输入图像时跳过该样本。
    for (const auto& d : dirs) {
        Sample s;
        s.name = d.filename().string();
        if (!resolveImage(d, _opt.pattern_source, s.source_path)) {
            IR_LOG_WARN("DatasetLoader: missing source for ", d.string());
            continue;
        }
        if (!resolveImage(d, _opt.pattern_target, s.target_path)) {
            IR_LOG_WARN("DatasetLoader: missing target for ", d.string());
            continue;
        }
        tryLoadGroundTruth(d, s.H_gt);
        out.push_back(std::move(s));
    }

    // 4. 最终按样本名排序，保证批处理输出次序稳定。
    std::sort(
        out.begin(), out.end(), [](const Sample& a, const Sample& b) { return a.name < b.name; });

    IR_LOG_INFO("DatasetLoader: loaded ", out.size(), " samples from ", _opt.root.string());
    return out;
}

} // namespace ir
