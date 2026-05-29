#include "dataset/dataset_loader.h"

#include <algorithm>
#include <fstream>

#include "utils/file_utils.h"
#include "utils/logger.h"

namespace fs = std::filesystem;

namespace ir {

DatasetLoader::DatasetLoader(const Options& opt) : opt_(opt) {}

bool DatasetLoader::resolveImage(const fs::path& dir,
                                 const std::string& stem,
                                 fs::path& out) const {
    for (const auto& ext : opt_.extensions) {
        const fs::path p = dir / (stem + ext);
        if (file_utils::fileExists(p)) {
            out = p;
            return true;
        }
    }
    return false;
}

bool DatasetLoader::tryLoadGroundTruth(const fs::path& dir, cv::Mat& H_gt) const {
    const fs::path p = dir / "H_gt.txt";
    if (!file_utils::fileExists(p)) return false;

    std::ifstream f(p);
    if (!f) return false;

    std::vector<double> v;
    v.reserve(9);
    double x;
    while (f >> x) v.push_back(x);
    if (v.size() != 9) {
        IR_LOG_WARN("H_gt.txt at ", p.string(), " expected 9 values, got ", v.size());
        return false;
    }
    cv::Mat H(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) H.at<double>(i / 3, i % 3) = v[i];
    H_gt = H;
    return true;
}

std::vector<Sample> DatasetLoader::load() const {
    std::vector<Sample> out;

    if (opt_.root.empty()) {
        IR_LOG_ERROR("DatasetLoader: empty root.");
        return out;
    }
    if (!fs::exists(opt_.root)) {
        IR_LOG_ERROR("DatasetLoader: root not found: ", opt_.root.string());
        return out;
    }

    std::vector<fs::path> dirs;
    if (!opt_.include.empty()) {
        for (const auto& name : opt_.include) {
            const fs::path d = opt_.root / name;
            if (fs::is_directory(d)) dirs.push_back(d);
            else IR_LOG_WARN("DatasetLoader: include entry not a directory: ", d.string());
        }
    } else {
        dirs = file_utils::listSubdirectories(opt_.root);
    }

    for (const auto& d : dirs) {
        Sample s;
        s.name = d.filename().string();
        if (!resolveImage(d, opt_.pattern_source, s.source_path)) {
            IR_LOG_WARN("DatasetLoader: missing source for ", d.string());
            continue;
        }
        if (!resolveImage(d, opt_.pattern_target, s.target_path)) {
            IR_LOG_WARN("DatasetLoader: missing target for ", d.string());
            continue;
        }
        tryLoadGroundTruth(d, s.H_gt);
        out.push_back(std::move(s));
    }

    std::sort(out.begin(), out.end(),
              [](const Sample& a, const Sample& b) { return a.name < b.name; });

    IR_LOG_INFO("DatasetLoader: loaded ", out.size(), " samples from ", opt_.root.string());
    return out;
}

} // namespace ir
