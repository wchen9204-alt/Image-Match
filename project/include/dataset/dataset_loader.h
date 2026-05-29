#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "dataset/sample.h"

namespace ir {

// ---------------------------------------------------------------------------
// DatasetLoader：扫描数据集目录，生成待配准样本列表。
// ---------------------------------------------------------------------------
class DatasetLoader {
public:
    struct Options {
        std::filesystem::path     root;
        std::string               pattern_source = "source";
        std::string               pattern_target = "target";
        std::vector<std::string>  include;       // 为空时自动发现样本。
        std::vector<std::string>  extensions = { ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff" };
    };

    explicit DatasetLoader(const Options& opt);

    // 发现所有样本；无法解析的条目会记录日志并跳过。
    std::vector<Sample> load() const;

private:
    Options opt_;

    bool resolveImage(const std::filesystem::path& dir,
                      const std::string& stem,
                      std::filesystem::path& out) const;

    bool tryLoadGroundTruth(const std::filesystem::path& dir, cv::Mat& H_gt) const;
};

} // namespace ir
