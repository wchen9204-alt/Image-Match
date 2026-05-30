#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "dataset/sample.h"

namespace ir {

/// 数据集扫描器和样本加载器。
///
/// 根据目录结构和命名规则自动发现样本，并尝试读取对应的真值矩阵。
class DatasetLoader {
public:
    /// 加载参数。
    struct Options {
        std::filesystem::path root;
        std::string pattern_source = "source";
        std::string pattern_target = "target";
        std::vector<std::string> include;
        std::vector<std::string> extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"};
    };

    /// 按给定配置创建加载器。
    explicit DatasetLoader(const Options& opt);

    /// 扫描数据集并返回所有可用样本。
    std::vector<Sample> load() const;

private:
    Options _opt;

    /// 在指定目录下尝试解析源图和目标图路径。
    bool resolveImage(const std::filesystem::path& dir,
                      const std::string& stem,
                      std::filesystem::path& out) const;

    /// 从目录中尝试读取真值单应矩阵。
    bool tryLoadGroundTruth(const std::filesystem::path& dir, cv::Mat& H_gt) const;
};

} // namespace ir
