#pragma once

#include <filesystem>
#include <string>

namespace ir {

// ---------------------------------------------------------------------------
// RegistrationApp：基于 FeaturePipeline 的命令行入口。
//
// 用法：
//     registration_app <pipeline.yaml> [image1] [image2] [output_dir]
//
// 后三个参数可选；未传入时使用 pipeline YAML 中的配置。
// ---------------------------------------------------------------------------
class RegistrationApp {
public:
    struct Args {
        std::filesystem::path pipeline_yaml;
        std::filesystem::path image1;
        std::filesystem::path image2;
        std::filesystem::path output_dir;
    };

    static int run(const Args& args);
    static int run(int argc, char** argv);

    static void printUsage(const std::string& exe);
};

} // namespace ir
