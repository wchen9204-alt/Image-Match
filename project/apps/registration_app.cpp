#include "registration_app.h"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "core/config.h"
#include "core/context.h"
#include "pipeline/feature_pipeline.h"
#include "utils/logger.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

std::string fmtMs(double v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << v << " ms";
    return oss.str();
}

void printSummary(const RegistrationContext& ctx) {
    const auto& r = ctx.result;
    std::cout << "\n================ Registration summary ================\n";
    std::cout << "  status        : " << (r.success ? "OK" : "FAILED") << "\n";
    std::cout << "  message       : " << r.message << "\n";
    std::cout << "  keypoints     : " << r.num_keypoints_first << " / "
                                       << r.num_keypoints_second << "\n";
    std::cout << "  raw matches   : " << r.num_raw_matches << "\n";
    std::cout << "  filtered      : " << r.num_filtered_matches << "\n";
    std::cout << "  inliers       : " << r.num_inliers
                                       << " (" << std::fixed << std::setprecision(3)
                                       << r.inlier_ratio << ")\n";
    std::cout << "  -- timings --\n";
    std::cout << "  load          : " << fmtMs(r.t_load_ms)     << "\n";
    std::cout << "  extract       : " << fmtMs(r.t_extract_ms)  << "\n";
    std::cout << "  match         : " << fmtMs(r.t_match_ms)    << "\n";
    std::cout << "  filter        : " << fmtMs(r.t_filter_ms)   << "\n";
    std::cout << "  geometry      : " << fmtMs(r.t_geometry_ms) << "\n";
    std::cout << "  warp          : " << fmtMs(r.t_warp_ms)     << "\n";
    std::cout << "  TOTAL         : " << fmtMs(r.t_total_ms)    << "\n";
    std::cout << "======================================================\n";
}

} // namespace

void RegistrationApp::printUsage(const std::string& exe) {
    std::cout
        << "Usage:\n"
        << "  " << exe << " <pipeline.yaml> [image1] [image2] [output_dir]\n\n"
        << "Examples:\n"
        << "  " << exe << " configs/pipeline/sift_pipeline.yaml\n"
        << "  " << exe << " configs/pipeline/orb_pipeline.yaml a.jpg b.jpg outputs\n"
        << std::endl;
}

int RegistrationApp::run(const Args& args) {
    // 1. 检查 pipeline YAML 文件是否传入并存在。
    if (args.pipeline_yaml.empty()) {
        std::cerr << "Pipeline YAML path is empty.\n";
        return 2;
    }
    if (!fs::exists(args.pipeline_yaml)) {
        std::cerr << "Pipeline YAML not found: "
                  << args.pipeline_yaml.string() << "\n";
        return 2;
    }

    // 2. 加载 pipeline 配置，这是后续创建 SIFT/ORB 等组件的入口。
    PipelineConfig cfg;
    try {
        cfg = Config::loadPipeline(args.pipeline_yaml);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load pipeline YAML: " << e.what() << "\n";
        return 3;
    }

    // 3. 如果命令行传入图片或输出目录，则覆盖 YAML 中的默认配置。
    if (!args.image1.empty()) cfg.image1_path = fs::weakly_canonical(args.image1);
    if (!args.image2.empty()) cfg.image2_path = fs::weakly_canonical(args.image2);
    if (!args.output_dir.empty()) cfg.output_dir = fs::weakly_canonical(args.output_dir);

    // 4. 检查必须的输入图片路径。
    if (cfg.image1_path.empty() || cfg.image2_path.empty()) {
        std::cerr << "Missing image1 / image2. Provide them either in the YAML "
                     "(io.image1 / io.image2) or as positional arguments.\n";
        return 4;
    }
    if (!fs::exists(cfg.image1_path)) {
        std::cerr << "image1 not found: " << cfg.image1_path.string() << "\n";
        return 5;
    }
    if (!fs::exists(cfg.image2_path)) {
        std::cerr << "image2 not found: " << cfg.image2_path.string() << "\n";
        return 5;
    }

    // 5. 创建输出目录，后续匹配图和配准结果会写到这里。
    if (!cfg.output_dir.empty()) {
        std::error_code ec;
        fs::create_directories(cfg.output_dir, ec);
    }

    // 6. 创建并配置配准管道，configure 会根据 YAML 创建具体算法组件。
    FeaturePipeline pipeline;
    if (!pipeline.configure(cfg)) {
        std::cerr << "Pipeline configure failed.\n";
        return 6;
    }

    // 7. 创建上下文对象，用于保存图片、特征、匹配、几何结果等中间数据。
    RegistrationContext ctx;

    // 8. 执行完整 pipeline；SIFT 特征提取等核心步骤会在这里触发。
    const bool ok = pipeline.run(ctx);

    // 9. 打印最终统计结果。
    printSummary(ctx);
    return ok ? 0 : 1;
}

int RegistrationApp::run(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argc > 0 ? argv[0] : "registration_app");
        return 1;
    }

    Args args;
    args.pipeline_yaml = argv[1];
    if (argc >= 3) args.image1     = argv[2];
    if (argc >= 4) args.image2     = argv[3];
    if (argc >= 5) args.output_dir = argv[4];

    return run(args);
}

} // namespace ir
