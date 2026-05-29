# SIFT执行流程 - 详细代码追踪

## 📍 从命令行到SIFT提取的完整追踪

### 🔴 步骤1: 程序入口 (main.cpp)
```cpp
// project/main.cpp
#include "apps/registration_app.h"

int main(int argc, char** argv) {
    // argc = 2
    // argv[0] = "registration_app"
    // argv[1] = "configs/pipeline/sift_pipeline.yaml"
    // argv[2] = "datasets/test1/source.png"  (可选)
    // argv[3] = "datasets/test1/target.png"  (可选)
    return ir::RegistrationApp::run(argc, argv);  // 调用APP入口
}
```

### 🔴 步骤2: 解析命令行参数 (registration_app.cpp)
```cpp
// project/apps/registration_app.cpp

int RegistrationApp::run(int argc, char** argv) {
    // 步骤2.1: 检查参数数量
    if (argc < 2) {
        printUsage(argc > 0 ? argv[0] : "registration_app");
        return 1;  // 参数不足
    }

    // 步骤2.2: 构建Args结构
    Args args;
    args.pipeline_yaml = argv[1];         // "configs/pipeline/sift_pipeline.yaml"
    if (argc >= 3) args.image1     = argv[2];
    if (argc >= 4) args.image2     = argv[3];
    if (argc >= 5) args.output_dir = argv[4];

    // 步骤2.3: 调用Run实现
    return run(args);
}

int RegistrationApp::run(const Args& args) {
    // 步骤2.4: 检查pipeline YAML文件
    if (args.pipeline_yaml.empty()) {
        std::cerr << "Pipeline YAML path is empty.\n";
        return 2;
    }
    if (!fs::exists(args.pipeline_yaml)) {
        std::cerr << "Pipeline YAML not found: " << args.pipeline_yaml.string() << "\n";
        return 2;
    }

    // 步骤2.5: 加载管道配置 ⭐ 这是关键！
    PipelineConfig cfg;
    try {
        cfg = Config::loadPipeline(args.pipeline_yaml);  // 加载sift_pipeline.yaml
    } catch (const std::exception& e) {
        std::cerr << "Failed to load pipeline YAML: " << e.what() << "\n";
        return 3;
    }

    // 步骤2.6: 覆盖命令行参数（如果有的话）
    if (!args.image1.empty()) cfg.image1_path = fs::weakly_canonical(args.image1);
    if (!args.image2.empty()) cfg.image2_path = fs::weakly_canonical(args.image2);
    if (!args.output_dir.empty()) cfg.output_dir = fs::weakly_canonical(args.output_dir);

    // 步骤2.7: 检查必须的图像路径
    if (cfg.image1_path.empty() || cfg.image2_path.empty()) {
        std::cerr << "Missing image1 / image2\n";
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

    // 步骤2.8: 创建输出目录
    if (!cfg.output_dir.empty()) {
        std::error_code ec;
        fs::create_directories(cfg.output_dir, ec);
    }

    // 步骤2.9: 创建管道对象
    FeaturePipeline pipeline;  // 继承自BasePipeline

    // 步骤2.10: 配置管道 ⭐ 这里会创建SIFT提取器
    if (!pipeline.configure(cfg)) {
        std::cerr << "Pipeline configure failed.\n";
        return 6;
    }

    // 步骤2.11: 创建上下文对象（容纳所有中间结果）
    RegistrationContext ctx;

    // 步骤2.12: 执行整个管道 ⭐⭐⭐ SIFT在这里执行
    const bool ok = pipeline.run(ctx);

    // 步骤2.13: 打印最终结果
    printSummary(ctx);

    return ok ? 0 : 1;
}
```

### 🔴 步骤3: 配置解析与组件创建 (config.h)
```cpp
// project/include/core/config.h

PipelineConfig Config::loadPipeline(const fs::path& path) {
    // 步骤3.1: 加载YAML文件
    YAML::Node node = load(path);  // 加载 sift_pipeline.yaml
    const fs::path base = path.parent_path();

    PipelineConfig cfg;
    cfg.name = yaml_utils::getString(node, "name", "sift_bf_homography");

    // 步骤3.2: 解析所有子配置路径
    cfg.feature_path  = resolvePath(base, yaml_utils::getString(node, "feature"));
    // 结果: "configs/feature/sift.yaml"
    
    cfg.matcher_path  = resolvePath(base, yaml_utils::getString(node, "matcher"));
    // 结果: "configs/matcher/bf.yaml"

    cfg.geometry_path = resolvePath(base, yaml_utils::getString(node, "geometry"));
    // 结果: "configs/geometry/homography.yaml"

    cfg.filter_paths.clear();
    if (node["filters"] && node["filters"].IsSequence()) {
        for (const auto& f : node["filters"]) {
            cfg.filter_paths.push_back(resolvePath(base, f.as<std::string>()));
        }
    }
    // 结果: ["configs/filter/ratio_test.yaml", "configs/filter/cross_check.yaml"]

    // 步骤3.3: 解析I/O配置
    if (node["io"] && node["io"].IsMap()) {
        const auto& io = node["io"];
        cfg.image1_path = resolvePath(base, yaml_utils::getString(io, "image1"));
        cfg.image2_path = resolvePath(base, yaml_utils::getString(io, "image2"));
        cfg.output_dir  = resolvePath(base, yaml_utils::getString(io, "output_dir", "outputs"));
    }

    // 步骤3.4: 解析可视化配置
    if (node["visualization"] && node["visualization"].IsMap()) {
        const auto& vis = node["visualization"];
        cfg.draw_matches      = yaml_utils::getBool(vis, "draw_matches",       true);
        cfg.draw_inliers_only = yaml_utils::getBool(vis, "draw_inliers_only",  true);
        cfg.max_matches_drawn = yaml_utils::getInt(vis, "max_matches_drawn",   100);
        cfg.warp              = yaml_utils::getBool(vis, "warp",               true);
    }

    // 步骤3.5: 输出日志
    IR_LOG_INFO("Pipeline '", cfg.name, "' loaded from ", path.string());
    IR_LOG_INFO("  feature  : ", cfg.feature_path.string());      // sift.yaml
    IR_LOG_INFO("  matcher  : ", cfg.matcher_path.string());      // bf.yaml
    for (const auto& f : cfg.filter_paths) {
        IR_LOG_INFO("  filter   : ", f.string());                 // ratio_test.yaml, cross_check.yaml
    }
    IR_LOG_INFO("  geometry : ", cfg.geometry_path.string());    // homography.yaml
    IR_LOG_INFO("  image1   : ", cfg.image1_path.string());
    IR_LOG_INFO("  image2   : ", cfg.image2_path.string());
    IR_LOG_INFO("  output   : ", cfg.output_dir.string());

    return cfg;
}
```

### 🔴 步骤4: 管道配置与组件创建 (base_pipeline.cpp)
```cpp
// project/src/pipeline/base_pipeline.cpp

bool BasePipeline::configure(const PipelineConfig& cfg) {
    config_ = cfg;

    // 步骤4.1: 创建特征提取器 ⭐ SIFT在这里被创建
    {
        YAML::Node feature_cfg = Config::load(cfg.feature_path);  // 加载sift.yaml
        extractor_ = Factory::createFeatureExtractor(feature_cfg);
        if (!extractor_) {
            IR_LOG_ERROR("Failed to create feature extractor.");
            return false;
        }
        IR_LOG_INFO("Feature extractor: ", extractor_->name());
        // 输出: "Feature extractor: SIFT"
    }

    // 步骤4.2: 创建匹配器
    {
        YAML::Node matcher_cfg = Config::load(cfg.matcher_path);  // 加载bf.yaml
        matcher_ = Factory::createMatcher(matcher_cfg);
        if (!matcher_) {
            IR_LOG_ERROR("Failed to create matcher.");
            return false;
        }
        IR_LOG_INFO("Matcher: ", matcher_->name());
        // 输出: "Matcher: BF"
    }

    // 步骤4.3: 创建滤波器链
    filters_.clear();
    for (const auto& filter_path : cfg.filter_paths) {
        YAML::Node filter_cfg = Config::load(filter_path);
        auto filter = Factory::createFilter(filter_cfg);
        if (filter) {
            filters_.push_back(filter);
            IR_LOG_INFO("Filter: ", filter->name());
        }
    }
    // 输出:
    // "Filter: RatioTest"
    // "Filter: CrossCheck"

    // 步骤4.4: 创建几何估计器
    {
        YAML::Node geom_cfg = Config::load(cfg.geometry_path);  // 加载homography.yaml
        geometry_ = Factory::createGeometryEstimator(geom_cfg);
        if (!geometry_) {
            IR_LOG_ERROR("Failed to create geometry estimator.");
            return false;
        }
        IR_LOG_INFO("Geometry estimator: ", geometry_->name());
        // 输出: "Geometry estimator: Homography"
    }

    // 步骤4.5: 创建变形器
    {
        warper_ = std::make_shared<PerspectiveWarper>();
        IR_LOG_INFO("Warper: Perspective");
    }

    IR_LOG_INFO("Pipeline configured successfully.");
    return true;
}
```

### 🔴 步骤5: 执行整个管道 (base_pipeline.cpp)
```cpp
// project/src/pipeline/base_pipeline.cpp

bool BasePipeline::run(RegistrationContext& ctx) {
    Timer t;  // 开始计时
    ctx.reset();
    ctx.image1_path = config_.image1_path;
    ctx.image2_path = config_.image2_path;
    ctx.output_dir = config_.output_dir;

    // ========== 阶段1: 加载图像 ==========
    {
        Timer stage_timer;
        if (!loadImages(ctx)) {
            ctx.result.success = false;
            ctx.result.message = "Failed to load images.";
            return false;
        }
        ctx.result.t_load_ms = stage_timer.elapsedMs();
        IR_LOG_INFO("Loaded images in ", ctx.result.t_load_ms, " ms");
    }

    // ========== 阶段2: SIFT特征提取 ⭐⭐⭐ ==========
    {
        Timer stage_timer;
        if (!runExtract(ctx)) {  // 调用提取器
            ctx.result.success = false;
            ctx.result.message = "Feature extraction failed.";
            return false;
        }
        ctx.result.t_extract_ms = stage_timer.elapsedMs();
        ctx.result.num_keypoints_first = ctx.feature_data.first.keypoints.size();
        ctx.result.num_keypoints_second = ctx.feature_data.second.keypoints.size();
        IR_LOG_INFO("Extraction in ", ctx.result.t_extract_ms, " ms");
        // 这里SIFT算法已执行！特征已提取！
    }

    // ========== 阶段3: 匹配 ==========
    {
        Timer stage_timer;
        if (!runMatch(ctx)) {
            ctx.result.success = false;
            ctx.result.message = "Matching failed.";
            return false;
        }
        ctx.result.t_match_ms = stage_timer.elapsedMs();
        ctx.result.num_raw_matches = 0;
        for (const auto& nb : ctx.match_data.raw_knn) {
            ctx.result.num_raw_matches += !nb.empty() ? 1 : 0;
        }
        IR_LOG_INFO("Matched in ", ctx.result.t_match_ms, " ms");
    }

    // ========== 阶段4: 滤波链 ==========
    {
        Timer stage_timer;
        if (!runFilters(ctx)) {
            ctx.result.success = false;
            ctx.result.message = "Filtering failed.";
            return false;
        }
        ctx.result.t_filter_ms = stage_timer.elapsedMs();
        ctx.result.num_filtered_matches = ctx.match_data.filtered.size();
        IR_LOG_INFO("Filtered in ", ctx.result.t_filter_ms, " ms");
    }

    // ========== 阶段5: 几何估计 ==========
    {
        Timer stage_timer;
        if (!runGeometry(ctx)) {
            ctx.result.success = false;
            ctx.result.message = "Geometry estimation failed.";
            return false;
        }
        ctx.result.t_geometry_ms = stage_timer.elapsedMs();
        ctx.result.num_inliers = ctx.geometry_data.num_inliers;
        ctx.result.inlier_ratio = ctx.geometry_data.inlier_ratio;
        IR_LOG_INFO("Geometry in ", ctx.result.t_geometry_ms, " ms");
    }

    // ========== 阶段6: 图像变形 ==========
    {
        Timer stage_timer;
        if (!runWarp(ctx)) {
            IR_LOG_WARN("Warping failed, but continuing...");
        }
        ctx.result.t_warp_ms = stage_timer.elapsedMs();
        IR_LOG_INFO("Warped in ", ctx.result.t_warp_ms, " ms");
    }

    // ========== 阶段7: 保存结果 ==========
    {
        Timer stage_timer;
        saveOutputs(ctx);
        IR_LOG_INFO("Saved outputs in ", stage_timer.elapsedMs(), " ms");
    }

    ctx.result.t_total_ms = t.elapsedMs();
    ctx.result.success = true;
    ctx.result.message = "Registration successful.";

    return true;
}

// 关键函数: 阶段2 - 特征提取
bool BasePipeline::runExtract(RegistrationContext& ctx) {
    if (!extractor_) {
        IR_LOG_ERROR("Feature extractor not configured.");
        return false;
    }
    // ⭐⭐⭐ 调用SIFT提取器的extract()方法
    return extractor_->extract(ctx);
}
```

### 🔴 步骤6: SIFT特征提取的核心实现 ⭐⭐⭐
```cpp
// project/src/feature/sift_extractor.cpp

// 构造函数: 从sift.yaml读取参数并创建SIFT对象
SiftExtractor::SiftExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    // 步骤6.1: 从YAML读取SIFT参数
    nfeatures_         = yaml_utils::getInt   (params, "nfeatures",         0);
    nOctaveLayers_     = yaml_utils::getInt   (params, "nOctaveLayers",     3);
    contrastThreshold_ = yaml_utils::getDouble(params, "contrastThreshold", 0.04);
    edgeThreshold_     = yaml_utils::getDouble(params, "edgeThreshold",     10.0);
    sigma_             = yaml_utils::getDouble(params, "sigma",             1.6);

    // 步骤6.2: 创建OpenCV的SIFT对象
    impl_ = cv::SIFT::create(
        nfeatures_,           // 0 = 保留所有特征
        nOctaveLayers_,       // 3 = 每个八度3层
        contrastThreshold_,   // 0.04 = 对比度阈值（筛选弱特征）
        edgeThreshold_,       // 10.0 = 边界阈值（筛选边界特征）
        sigma_);              // 1.6 = 初始高斯模糊

    // 步骤6.3: 输出日志
    IR_LOG_INFO("SIFT created: nfeatures=",      nfeatures_,
                ", nOctaveLayers=",              nOctaveLayers_,
                ", contrast=",                   contrastThreshold_,
                ", edge=",                       edgeThreshold_,
                ", sigma=",                      sigma_);
}

// 主函数: 执行SIFT特征提取
bool SiftExtractor::extract(RegistrationContext& ctx) {
    // 步骤6.4: 检查SIFT对象
    if (!impl_) {
        IR_LOG_ERROR("SIFT extractor not constructed.");
        return false;
    }

    // 步骤6.5: 获取特征数据结构
    auto& fd = ctx.feature_data;
    fd.type      = FeatureType::SIFT;      // 标记为SIFT特征
    fd.norm_type = NormType::L2;           // SIFT使用L2距离

    // 步骤6.6: 检查输入图像
    if (fd.first.image.empty() || fd.second.image.empty()) {
        IR_LOG_ERROR("SIFT::extract - source images are empty.");
        return false;
    }

    // 步骤6.7: 转换为灰度图（如果需要）
    if (fd.first.gray.empty()) {
        cv::cvtColor(fd.first.image, fd.first.gray, cv::COLOR_BGR2GRAY);
    }
    if (fd.second.gray.empty()) {
        cv::cvtColor(fd.second.image, fd.second.gray, cv::COLOR_BGR2GRAY);
    }

    // ⭐⭐⭐ 步骤6.8: 执行SIFT检测和计算
    // 这是SIFT算法的核心计算！
    impl_->detectAndCompute(
        fd.first.gray,              // 输入: 图像1灰度图
        cv::noArray(),              // 掩码: 无
        fd.first.keypoints,         // 输出: 关键点位置 (vector<KeyPoint>)
        fd.first.descriptors);      // 输出: 128维浮点描述子 (CV_32F)

    impl_->detectAndCompute(
        fd.second.gray,             // 输入: 图像2灰度图
        cv::noArray(),              // 掩码: 无
        fd.second.keypoints,        // 输出: 关键点位置
        fd.second.descriptors);     // 输出: 128维浮点描述子

    // 步骤6.9: 输出统计日志
    IR_LOG_INFO("SIFT extracted ",
                fd.first.keypoints.size(),   // 图像1的关键点数
                " / ",
                fd.second.keypoints.size(),  // 图像2的关键点数
                " keypoints");

    // 步骤6.10: 返回成功（如果任何一个图像的特征为空则失败）
    return !fd.empty();
}
```

### 🔴 步骤7: 之后的匹配和滤波阶段

```cpp
// ========== 阶段3: BFMatcher匹配 ==========
// project/src/matcher/bf_matcher.cpp

bool BFMatcher::match(RegistrationContext& ctx) {
    auto& md = ctx.match_data;
    auto& fd = ctx.feature_data;

    // 步骤7.1: 检查SIFT提取的结果
    if (fd.first.descriptors.empty() || fd.second.descriptors.empty()) {
        IR_LOG_WARN("BFMatcher: no descriptors available.");
        return false;
    }

    // 步骤7.2: 创建暴力匹配器（如果需要）
    if (!impl_) {
        NormType norm = (norm_type_ == NormType::AUTO) ? fd.norm_type : norm_type_;
        impl_ = cv::BFMatcher::create(toCvNorm(norm), false);
    }

    // ⭐ 步骤7.3: 执行K最近邻匹配（k=2）
    // 输入: SIFT的128维描述子
    // 输出: 每个点的2个最近邻
    impl_->knnMatch(
        fd.first.descriptors,      // 查询描述子 (N×128 CV_32F)
        fd.second.descriptors,     // 训练描述子 (M×128 CV_32F)
        md.raw_knn,                // 输出: N个查询点，每个有2个匹配
        knn_k_);                   // k=2

    IR_LOG_INFO("Raw matches (k=2): ", md.raw_knn.size());
    return true;
}

// ========== 阶段4a: 比值测试滤波 ==========
// project/src/filter/ratio_test.cpp

bool RatioTestFilter::apply(RegistrationContext& ctx) {
    auto& md = ctx.match_data;

    if (md.raw_knn.empty()) {
        IR_LOG_WARN("RatioTest: no k-NN matches.");
        return false;
    }

    std::vector<cv::DMatch> kept;
    kept.reserve(md.raw_knn.size());

    // ⭐ Lowe's ratio test: distance[0] / distance[1] < 0.75
    for (const auto& nb : md.raw_knn) {
        if (nb.size() >= 2) {  // 确保有2个邻近
            double ratio = nb[0].distance / nb[1].distance;
            if (ratio < ratio_) {  // 0.75 (从yaml读取)
                kept.push_back(nb[0]);  // 只保留最佳匹配，丢弃次近
            }
        }
    }

    IR_LOG_INFO("RatioTest kept ", kept.size(), " / ", md.raw_knn.size(), " matches");
    md.filtered = std::move(kept);
    return true;
}

// ========== 阶段4b: 交叉检验滤波 ==========
// project/src/filter/cross_check.cpp

bool CrossCheckFilter::apply(RegistrationContext& ctx) {
    auto& fd = ctx.feature_data;
    auto& md = ctx.match_data;

    // 步骤: 获取前向匹配（图1→图2）
    std::vector<cv::DMatch> forward = md.filtered;

    // 步骤: 反向匹配（图2→图1）
    NormType norm = fd.norm_type;
    cv::Ptr<cv::BFMatcher> rev = cv::BFMatcher::create(toCvNorm(norm), false);

    std::vector<cv::DMatch> reverse;
    rev->match(
        fd.second.descriptors,      // 查询: 图2
        fd.first.descriptors,       // 训练: 图1
        reverse);                   // 输出

    // ⭐ 步骤: 检查双向一致性
    // 图1→图2 的最佳匹配必须等于 图2→图1 的最佳匹配
    std::vector<cv::DMatch> kept;
    for (const auto& m : forward) {
        // m.queryIdx -> 图1, m.trainIdx -> 图2
        // 检查图2[m.trainIdx] 的最佳匹配是否是 图1[m.queryIdx]
        for (const auto& r : reverse) {
            if (r.queryIdx == m.trainIdx && r.trainIdx == m.queryIdx) {
                kept.push_back(m);
                break;
            }
        }
    }

    IR_LOG_INFO("CrossCheck kept ", kept.size(), " / ", forward.size(), " matches");
    md.filtered = std::move(kept);
    return true;
}
```

### 🔴 步骤8: 几何估计与最终结果

```cpp
// ========== 阶段5: 单应矩阵估计 ==========
// project/src/geometry/homography_estimator.cpp

bool HomographyEstimator::estimate(RegistrationContext& ctx) {
    auto& fd = ctx.feature_data;
    auto& md = ctx.match_data;
    auto& gd = ctx.geometry_data;

    if (md.filtered.empty()) {
        IR_LOG_WARN("HomographyEstimator: no matches.");
        return false;
    }

    // ⭐ 步骤: 从匹配的关键点坐标构建点集
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : md.filtered) {
        pts1.push_back(fd.first.keypoints[m.queryIdx].pt);
        pts2.push_back(fd.second.keypoints[m.trainIdx].pt);
    }

    // ⭐⭐ 步骤: 使用RANSAC估计单应矩阵
    // 输入: SIFT匹配后的关键点坐标
    // 输出: 3×3变换矩阵 H
    std::vector<unsigned char> inlier_mask;
    gd.H = cv::findHomography(
        pts1,                       // 图像1的点
        pts2,                       // 图像2的点
        cv::RANSAC,                 // RANSAC算法
        inlier_threshold_,          // 内点阈值
        inlier_mask,                // 输出: 哪些点是内点
        2000,                       // RANSAC最大迭代数
        confidence_);               // 置信度

    // ⭐ 步骤: 统计内点
    gd.num_inliers = cv::countNonZero(inlier_mask);
    gd.inlier_ratio = (double)gd.num_inliers / md.filtered.size();
    md.inlier_mask = inlier_mask;

    // 步骤: 提取内点匹配
    md.inliers.clear();
    for (size_t i = 0; i < md.filtered.size(); ++i) {
        if (inlier_mask[i]) {
            md.inliers.push_back(md.filtered[i]);
        }
    }

    IR_LOG_INFO("Homography: ", gd.num_inliers, " inliers / ", md.filtered.size(),
                " (ratio: ", std::fixed << std::setprecision(3) << gd.inlier_ratio, ")");

    gd.type = GeometryType::HOMOGRAPHY;
    gd.valid = !gd.H.empty();
    return gd.valid;
}
```

---

## 📊 SIFT执行流程的数据转换示例

假设两张图像各有5000个关键点：

```
步骤1: 加载图像
  fd.first.image (1080×1920×3 BGR) → 已加载
  fd.second.image (1080×1920×3 BGR) → 已加载

步骤2: SIFT提取
  fd.first.image (BGR) → 转灰度 → detectAndCompute()
    输出: fd.first.keypoints (5000个 cv::KeyPoint)
          fd.first.descriptors (5000×128 CV_32F)
  
  fd.second.image (BGR) → 转灰度 → detectAndCompute()
    输出: fd.second.keypoints (4800个 cv::KeyPoint)
          fd.second.descriptors (4800×128 CV_32F)

步骤3: 匹配
  raw_knn = knnMatch(5000×128, 4800×128)
    输出: 5000个查询点，每个有2个最近邻
    总共: ~10000个候选匹配

步骤4a: 比值测试
  检查每个匹配的 distance[0]/distance[1] < 0.75
    通过: ~3500个匹配
    原因: 好的匹配最近邻明显比次近更近

步骤4b: 交叉检验
  检查图1→图2 的最佳邻近是否等于 图2→图1 的最佳邻近
    通过: ~3000个匹配
    原因: 坏的匹配不会双向一致

步骤5: 几何估计 (RANSAC)
  pts1 = 3000个来自图1的关键点坐标
  pts2 = 3000个来自图2的关键点坐标
  
  findHomography(pts1, pts2, RANSAC)
    内点: ~2500个 (83%内点率)
    外点: ~500个 (被视为误匹配)
    输出: H = 3×3矩阵

步骤6: 变形
  warpPerspective(image2, H)
    输出: 变形后的图像

步骤7: 保存
  visualization_manager: 绘制所有过程
  输出: matches/, inliers/, overlay/, warped/ 等
```

---

## 🎯 关键函数速查

| 函数 | 文件 | 作用 |
|------|------|------|
| `main()` | main.cpp | 程序入口 |
| `RegistrationApp::run()` | registration_app.cpp | 解析参数，调用管道 |
| `Config::loadPipeline()` | config.cpp | 加载并解析sift_pipeline.yaml |
| `BasePipeline::configure()` | base_pipeline.cpp | 创建所有组件（包括SIFT） |
| `BasePipeline::run()` | base_pipeline.cpp | 执行7个阶段 |
| `SiftExtractor::extract()` | sift_extractor.cpp | ⭐⭐⭐ SIFT提取关键步骤 |
| `BFMatcher::match()` | bf_matcher.cpp | 匹配SIFT描述子 |
| `RatioTestFilter::apply()` | ratio_test.cpp | Lowe's比值测试 |
| `CrossCheckFilter::apply()` | cross_check.cpp | 双向一致性检验 |
| `HomographyEstimator::estimate()` | homography_estimator.cpp | RANSAC估计H矩阵 |

---

## 📝 查看执行日志

运行命令后，查看日志输出：
```bash
cd project
./build/bin/registration_app configs/pipeline/sift_pipeline.yaml

# 输出示例:
# [INFO] Pipeline 'sift_bf_homography' loaded from ...
# [INFO] Feature extractor: SIFT
# [INFO] Matcher: BF
# [INFO] Filter: RatioTest
# [INFO] Filter: CrossCheck
# [INFO] Geometry estimator: Homography
# [INFO] SIFT created: nfeatures=0, nOctaveLayers=3, contrast=0.04, edge=10, sigma=1.6
# [INFO] Loaded images in 45.23 ms
# [INFO] SIFT extracted 5000 / 4800 keypoints
# [INFO] Extraction in 1234.56 ms
# [INFO] Raw matches (k=2): 5000
# [INFO] Matched in 45.67 ms
# [INFO] RatioTest kept 3500 / 5000 matches
# [INFO] CrossCheck kept 3000 / 3500 matches
# [INFO] Filtered in 12.34 ms
# [INFO] Homography: 2500 inliers / 3000 (ratio: 0.833)
# [INFO] Geometry in 56.78 ms
# [INFO] Warped in 123.45 ms
# [INFO] Saved outputs in 234.56 ms
#
# ================ Registration summary ================
#   status        : OK
#   keypoints     : 5000 / 4800
#   raw matches   : 5000
#   filtered      : 3000
#   inliers       : 2500
#   -- timings --
#   extract       : 1234.56 ms
#   match         : 45.67 ms
#   filter        : 12.34 ms
#   geometry      : 56.78 ms
# ======================================================
```
