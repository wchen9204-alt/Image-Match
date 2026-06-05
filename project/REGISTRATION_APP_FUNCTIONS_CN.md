### `std::string fmtMs(double v)`
```cpp
// 将毫秒数值格式化为统一精度的字符串，保证终端摘要的小数位一致。
// 步骤：创建 ostringstream → fixed + setprecision(2) → 追加 " ms" 后缀。
std::string fmtMs(double v);
```

### `void appendTimingSummary(std::ostringstream& oss, const RegistrationResult& r)`
```cpp
// 向输出流追加各阶段耗时表格（load / extract / match / filter / geometry / warp / total）。
// 避免点特征法和结构法维护两套耗时输出格式。
// 步骤：输出 "-- timings --" 标题 → 依次写入 7 个阶段耗时（调用 fmtMs）。
void appendTimingSummary(std::ostringstream& oss, const RegistrationResult& r);
```

### `void appendEvaluationSummary(std::ostringstream& oss, const EvaluationData& evaluation)`
```cpp
// 向输出流追加评测指标（PSNR / RMSE / SSIM 等）的名称和数值。
// 若指标无效则输出 "N/A"，若有备注则追加在括号中。
// 步骤：检查 metrics 是否为空 → 输出 "-- metrics --" 标题 → 遍历每个指标并格式化输出。
void appendEvaluationSummary(std::ostringstream& oss, const EvaluationData& evaluation);
```

### `std::string jsonEscape(const std::string& s)`
```cpp
// 对字符串中的 JSON 特殊字符（\ " \n \r \t）做转义处理。
// 步骤：遍历每个字符 → 按 switch 匹配特殊字符并追加转义序列 → 普通字符直接追加。
std::string jsonEscape(const std::string& s);
```

### `std::string buildKeypointSummaryText(const RegistrationContext& ctx)`
```cpp
// 构建点特征法的可读文本摘要，聚焦 keypoint/descriptor 匹配链路。
// 步骤：
//   1. 输出 "Keypoint registration summary" 标题
//   2. 输出 status（OK/FAILED）和 message
//   3. 输出 keypoints 数量（first / second）
//   4. 输出 raw matches → filtered → inliers（含 inlier_ratio）
//   5. 若 warp_overlap_iou >= 0，追加 IoU 行
//   6. 若 warp_photometric_error >= 0，追加 NMAD 行
//   7. 调用 appendTimingSummary 和 appendEvaluationSummary
//   8. 输出结束分隔线
std::string buildKeypointSummaryText(const RegistrationContext& ctx);
```

### `std::string buildStructureSummaryText(const RegistrationContext& ctx)`
```cpp
// 构建结构法的可读文本摘要，聚焦边缘/直线/轮廓数量及结构配准的估计结果。
// 步骤：
//   1. 输出 "Structure registration summary" 标题
//   2. 输出 status（OK/FAILED）和 message
//   3. 输出 structure type（EDGE/LINE/CONTOUR）
//   4. 输出 structures 数量（first / second）
//   5. 若几何数据 A 矩阵有效，输出 translation（dx, dy）
//   6. 输出 response（inlier_ratio）
//   7. 若 warp_overlap_iou >= 0，追加 IoU 行
//   8. 若 warp_photometric_error >= 0，追加 NMAD 行
//   9. 调用 appendTimingSummary 和 appendEvaluationSummary
//  10. 输出结束分隔线
std::string buildStructureSummaryText(const RegistrationContext& ctx);
```

### `std::string buildSummaryText(const RegistrationContext& ctx, bool structurePipeline)`
```cpp
// 根据 pipeline 类型选择对应的摘要格式。
// 步骤：判断 structurePipeline 标志 → 调用 buildStructureSummaryText 或 buildKeypointSummaryText。
std::string buildSummaryText(const RegistrationContext& ctx, bool structurePipeline);
```

### `void printSummary(const RegistrationContext& ctx, bool structurePipeline)`
```cpp
// 将 buildSummaryText 的结果直接输出到 std::cout。
// 步骤：调用 buildSummaryText → std::cout 输出。
void printSummary(const RegistrationContext& ctx, bool structurePipeline);
```

### `std::shared_ptr<IPipeline> createPipelineForConfig(const PipelineConfig& cfg)`
```cpp
// 根据配置中是否包含 structure_path 字段，自动选择 StructurePipeline 或 KeypointPipeline。
// 步骤：检查 cfg.structure_path 是否为空 → 非空创建 StructurePipeline → 否则创建 KeypointPipeline。
std::shared_ptr<IPipeline> createPipelineForConfig(const PipelineConfig& cfg);
```

### `std::string methodFamilyDir(MethodFamily family)`
```cpp
// 将 MethodFamily 枚举转换为输出目录名。
// 步骤：STRUCTURE → "structure"，KEYPOINT → "keypoint"。
std::string methodFamilyDir(MethodFamily family);
```

### `std::string sampleStemFromPaths(const fs::path& image1, const fs::path& image2)`
```cpp
// 从两个图像路径生成样本标识名（去掉扩展名的文件名，用 "__" 连接）。
// 步骤：取 image1.stem() → 拼 "__" → 拼 image2.stem()。
std::string sampleStemFromPaths(const fs::path& image1, const fs::path& image2);
```

### `fs::path normalizeOutputBaseRoot(fs::path root)`
```cpp
// 规范化输出根路径：若路径末级是 "single" 或 "batch"，则上溯到父目录。
// 保证后续 buildOutputDir 在正确的根路径上拼接 single/batch 层级，避免出现 .../single/single/...。
// 步骤：取 root.filename() → 若末级为 "single" 或 "batch" → 返回 parent_path() → 否则原样返回。
fs::path normalizeOutputBaseRoot(fs::path root);
```

### `std::string buildSummaryJson(const RegistrationContext& ctx, const PipelineConfig& cfg, const std::string& sample_name)`
```cpp
// 构建单次运行的完整 JSON 摘要。
// 步骤：
//   1. 输出 JSON 开头和 pipeline_name / method_family / sample_name / status / message
//   2. 输出 image1_path / image2_path
//   3. 输出 counts 块（num_structures 或 num_keypoints + raw_matches / filtered / inliers）
//   4. 输出 quality 块（inlier_ratio / mean_reproj_error / warp_overlap_iou）
//   5. 若为结构法且有有效 A 矩阵 → 输出 translation 块（dx, dy）
//   6. 输出 timings_ms 块（load / extract / match / filter / geometry / warp / total）
//   7. 输出 metrics 块（遍历所有评测指标，无效值输出 null）
//   8. 输出 JSON 结尾
std::string buildSummaryJson(const RegistrationContext& ctx,
                             const PipelineConfig& cfg,
                             const std::string& sample_name);
```

### `void writeRunSummaryFiles(const RegistrationContext& ctx, const PipelineConfig& cfg, const std::string& sample_name)`
```cpp
// 写入单次运行的文本摘要和 JSON 摘要到输出目录。
// 步骤：
//   1. 检查 output_dir 是否为空 → 空则直接返回
//   2. 调用 buildSummaryText / buildSummaryJson 生成内容
//   3. 写入 run_summary.txt
//   4. 写入 run_summary.json
void writeRunSummaryFiles(const RegistrationContext& ctx,
                          const PipelineConfig& cfg,
                          const std::string& sample_name);
```

### `std::vector<std::string> collectMetricColumns(const std::vector<EvaluationData>& evaluations)`
```cpp
// 从所有 EvaluationData 中收集唯一的指标名称列表，用于 CSV 表头生成。
// 步骤：遍历所有 evaluations → 遍历每个 evaluation 的 metrics → 去重加入 columns。
std::vector<std::string> collectMetricColumns(const std::vector<EvaluationData>& evaluations);
```

### `static void RegistrationApp::printUsage(const std::string& exe)`
```cpp
// 输出命令行用法说明（Usage / Examples）。
// 步骤：输出单次运行格式 → 输出批处理格式 → 输出使用示例。
static void printUsage(const std::string& exe);
```

### `static int RegistrationApp::runSingle(const Args& args)`
```cpp
// 执行单次配准（给定一个 pipeline YAML 和一对可选图像路径）。
// 步骤：
//   1. 校验 pipeline YAML 路径存在性
//   2. 调用 Config::loadPipeline 加载 pipeline 配置（含子配置路径）
//   3. 命令行传入的 image1/image2/output_dir 覆盖 YAML 中的值
//   4. 校验 image1 / image2 路径存在性
//   5. 调用 normalizeOutputBaseRoot + buildOutputDir 生成最终输出目录
//   6. 调用 createPipelineForConfig 创建 KeypointPipeline 或 StructurePipeline
//   7. 调用 pipeline->configure(cfg) 创建所有算法组件
//   8. 调用 pipeline->run(ctx) 执行完整配准流程
//   9. 写入摘要文件（writeRunSummaryFiles + writeSummaryCsv）
//  10. 输出终端摘要（printSummary）
//  11. 返回 0（成功）或非 0（失败）
static int runSingle(const Args& args);
```

### `static bool RegistrationApp::isBatchYaml(const YAML::Node& node)`
```cpp
// 判断 YAML 根节点是否为批处理配置（同时包含 pipeline 和 dataset 键）。
// 步骤：检查 node 是否为 Map → 检查是否包含 "pipeline" 和 "dataset" 键。
static bool isBatchYaml(const YAML::Node& node);
```

### `static RegistrationApp::BatchConfig RegistrationApp::loadBatchConfig(const std::filesystem::path& yaml_path)`
```cpp
// 加载批处理配置文件，解析 pipeline 引用、数据集参数和输出设置。
// 步骤：
//   1. 调用 Config::load 读取 YAML
//   2. 读取 name（默认取文件名 stem）
//   3. 解析 pipeline 路径（相对 batch.yaml 所在目录）
//   4. 解析 dataset：root / pattern_source / pattern_target / include
//   5. 解析 output：root / save_visuals / summary_csv
static BatchConfig loadBatchConfig(const std::filesystem::path& yaml_path);
```

### `static std::filesystem::path RegistrationApp::resolveBatchOutputRoot(const BatchConfig& batch, const PipelineConfig& pipeline_cfg)`
```cpp
// 规范化批处理输出根目录（去掉可能存在的 single/batch 末级目录）。
// 步骤：检查 batch.output_root 是否为空 → 调用 normalizeOutputBaseRoot 规范化。
static std::filesystem::path resolveBatchOutputRoot(const BatchConfig& batch,
                                                     const PipelineConfig& pipeline_cfg);
```

### `static std::filesystem::path RegistrationApp::buildOutputDir(OutputMode mode, const std::filesystem::path& base_root, const PipelineConfig& cfg, const std::string& sample_name)`
```cpp
// 按统一规则拼接输出目录路径：{base_root}/{single|batch}/{keypoint|structure}/{pipeline_name}/{sample_name}
// 步骤：检查 base_root 是否为空 → 根据 mode 取 "single" 或 "batch" → 追加 methodFamilyDir → pipeline name → sample_name。
static std::filesystem::path buildOutputDir(OutputMode mode,
                                             const std::filesystem::path& base_root,
                                             const PipelineConfig& cfg,
                                             const std::string& sample_name);
```

### `static void RegistrationApp::writeSummaryCsv(...)`
```cpp
// 将批量配准结果写入汇总 CSV 文件。
// 步骤：
//   1. 调用 collectMetricColumns 统计所有指标列名
//   2. 写 CSV 表头：sample_name, success, message, counts, raw/filtered/inlier,
//      inlier_ratio, reproj_error, warp_iou, warp_photometric_error, 各阶段耗时, 评测指标
//   3. 遍历每个样本写一行数据（success → "1" 或 "0"，字段用逗号分隔）
//   4. 调用 file_utils::writeWholeFile 写入磁盘
static void writeSummaryCsv(const std::filesystem::path& csv_path,
                            MethodFamily family,
                            const std::vector<std::string>& sample_names,
                            const std::vector<RegistrationResult>& results,
                            const std::vector<EvaluationData>& evaluations);
```

### `static int RegistrationApp::runBatch(const std::filesystem::path& batch_yaml)`
```cpp
// 执行批量配准：加载批处理配置 → 扫描数据集 → 遍历所有样本逐次配准 → 输出汇总。
// 步骤：
//   1. 调用 loadBatchConfig 加载批处理配置
//   2. 调用 Config::loadPipeline 加载 pipeline 模板（复用为所有样本的基础配置）
//   3. 调用 resolveBatchOutputRoot 规范化输出根目录
//   4. 调用 DatasetLoader::load() 扫描并加载所有样本
//   5. 创建 pipeline_root 输出目录
//   6. 遍历每个样本：
//      a. 从 base_cfg 复制配置，替换 image1 / image2 / output_dir
//      b. 若 batch 配置关闭 save_visuals → 关闭 draw_matches 和 warp
//      c. 调用 createPipelineForConfig + configure 创建并配置 pipeline
//      d. 配置失败 → 记录失败结果，跳过该样本，继续下一个
//      e. 调用 pipeline->run(ctx) 执行完整配准
//      f. 写入该样本的摘要文件（writeRunSummaryFiles）
//      g. 收集 result / evaluation 到汇总容器
//   7. 若 batch.summary_csv 为 true → 调用 writeSummaryCsv 输出总表
//   8. 输出控制台汇总："Batch summary: N / M samples succeeded." + 成功样本列表
//   9. 返回 0（全部成功）或 1（存在失败）
static int runBatch(const std::filesystem::path& batch_yaml);
```

### `static int RegistrationApp::run(const Args& args)`
```cpp
// 顶层路由：根据 YAML 内容自动分发到 runSingle 或 runBatch。
// 步骤：
//   1. 校验 pipeline_yaml 非空且存在
//   2. 尝试加载 YAML（Config::load）
//   3. 调用 isBatchYaml 判断 → 是批处理则调用 runBatch
//   4. 否则调用 runSingle
static int run(const Args& args);
```

### `static int RegistrationApp::run(int argc, char** argv)`
```cpp
// CLI 入口：解析命令行参数为 Args 结构体，然后委托给 run(Args)。
// 步骤：
//   1. 检查 argc ≥ 2（至少需要 pipeline YAML 路径）
//   2. 解析位置参数：argv[1]=pipeline_yaml, argv[2]=image1, argv[3]=image2, argv[4]=output_dir
//   3. 调用 run(args) 路由到单次或批处理
//   4. 返回退出码
static int run(int argc, char** argv);
```

### 调用关系图

```
run(argc, argv)                          ← CLI 入口
  └─ run(args)                           ← 路由分发
       ├─ [batch] runBatch(batch_yaml)
       │    ├─ loadBatchConfig()         ← 解析 batch YAML
       │    ├─ resolveBatchOutputRoot()   ← 规范化输出路径
       │    ├─ DatasetLoader::load()      ← 扫描数据集
       │    └─ for each sample:
       │         ├─ createPipelineForConfig()
       │         ├─ pipeline->run(ctx)
       │         ├─ writeRunSummaryFiles()
       │         │    ├─ buildSummaryText()
       │         │    │    └─ buildKeypointSummaryText() 或 buildStructureSummaryText()
       │         │    │         ├─ appendTimingSummary()
       │         │    │         └─ appendEvaluationSummary()
       │         │    └─ buildSummaryJson()
       │         │         └─ jsonEscape()
       │         └─ printSummary()
       │    └─ writeSummaryCsv()
       │         └─ collectMetricColumns()
       │
       └─ [single] runSingle(args)
            ├─ createPipelineForConfig()
            ├─ buildOutputDir()
            │    ├─ normalizeOutputBaseRoot()
            │    ├─ methodFamilyDir()
            │    └─ sampleStemFromPaths()
            ├─ pipeline->run(ctx)
            ├─ writeRunSummaryFiles()
            ├─ writeSummaryCsv()
            └─ printSummary()

工具函数：
  fmtMs()                    ← 格式化毫秒耗时
  methodFamilyDir()          ← 枚举 → 目录名字符串
  sampleStemFromPaths()      ← 路径 → 样本标识名
  normalizeOutputBaseRoot()  ← 输出根路径规范化
  isBatchYaml()              ← YAML 类型判断
  jsonEscape()               ← JSON 字符串转义
  collectMetricColumns()     ← 收集唯一指标列名
```
