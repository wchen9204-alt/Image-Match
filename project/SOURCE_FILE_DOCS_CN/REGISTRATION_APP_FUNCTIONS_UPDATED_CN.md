# `registration_app` 当前实现笔记（修正版）

本文档基于当前工程中的以下文件整理，而不是旧版粘贴内容：

- `project/apps/registration_app.h`
- `project/apps/registration_app.cpp`

说明：

- 你提到的“`registration.cpp`”在当前工程里对应实现文件实际是 `apps/registration_app.cpp`。
- 这份文档只修正 `registration_app` 相关内容，不包含 contour 新模块的额外笔记。
- 文档风格沿用“函数职责 + 关键步骤 + 调用关系”的说明方式，但内容以当前代码为准。

---

## 1. `RegistrationApp` 类

### 1.1 类作用

`RegistrationApp` 是整个命令行程序的入口调度层，负责把“YAML 配置 + 命令行参数”转成具体运行流程。

它当前支持 3 类入口：

1. 单次运行：执行一个样本的配准
2. 批处理运行：遍历一个数据集目录批量执行
3. 对比运行：遍历多组算法组合并输出 comparison 表

它本身不直接做图像配准，而是负责：

- 判断 YAML 类型
- 加载/覆盖配置
- 选择合适的 Pipeline
- 组织输出目录
- 写出文本摘要、JSON 摘要、CSV 汇总

---

## 2. `registration_app.h`

### `class RegistrationApp`
```cpp
class RegistrationApp
```

```cpp
/// 命令行应用入口。
///
/// 负责解析单次配准与批处理两类运行模式，组织 YAML 加载、参数覆盖、
/// 输出目录规划以及汇总结果写出。
class RegistrationApp;
```

补充理解：

- 这份注释略旧，因为当前实现已经不止“单次 + 批处理”两类。
- 现在还新增了 `compare` 类型 YAML 路由，因此类职责应理解为：
  “命令行总入口 + YAML 类型分发器 + 运行结果汇总器”。

---

### `struct Args`
```cpp
struct Args {
    std::filesystem::path pipeline_yaml;
    std::filesystem::path image1;
    std::filesystem::path image2;
    std::filesystem::path output_dir;
};
```

```cpp
/// 单次运行的命令行参数。
/// pipeline_yaml：主配置 YAML 路径
/// image1 / image2：可选的命令行输入图像，会覆盖 YAML 中 io.image1 / io.image2
/// output_dir：可选输出根目录，会覆盖 YAML 中 io.output_dir
struct Args;
```

补充理解：

- `Args` 只表达命令行位置参数，不承担完整配置语义。
- 真正运行时仍要通过 `Config::loadPipeline(...)` 加载完整 `PipelineConfig`。

---

### `enum class OutputMode`
```cpp
enum class OutputMode {
    SINGLE,
    BATCH
};
```

```cpp
/// 输出目录模式：
/// SINGLE -> 输出路径中拼接 single
/// BATCH  -> 输出路径中拼接 batch
enum class OutputMode;
```

补充理解：

- 当前 `compare` 运行没有单独定义 `COMPARE` 枚举。
- compare 模式自己直接控制输出根目录和子目录，不走 `buildOutputDir(OutputMode, ...)` 这套统一拼接。

---

### `struct BatchConfig`
```cpp
struct BatchConfig {
    std::string name;
    std::filesystem::path pipeline_yaml;
    DatasetLoader::Options dataset;
    std::filesystem::path output_root;
    bool save_visuals = true;
    bool summary_csv = true;
};
```

```cpp
/// 批处理配置快照。
/// 负责保存 batch.yaml 中会被 runBatch 使用的关键信息。
struct BatchConfig;
```

补充理解：

- 它不是完整 PipelineConfig。
- 它只保存批处理外层逻辑需要的字段：
  - 用哪个 pipeline
  - 扫哪个数据集
  - 输出到哪
  - 是否关闭可视化
  - 是否输出 summary.csv

---

## 3. 匿名命名空间内的辅助函数

这些函数都定义在 `registration_app.cpp` 的匿名命名空间里，作用是“只给当前实现文件内部使用”。

---

### `std::string fmtMs(double v)`
```cpp
std::string fmtMs(double v);
```

```cpp
// 将毫秒数格式化为统一精度的字符串，保证终端摘要的小数位一致。
// 步骤：
//   1. 创建 ostringstream
//   2. 设置 fixed + setprecision(2)
//   3. 追加 " ms"
//   4. 返回字符串
```

---

### `void appendTimingSummary(std::ostringstream& oss, const RegistrationResult& r)`
```cpp
void appendTimingSummary(std::ostringstream& oss, const RegistrationResult& r);
```

```cpp
// 向摘要文本中追加统一的阶段耗时表。
// 输出顺序固定为：
//   load / extract / match / filter / geometry / warp / total
```

补充理解：

- 它把 keypoint 和 structure 两条线的耗时格式统一了。
- 这样 `buildKeypointSummaryText` 和 `buildStructureSummaryText` 不需要各维护一套输出模板。

---

### `void appendEvaluationSummary(std::ostringstream& oss, const EvaluationData& evaluation)`
```cpp
void appendEvaluationSummary(std::ostringstream& oss, const EvaluationData& evaluation);
```

```cpp
// 向摘要文本中追加评测指标。
// 若 evaluation.metrics 为空则不输出任何内容。
// 若某个指标 valid=false，则输出 N/A。
// 若指标附带 note，则以 "(note)" 形式附加。
```

---

### `std::string jsonEscape(const std::string& s)`
```cpp
std::string jsonEscape(const std::string& s);
```

```cpp
// 对字符串中的 JSON 特殊字符进行转义。
// 当前处理：
//   \   "   \n   \r   \t
```

补充理解：

- 这里没有引入 JSON 库，而是手工拼接 JSON。
- 因此这个函数是 `buildSummaryJson(...)` 的安全前置步骤。

---

### `std::string buildKeypointSummaryText(const RegistrationContext& ctx)`
```cpp
std::string buildKeypointSummaryText(const RegistrationContext& ctx);
```

```cpp
// 构建点特征法的文本摘要。
// 重点输出：
//   status / message
//   keypoints 数量
//   raw_matches / filtered / inliers
//   inlier_ratio
//   warp IoU / warp NMAD（若有效）
//   timings
//   metrics
```

---

### `std::string buildStructureSummaryText(const RegistrationContext& ctx)`
```cpp
std::string buildStructureSummaryText(const RegistrationContext& ctx);
```

```cpp
// 构建结构法的文本摘要。
// 重点输出：
//   status / message
//   structure type
//   structures 数量
//   translation
//   response（当前直接复用 result.inlier_ratio）
//   warp IoU / warp NMAD（若有效）
//   timings
//   metrics
```

这里是当前版本相对旧版最重要的修正点之一：

```cpp
if (gd.valid && !gd.A.empty() && gd.A.rows >= 2 && gd.A.cols >= 3) {
    ...
} else if (!smd.affine.empty() && smd.affine.rows >= 2 && smd.affine.cols >= 3) {
    ...
} else {
    ...
}
```

补充理解：

- 旧理解通常会默认“结构法平移总是来自几何估计器”。
- 当前代码不是这样：
  1. 如果 `geometry_data` 有效，优先显示几何估计结果
  2. 如果几何估计失败，但结构匹配阶段自己给出了 `affine`
     ，则回退显示 `structure_match_data.affine`
  3. 如果连 affine 都没有，就回退到 `structure_match_data.translation`

这说明当前摘要逻辑已经显式处理了“几何估计失败但关联器仍给出平移/仿射先验”的情况。

---

### `std::string buildSummaryText(const RegistrationContext& ctx, MethodFamily family)`
```cpp
std::string buildSummaryText(const RegistrationContext& ctx, MethodFamily family);
```

```cpp
// 根据方法族选择摘要格式：
//   STRUCTURE -> buildStructureSummaryText
//   其他      -> buildKeypointSummaryText
```

补充理解：

- 旧版通常用 `bool structurePipeline` 区分。
- 当前版本已经改成 `MethodFamily family`，这比 bool 更清晰，也给 DIRECT 扩展留了位置。

---

### `void printSummary(const RegistrationContext& ctx, MethodFamily family)`
```cpp
void printSummary(const RegistrationContext& ctx, MethodFamily family);
```

```cpp
// 直接将 buildSummaryText(...) 的结果输出到 std::cout。
```

---

### `std::shared_ptr<IPipeline> createPipelineForConfig(const PipelineConfig& cfg)`
```cpp
std::shared_ptr<IPipeline> createPipelineForConfig(const PipelineConfig& cfg);
```

```cpp
// 根据 cfg.method_family 创建对应的 pipeline。
// 当前逻辑：
//   STRUCTURE -> StructurePipeline
//   DIRECT    -> 预留 TODO
//   KEYPOINT  -> KeypointPipeline
```

补充理解：

- 旧版理解往往是“看有没有 structure_path 来决定创建哪种 pipeline”。
- 当前版本不再靠某个路径字段隐式判断，而是明确依赖 `method_family`。
- 这意味着配置语义更清楚，也更容易支持 DIRECT 模式。

---

### `std::string sampleStemFromPaths(const fs::path& image1, const fs::path& image2)`
```cpp
std::string sampleStemFromPaths(const fs::path& image1, const fs::path& image2);
```

```cpp
// 用两个输入图像的 stem 生成样本标识名：
//   image1.stem() + "__" + image2.stem()
```

---

### `fs::path normalizeOutputBaseRoot(fs::path root)`
```cpp
fs::path normalizeOutputBaseRoot(fs::path root);
```

```cpp
// 规范化输出根目录。
// 若末级目录已经是 "single" 或 "batch"，则退回父目录。
// 目的是避免 buildOutputDir 后出现：
//   .../single/single/...
//   .../batch/batch/...
```

---

### `std::string buildSummaryJson(const RegistrationContext& ctx, const PipelineConfig& cfg, const std::string& sample_name)`
```cpp
std::string buildSummaryJson(const RegistrationContext& ctx,
                             const PipelineConfig& cfg,
                             const std::string& sample_name);
```

```cpp
// 构建 JSON 摘要字符串。
// 输出内容包括：
//   pipeline_name
//   method_family
//   sample_name
//   status / message
//   image1_path / image2_path
//   counts
//   quality
//   translation（仅 structure）
//   timings_ms
//   metrics
```

这里也有一个当前版本新增的关键修正：

- 如果结构法几何估计有效，则 `translation` 写 `geometry_data.A`
- 否则直接写 `structure_match_data.translation`

也就是说，JSON 摘要与文本摘要现在是一致的回退策略。

---

### `void writeRunSummaryFiles(const RegistrationContext& ctx, const PipelineConfig& cfg, const std::string& sample_name)`
```cpp
void writeRunSummaryFiles(const RegistrationContext& ctx,
                          const PipelineConfig& cfg,
                          const std::string& sample_name);
```

```cpp
// 在 output_dir 下写出：
//   run_summary.txt
//   run_summary.json
```

步骤：

1. 若 `ctx.output_dir` 为空则直接返回
2. 用 `buildSummaryText(...)` 生成文本摘要
3. 用 `buildSummaryJson(...)` 生成 JSON 摘要
4. 写入 `run_summary.txt`
5. 写入 `run_summary.json`

---

### `std::vector<std::string> collectMetricColumns(const std::vector<EvaluationData>& evaluations)`
```cpp
std::vector<std::string> collectMetricColumns(const std::vector<EvaluationData>& evaluations);
```

```cpp
// 从全部 EvaluationData 中收集唯一的指标名，用于 CSV 表头生成。
```

补充理解：

- 不同样本不一定拥有完全相同的 metrics 集合。
- 所以写 CSV 之前要先求并集，再按列展开。

---

## 4. `RegistrationApp` 成员函数

### `static void printUsage(const std::string& exe)`
```cpp
static void printUsage(const std::string& exe);
```

```cpp
// 输出命令行帮助信息。
// 当前展示：
//   1. 单次运行格式
//   2. batch 运行格式
//   3. 若干 examples
```

补充理解：

- 当前帮助文本还没有把 compare YAML 用法单独写进去。
- 但 `run(const Args&)` 实际已经支持 compare 配置。

---

### `static int runSingle(const Args& args)`
```cpp
static int runSingle(const Args& args);
```

```cpp
// 执行单个样本的配准流程。
```

关键步骤：

1. 校验 `pipeline_yaml` 是否为空、是否存在
2. 调用 `Config::loadPipeline(...)` 加载 `PipelineConfig`
3. 若命令行传入 `image1/image2`，覆盖 YAML 中对应输入路径
4. 校验最终输入图像路径是否存在
5. 通过 `normalizeOutputBaseRoot(...) + buildOutputDir(...)` 生成统一输出目录
6. 创建输出目录
7. 通过 `createPipelineForConfig(...)` 创建实际 Pipeline
8. 调用 `pipeline->configure(cfg)`
9. 创建 `RegistrationContext`
10. 调用 `pipeline->run(ctx)`
11. 写出 `run_summary.txt / run_summary.json`
12. 额外写一个单样本 `summary.csv`
13. 调用 `pipeline->showWindows(ctx)`
14. 输出终端摘要
15. 成功返回 `0`，失败返回 `1`

补充理解：

- `runSingle(...)` 不是简单的“加载配置然后跑”。
- 它还承担了单次运行的输出目录规范化和摘要产物落盘。

---

### `static bool isBatchYaml(const YAML::Node& node)`
```cpp
static bool isBatchYaml(const YAML::Node& node);
```

```cpp
// 判断某个 YAML 是否是 batch 配置。
// 当前判定条件：
//   node 是 Map
//   且同时存在 "pipeline" 和 "dataset"
```

---

### `static bool isCompareYaml(const YAML::Node& node)`
```cpp
static bool isCompareYaml(const YAML::Node& node);
```

```cpp
// 判断某个 YAML 是否是 compare 配置。
// 当前判定条件：
//   node 是 Map
//   且同时存在 "base_pipeline" 和 "combinations"
```

这部分是当前实现相对旧版新增的内容，旧笔记里通常没有。

---

### `static int runCompare(const std::filesystem::path& compare_yaml)`
```cpp
static int runCompare(const std::filesystem::path& compare_yaml);
```

```cpp
// 执行对比实验：
//   基于一个基础 pipeline 和一个基础 structure YAML
//   遍历 combinations 中给出的多组 (extractor + descriptor)
//   逐组批量执行并输出 comparison.csv
```

关键步骤：

1. 读取 compare YAML
2. 解析 `base_pipeline` 和 `base_structure`
3. 加载基础 `structureNode`
4. 解析 dataset 配置
5. 解析输出根目录，并创建 `tmp` 目录
6. 遍历 `combinations`
7. 对每组组合修改 `structureNode`
   - `extractor.method`
   - `association.params.line_descriptor.descriptor`
   - `association.params.line_descriptor.geometric_filter`
8. 将修改后的结构配置写入临时 YAML
9. 加载基础 pipeline 配置
10. 替换 `pipelineCfg.structure_path`
11. 关闭 `draw_matches`，保留 `warp`
12. 重新扫描数据集样本
13. 对每个样本执行 pipeline
14. 统计：
    - 成功数量
    - 平均 IoU
    - 平均 PSNR
15. 收集所有组合结果
16. 写出 `comparison.csv`

补充理解：

- 这个 compare 分支本质上是“参数组合实验控制器”。
- 它不是简单调用 `runBatch(...)`，而是自己重建结构 YAML，再逐组合执行。
- 当前 compare 逻辑明显偏向 line descriptor 系列实验，因为它直接写死修改：
  `association.params.line_descriptor.descriptor`

---

### `static BatchConfig loadBatchConfig(const std::filesystem::path& yaml_path)`
```cpp
static BatchConfig loadBatchConfig(const std::filesystem::path& yaml_path);
```

```cpp
// 加载 batch.yaml，并解析出 runBatch 真正关心的那部分字段。
```

关键步骤：

1. `Config::load(yaml_path)` 读取 YAML
2. 读取 `name`，若缺失则默认取文件 stem
3. 以 `batch.yaml` 所在目录为基准解析 `pipeline`
4. 读取 `dataset.root / pattern_source / pattern_target / include`
5. 读取 `output.root / save_visuals / summary_csv`

补充理解：

- 它把“路径解析相对于 batch 文件目录”这件事集中做掉了。
- 后续 `runBatch(...)` 就不需要再关心相对路径基准问题。

---

### `static std::filesystem::path resolveBatchOutputRoot(const BatchConfig& batch, const PipelineConfig& pipeline_cfg)`
```cpp
static std::filesystem::path resolveBatchOutputRoot(const BatchConfig& batch,
                                                    const PipelineConfig& pipeline_cfg);
```

```cpp
// 规范化批处理输出根目录。
// 当前实现实际只使用 batch.output_root。
```

补充理解：

- 参数里虽然带 `pipeline_cfg`，但当前版本函数体并没有使用它。
- 说明这个接口可能是为了兼容未来扩展保留下来的。

---

### `static std::filesystem::path buildOutputDir(OutputMode mode, const std::filesystem::path& base_root, const PipelineConfig& cfg, const std::string& sample_name)`
```cpp
static std::filesystem::path buildOutputDir(OutputMode mode,
                                            const std::filesystem::path& base_root,
                                            const PipelineConfig& cfg,
                                            const std::string& sample_name);
```

```cpp
// 统一拼接输出目录：
//   {base_root}/{single|batch}/{method_family}/{pipeline_name}/{sample_name}
```

补充理解：

- 这正是当前项目输出目录规范的核心函数。
- 旧版理解如果还是“按 keypoint/structure 由某个 bool 控制”，这里需要改成：
  通过 `methodFamilyDir(cfg.methodFamily())` 生成中间层目录。

---

### `static void writeSummaryCsv(...)`
```cpp
static void writeSummaryCsv(const std::filesystem::path& csv_path,
                            MethodFamily family,
                            const std::vector<std::string>& sample_names,
                            const std::vector<RegistrationResult>& results,
                            const std::vector<EvaluationData>& evaluations);
```

```cpp
// 写批量/单样本统一 CSV 汇总。
```

关键步骤：

1. 调用 `collectMetricColumns(...)` 收集指标列
2. 先写表头
   - `sample_name, success, message`
   - 结构法写 `num_structures_first/second`
   - 点特征法写 `num_keypoints_first/second`
   - 后面追加 matches / inliers / quality / timings / metrics
3. 遍历每个样本输出一行
4. 对 `message`、`sample_name` 调用 `csvEscape(...)`
5. 最后调用 `file_utils::writeWholeFile(...)`

补充理解：

- 这个函数现在已经是“单次 + 批处理通用”的汇总输出器。
- 单次运行也会写一份只包含当前样本的 `summary.csv`。

---

### `static int runBatch(const std::filesystem::path& batch_yaml)`
```cpp
static int runBatch(const std::filesystem::path& batch_yaml);
```

```cpp
// 执行批量任务。
```

关键步骤：

1. 调用 `loadBatchConfig(...)`
2. 调用 `Config::loadPipeline(...)` 得到基础 `PipelineConfig`
3. 调用 `resolveBatchOutputRoot(...)`
4. 构造 `pipeline_root`
5. 用 `DatasetLoader` 扫描样本
6. 若样本为空则返回错误码 `7`
7. 创建输出目录
8. 初始化汇总容器
9. 遍历每个样本
10. 复制 `base_cfg`
11. 用样本路径替换 `image1_path / image2_path`
12. 用 `buildOutputDir(...)` 生成该样本输出目录
13. 若 `save_visuals=false`，关闭 `draw_matches` 和 `warp`
14. 强制关闭窗口显示相关选项
15. 创建 pipeline 并 `configure`
16. 若配置失败，记录失败结果并继续下一个样本
17. 执行 `pipeline->run(ctx)`
18. 写样本摘要文件
19. 打印终端摘要
20. 收集 `result / evaluation`
21. 统计成功样本名
22. 若 `summary_csv=true`，在 pipeline 层输出 `summary.csv`
23. 打印批处理总摘要
24. 若全部成功返回 `0`，否则返回 `1`

补充理解：

- 当前批处理不是简单把 `runSingle(...)` 套一层循环。
- 它有自己独立的“样本级容错”和“整批汇总”逻辑。

---

### `static int run(const Args& args)`
```cpp
static int run(const Args& args);
```

```cpp
// 总入口路由函数。
// 当前支持：
//   compare YAML
//   batch YAML
//   single pipeline YAML
```

关键步骤：

1. 校验 YAML 路径非空且存在
2. 调用 `Config::load(...)` 先把 YAML 当普通节点读出来
3. 若 `isCompareYaml(node)` 成立，进入 `runCompare(...)`
4. 否则若 `isBatchYaml(node)` 成立，进入 `runBatch(...)`
5. 否则进入 `runSingle(...)`

补充理解：

- 这里是当前版本相对旧版的另一个核心修正点。
- 旧版可能只知道“batch / single 二选一”，现在已经是“三路分发”。

---

### `static int run(int argc, char** argv)`
```cpp
static int run(int argc, char** argv);
```

```cpp
// 命令行入口。
// 负责把位置参数打包成 Args，再委托给 run(const Args&)。
```

关键步骤：

1. 若 `argc < 2`，输出帮助并返回 `1`
2. `argv[1]` 作为 YAML 路径
3. `argv[2]`、`argv[3]`、`argv[4]` 依次作为 `image1/image2/output_dir`
4. 调用 `run(args)`

补充理解：

- 当前命令行层保持了“最小解析”风格，没有引入复杂 CLI 框架。
- 复杂逻辑都留在 `run(const Args&)` 和后续配置层处理。

---

## 5. 当前版本最需要修正的旧理解

如果你之前的笔记基于老版代码，下面这些点最值得更新：

### 5.1 `buildSummaryText(...)` 不再用 `bool structurePipeline`

旧理解：

- 用一个 bool 判断走结构法摘要还是点特征摘要

当前实现：

- 改成 `MethodFamily family`
- 语义更明确，也给 DIRECT 预留了扩展位

---

### 5.2 `createPipelineForConfig(...)` 不再靠 `structure_path` 隐式判断

旧理解：

- 配置里有 `structure_path` 就创建 `StructurePipeline`

当前实现：

- 显式看 `cfg.method_family`

这代表 pipeline 类型判断规则已经从“路径存在性”升级为“配置语义字段”。

---

### 5.3 入口分发已经不是只有 `single / batch`

旧理解：

- `run(args)` 只区分 `batch.yaml` 和普通 pipeline YAML

当前实现：

- 先判断 `compare`
- 再判断 `batch`
- 否则才是 `single`

---

### 5.4 结构法摘要里的平移有回退逻辑

旧理解：

- 结构法平移直接来自 `geometry_data.A`

当前实现：

1. 优先用有效的 `geometry_data.A`
2. 否则回退到 `structure_match_data.affine`
3. 再否则回退到 `structure_match_data.translation`

这意味着当前摘要系统已经专门处理了“几何估计失败但结构关联结果仍可报告”的场景。

---

### 5.5 单次运行现在也会写 `summary.csv`

旧理解：

- CSV 只是批处理输出

当前实现：

- `runSingle(...)` 也会写一份只包含当前样本的 `summary.csv`

---

## 6. 调用关系图（当前版）

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


---

## 7. 一句话总结

当前 `registration_app.cpp` 已经从“单次/批处理的薄入口”发展成了一个更完整的运行控制层，新增了：

- `compare` 配置路由
- 文本摘要与 JSON 摘要
- 单样本 CSV 输出
- 基于 `method_family` 的 pipeline 分发
- 结构法平移结果的失败回退逻辑

所以如果继续沿用旧版笔记，最容易出错的就是：

- 函数签名变了
- 分发路径变了
- 摘要逻辑变了
- `compare` 分支漏记了

