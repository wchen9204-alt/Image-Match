# `core/config` 当前实现笔记

本文档基于当前工程中的以下文件整理：

- `project/include/core/config.h`
- `project/src/core/config.cpp`

说明：

- 这一组文件负责“配置读取与路径解析”。
- 它不负责算法执行，只负责把 YAML 配置文件转换成程序内部可直接使用的 `PipelineConfig`。
- 文档风格沿用 `registration_app` 那份说明：按“结构/函数职责 + 关键步骤 + 旧理解修正点”组织。

---

## 1. `config.h` 的定位

`core/config.h` 主要定义了两类内容：

1. 配准方法族枚举 `MethodFamily`
2. pipeline 级配置结构 `PipelineConfig`
3. 配置读取工具类 `Config`

它是“应用层 YAML 配置”和“程序内部运行配置”之间的桥梁。

---

## 2. `MethodFamily`

### `enum class MethodFamily`
```cpp
enum class MethodFamily {
    KEYPOINT,
    STRUCTURE,
    DIRECT
};
```

```cpp
/// 配准方法族，用于区分点特征法、结构法、直接法等的输出和摘要组织。
enum class MethodFamily;
```

补充理解：

- 这个枚举不是算法类型本身，而是更高一层的“方法族分类”。
- 它决定的通常不是某个检测器/匹配器，而是：
  - 该走哪个 Pipeline
  - 输出目录怎么分层
  - 摘要里该打印哪种统计信息

当前三类含义：

- `KEYPOINT`：点特征法
- `STRUCTURE`：结构法
- `DIRECT`：直接法，当前代码里已预留但还未完整展开

---

### `inline const char* methodFamilyDir(MethodFamily f)`
```cpp
inline const char* methodFamilyDir(MethodFamily f);
```

```cpp
/// 将 MethodFamily 枚举转为输出目录名。
```

当前映射：

- `KEYPOINT -> "keypoint"`
- `STRUCTURE -> "structure"`
- `DIRECT -> "direct"`
- 兜底 -> `"unknown"`

补充理解：

- 这是输出目录分层的核心辅助函数之一。
- `registration_app.cpp` 中的 `buildOutputDir(...)` 就依赖它来生成：
  `{base_root}/{single|batch}/{method_family}/{pipeline_name}/{sample_name}`

---

### `inline const char* methodFamilyLabel(MethodFamily f)`
```cpp
inline const char* methodFamilyLabel(MethodFamily f);
```

```cpp
/// 将 MethodFamily 枚举转为人类可读标签。
```

当前映射：

- `KEYPOINT -> "KeypointPipeline"`
- `STRUCTURE -> "StructurePipeline"`
- `DIRECT -> "DirectPipeline"`
- 兜底 -> `"UnknownPipeline"`

补充理解：

- 它比 `methodFamilyDir(...)` 更偏“显示用途”。
- 一个用于目录名，一个用于更可读的标签名。

---

## 3. `PipelineConfig`

### `struct PipelineConfig`
```cpp
struct PipelineConfig {
    ...
};
```

```cpp
/// 单个 pipeline 配置解析后的结果。
struct PipelineConfig;
```

### 3.1 作用

`PipelineConfig` 是整个项目最核心的“运行时配置快照”之一。

它的作用是：

- 接收 `Config::loadPipeline(...)` 解析后的结果
- 统一保存：
  - 方法族
  - 子配置路径
  - IO 路径
  - 可视化开关
  - warp 校验参数
- 作为 `pipeline->configure(cfg)` 的输入

它不是原始 YAML，也不是某个单独模块的参数，而是“应用层整理过的完整 pipeline 配置”。

---

### 3.2 字段分组理解

#### 1. 基本标识
```cpp
std::string name;
MethodFamily method_family = MethodFamily::KEYPOINT;
```

说明：

- `name`：pipeline 名称，通常来自 YAML 的 `name`
- `method_family`：该 pipeline 属于哪一类方法族

补充理解：

- 当前实现中，`method_family` 已经成为 pipeline 分发的核心依据。
- 它不再只是一个“显示字段”，而是会影响：
  - `createPipelineForConfig(...)`
  - 输出目录
  - 摘要格式

---

#### 2. 各方法族/阶段子配置路径
```cpp
std::filesystem::path keypoint_path;
std::filesystem::path structure_path;
std::filesystem::path direct_path;
std::filesystem::path matcher_path;
std::filesystem::path geometry_path;
std::vector<std::filesystem::path> filter_paths;
std::filesystem::path evaluator_path;
```

说明：

- 保存 pipeline YAML 中引用的子 YAML 的最终解析路径。

补充理解：

- 这些路径在 `loadPipeline(...)` 中会统一通过 `resolvePath(...)` 变成可直接使用的路径。
- 换句话说，后续模块基本不再处理“相对路径怎么拼”的问题。

---

#### 3. 输入输出路径
```cpp
std::filesystem::path image1_path;
std::filesystem::path image2_path;
std::filesystem::path output_dir;
```

说明：

- 对应 pipeline YAML 中 `io:` 下的配置。

补充理解：

- 这几个值可能在 `RegistrationApp::runSingle(...)` / `runBatch(...)` 中被命令行或样本路径覆盖。
- 所以它们是“基础配置值”，不是绝对最终值。

---

#### 4. 可视化控制
```cpp
bool draw_keypoints = false;
bool draw_matches = true;
bool draw_inliers_only = false;
int max_matches_drawn = 100;
bool warp = true;
bool show_source_window = false;
bool show_target_window = false;
bool show_warped_window = false;
int wait_key = 0;
```

说明：

- 对应 `visualization:` 节。

补充理解：

- 这里同时控制两类行为：
  1. 是否生成输出图像
  2. 是否弹出窗口显示

也就是说：

- `draw_matches` / `warp` 更偏“保存输出内容”
- `show_*_window` / `wait_key` 更偏“交互显示行为”

---

#### 5. 配准结果校验参数
```cpp
bool validate_warp_overlap = false;
double min_warp_overlap_iou = 0.20;
int warp_overlap_foreground_threshold = 10;

bool validate_warp_photometric = false;
double max_warp_photometric_error = 0.15;
```

说明：

- 对应 `validation:` 节。

补充理解：

- 这些字段并不是估计器参数，而是 warp 完成后的结果校验标准。
- 是否继续认为本次配准成功，会受到这些开关和阈值影响。

---

### `MethodFamily methodFamily() const`
```cpp
MethodFamily methodFamily() const { return method_family; }
```

```cpp
/// 返回当前 pipeline 的方法族。
```

补充理解：

- 这是一个很小的包装函数，但项目里很多地方都用它而不是直接读字段。
- 这样调用侧语义会更清楚，比如：
  - `cfg.methodFamily()`
  - 比直接写 `cfg.method_family` 更像“获取配置含义”

---

## 4. `Config` 类

### `class Config`
```cpp
class Config {
public:
    static YAML::Node load(const std::filesystem::path& path);
    static PipelineConfig loadPipeline(const std::filesystem::path& path);
    static std::filesystem::path resolvePath(const std::filesystem::path& base_dir,
                                             const std::string& relative_or_absolute);
};
```

```cpp
/// 配置文件加载与路径解析工具。
class Config;
```

### 4.1 类作用

`Config` 是一个纯工具类，负责三件事：

1. 从磁盘读取 YAML
2. 解析 pipeline YAML 为 `PipelineConfig`
3. 统一处理配置路径解析

它的定位可以理解为：

- “配置装载器”
- “路径归一化入口”
- “pipeline 顶层配置解释器”

---

## 5. `config.cpp` 中的函数说明

### `static YAML::Node Config::load(const fs::path& path)`
```cpp
static YAML::Node load(const std::filesystem::path& path);
```

```cpp
// 从磁盘读取一个 YAML 文件。
// 若文件不存在，抛出 runtime_error。
// 若 YAML 解析失败，包装 yaml-cpp 异常后继续抛出 runtime_error。
```

关键步骤：

1. 检查 `fs::exists(path)`
2. 若不存在，抛出：
   `Config::load - file not found: ...`
3. 调用 `YAML::LoadFile(path.string())`
4. 若 `yaml-cpp` 抛异常，捕获并重新包装成更清晰的 `runtime_error`

补充理解：

- 这个函数不返回“失败标志”，而是直接抛异常。
- 所以调用方通常需要：
  - 要么自己 `try/catch`
  - 要么让上层统一处理

当前项目里：

- `registration_app.cpp` 的入口层就承担了这部分异常处理

---

### `static fs::path Config::resolvePath(const fs::path& base_dir, const std::string& relative_or_absolute)`
```cpp
static std::filesystem::path resolvePath(const std::filesystem::path& base_dir,
                                         const std::string& relative_or_absolute);
```

```cpp
// 将路径字符串解析成最终可用路径。
// 允许输入：
//   - 空字符串
//   - 绝对路径
//   - 相对 base_dir 的路径
//   - 相对当前工作目录的路径
```

这是 `core/config` 里最容易被低估、但其实很关键的函数。

### 关键步骤

1. 如果输入字符串为空，直接返回空路径
2. 构造 `fs::path p(relative_or_absolute)`
3. 如果 `p` 是绝对路径且存在，直接返回它
4. 如果给了 `base_dir`：
   - 先尝试 `base_dir / p`
   - 若存在，返回其 `weakly_canonical(...)`
5. 再尝试：
   - `current_path() / p`
6. 如果还找不到：
   - 从 `base_dir` 开始向上回退最多 4 层
   - 每层都尝试 `walk / p`
7. 如果都没命中：
   - 若 `base_dir` 非空，返回 `weakly_canonical(base_dir / p)`
   - 否则返回 `weakly_canonical(current_path() / p)`

### 补充理解

- 它不只是“相对路径拼接”，而是一套比较宽松的路径搜寻策略。
- 当前设计明显是为了兼容：
  - 配置文件相对引用
  - 在不同工作目录下运行程序
  - 一些旧 YAML 的路径写法

### 旧理解容易出错的地方

旧理解可能会以为：

- 只做 `base_dir / relative_path`

但当前实现其实还会：

- 尝试当前工作目录
- 尝试向上回退父目录搜索

这意味着它更像“容错型解析器”，不是单纯的拼路径函数。

---

### `static PipelineConfig Config::loadPipeline(const fs::path& path)`
```cpp
static PipelineConfig loadPipeline(const std::filesystem::path& path);
```

```cpp
// 读取一个 pipeline YAML，并解析成 PipelineConfig。
```

这是整个配置模块最核心的函数。

---

### 5.1 它负责什么

它做的不是“把 YAML 原样读出来”，而是：

1. 读取 pipeline YAML
2. 识别该 pipeline 属于哪种方法族
3. 把子配置路径全部解析成规范路径
4. 把 IO / visualization / validation 等高层配置装入 `PipelineConfig`
5. 输出日志，方便确认当前实际生效的配置

---

### 5.2 关键步骤

#### 步骤 1：读取 YAML
```cpp
YAML::Node node = load(path);
const fs::path base = path.parent_path();
```

说明：

- `base` 后续会作为相对路径解析的基准目录。

---

#### 步骤 2：初始化 `PipelineConfig`
```cpp
PipelineConfig cfg;
cfg.name = yaml_utils::getString(node, "name", path.stem().string());
```

说明：

- 若 YAML 没写 `name`，则默认使用文件名 stem。

---

#### 步骤 3：解析 `method_family`
```cpp
const std::string familyStr = yaml_utils::getString(node, "method_family");
...
```

当前逻辑：

1. 如果 YAML 显式提供了 `method_family`
   - `"keypoint"` / `"KEYPOINT"` -> `MethodFamily::KEYPOINT`
   - `"structure"` / `"STRUCTURE"` -> `MethodFamily::STRUCTURE`
   - `"direct"` / `"DIRECT"` -> `MethodFamily::DIRECT`
2. 如果没写 `method_family`
   - 且 `structure` 字段非空
   - 则自动推断为 `STRUCTURE`
3. 否则保持默认值 `KEYPOINT`

补充理解：

- 这一步里显式字段优先，路径推断只是兼容旧配置。
- 当前项目已经在向“显式 method_family”迁移。

---

#### 步骤 4：解析各子配置路径
```cpp
const std::string keypoint_entry = yaml_utils::getString(
    node, "keypoint", yaml_utils::getString(node, "feature"));
cfg.keypoint_path = resolvePath(base, keypoint_entry);
cfg.structure_path = resolvePath(base, yaml_utils::getString(node, "structure"));
cfg.direct_path   = resolvePath(base, yaml_utils::getString(node, "direct"));
cfg.matcher_path  = resolvePath(base, yaml_utils::getString(node, "matcher"));
cfg.geometry_path = resolvePath(base, yaml_utils::getString(node, "geometry"));
```

说明：

- `keypoint` 字段还兼容旧名字 `feature`
- 每个路径都会走 `resolvePath(...)`

补充理解：

- 这一步完成后，调用侧可以直接用这些路径，不需要再次做路径拼接。

---

#### 步骤 5：解析 `filters`
```cpp
if (node["filters"] && node["filters"].IsSequence()) {
    for (const auto& f : node["filters"]) {
        cfg.filter_paths.push_back(resolvePath(base, f.as<std::string>()));
    }
}
```

说明：

- `filters` 是一个 YAML 序列
- 每个元素都会解析成最终路径并压入 `filter_paths`

---

#### 步骤 6：解析 `evaluator`
```cpp
cfg.evaluator_path = resolvePath(base, yaml_utils::getString(node, "evaluator"));
```

---

#### 步骤 7：解析 `io`
```cpp
if (node["io"] && node["io"].IsMap()) {
    ...
}
```

解析内容：

- `image1`
- `image2`
- `output_dir`

注意点：

```cpp
cfg.output_dir = resolvePath(base, yaml_utils::getString(io, "output_dir", "outputs"));
```

也就是说：

- 如果没配置 `output_dir`
- 默认值不是空，而是 `"outputs"`

---

#### 步骤 8：解析 `visualization`
```cpp
if (node["visualization"] && node["visualization"].IsMap()) {
    ...
}
```

解析内容：

- `draw_keypoints`
- `draw_matches`
- `draw_inliers_only`
- `max_matches_drawn`
- `warp`
- `show_source_window`
- `show_target_window`
- `show_warped_window`
- `wait_key`

---

#### 步骤 9：解析 `validation`
```cpp
if (node["validation"] && node["validation"].IsMap()) {
    ...
}
```

解析两个子块：

1. `warp_overlap`
   - `enabled`
   - `min_iou`
   - `foreground_threshold`

2. `photometric`
   - `enabled`
   - `max_nmad`

补充理解：

- 这里配置的是 warp 结果校验规则，不是几何估计器参数。

---

#### 步骤 10：输出日志
```cpp
IR_LOG_INFO("Pipeline '", cfg.name, "' loaded from ", path.string());
...
```

输出内容包括：

- pipeline 名称
- keypoint 路径
- structure 路径
- matcher 路径
- 每个 filter 路径
- geometry 路径
- image1 / image2
- output

补充理解：

- 这一步很重要，因为它让“最终实际生效的配置”能在日志里直接看到。
- 出现路径写错、method_family 理解错、输出目录不对时，第一时间就能从这里排查。

---

## 6. 当前版本最需要修正的旧理解

如果之前的理解基于更老的实现，这几个点最值得更新：

### 6.1 `method_family` 现在是显式一级配置

旧理解：

- pipeline 类型可能主要靠 `structure` 路径是否为空推断

当前实现：

- 优先读 YAML 中的 `method_family`
- 只有缺失时才对旧结构做兼容推断

---

### 6.2 `keypoint` 仍兼容旧字段名 `feature`

当前代码：

```cpp
const std::string keypoint_entry = yaml_utils::getString(
    node, "keypoint", yaml_utils::getString(node, "feature"));
```

说明：

- 旧 YAML 写 `feature: xxx.yaml` 仍可兼容
- 新写法建议统一使用 `keypoint`

---

### 6.3 `resolvePath(...)` 比单纯路径拼接更“宽松”

旧理解：

- 相对路径只会按 `base_dir / path` 拼一次

当前实现：

- 还会尝试当前工作目录
- 还会向上回退最多 4 层

这说明它是一个“容错型路径解析器”。

---

### 6.4 `output_dir` 的默认值不是空，而是 `"outputs"`

也就是说，即使 YAML 没显式写输出目录，`PipelineConfig` 里通常也会得到一个默认输出根。

---

## 7. 调用关系图

```text
Config::load(path)
  -> 检查文件是否存在
  -> YAML::LoadFile()
  -> 返回 YAML::Node

Config::resolvePath(base_dir, path_str)
  -> 空字符串直接返回空路径
  -> 尝试绝对路径
  -> 尝试 base_dir / p
  -> 尝试 current_path() / p
  -> 尝试从 base_dir 向上最多 4 层搜索
  -> 最后返回一个 weakly_canonical 路径

Config::loadPipeline(path)
  -> Config::load(path)
  -> 解析 name
  -> 解析 method_family
  -> 解析 keypoint / structure / direct / matcher / geometry
  -> 解析 filters
  -> 解析 evaluator
  -> 解析 io
  -> 解析 visualization
  -> 解析 validation
  -> 输出日志
  -> 返回 PipelineConfig
```

---

## 8. 一句话总结

`core/config.h/.cpp` 的核心价值不是“读 YAML”这么简单，而是把：

- 方法族语义
- 子配置路径
- 输入输出路径
- 可视化控制
- warp 校验规则

统一收敛成一份 `PipelineConfig`，并用一套相对稳健的路径解析策略把旧配置和新配置都接住。

