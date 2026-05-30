# 详细 API 说明文档

本文按类和结构体分组说明项目 API。说明面向查阅和使用，不贴实现代码。

## 1. App

### `RegistrationApp`

命令行应用入口，负责执行单次配准或批量配准。

成员类型：

| 名称 | 用途 |
|---|---|
| `Args` | 单次运行参数，包含 pipeline YAML、输入图像和输出目录。 |
| `BatchConfig` | 批量运行配置，包含 pipeline、数据集、输出目录和导出选项。 |

`Args` 成员变量：

| 变量 | 用途 |
|---|---|
| `pipeline_yaml` | pipeline 配置文件路径。 |
| `image1` | 可选的第一张输入图像路径，用于覆盖 YAML 中的配置。 |
| `image2` | 可选的第二张输入图像路径，用于覆盖 YAML 中的配置。 |
| `output_dir` | 可选输出目录，用于覆盖 YAML 中的配置。 |

`BatchConfig` 成员变量：

| 变量 | 用途 |
|---|---|
| `name` | 批量任务名称。 |
| `pipeline_yaml` | 批量运行复用的单次 pipeline 配置。 |
| `dataset` | 数据集扫描配置。 |
| `output_root` | 批量输出根目录。 |
| `save_visuals` | 是否保存可视化结果。 |
| `summary_csv` | 是否写出汇总 CSV。 |

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `run(const Args& args)` | 使用已解析参数执行应用。 | `args`：运行参数。 | 进程返回码，`0` 表示成功。 |
| `run(int argc, char** argv)` | 从命令行参数解析并执行应用。 | `argc/argv`：命令行参数。 | 进程返回码。 |
| `printUsage(const std::string& exe)` | 打印命令行用法。 | `exe`：可执行文件名。 | 无。 |
| `runSingle(const Args& args)` | 执行单次配准。 | `args`：单次运行参数。 | 进程返回码。 |
| `runBatch(const std::filesystem::path& batch_yaml)` | 执行批量配准。 | `batch_yaml`：批量配置路径。 | 进程返回码。 |
| `isBatchYaml(const YAML::Node& node)` | 判断 YAML 是否为批量配置。 | `node`：YAML 根节点。 | `true` 表示批量配置。 |
| `loadBatchConfig(const std::filesystem::path& yaml_path)` | 加载批量配置。 | `yaml_path`：配置路径。 | `BatchConfig`。 |
| `resolveBatchOutputRoot(...)` | 计算批量输出根目录。 | `batch`：批量配置；`pipeline_cfg`：单次 pipeline 配置。 | 输出根路径。 |
| `writeSummaryCsv(...)` | 写出批量运行摘要 CSV。 | `csv_path`：CSV 路径；`sample_names`：样本名；`results`：结果列表。 | 无。 |

## 2. Core

### `PipelineConfig`

保存单次 pipeline YAML 解析后的配置。

成员变量：

| 变量 | 用途 |
|---|---|
| `name` | pipeline 名称。 |
| `feature_path` | 特征配置路径。 |
| `matcher_path` | 匹配器配置路径。 |
| `geometry_path` | 几何估计配置路径。 |
| `filter_paths` | 过滤器配置路径列表，按顺序执行。 |
| `image1_path` | 第一张输入图像路径。 |
| `image2_path` | 第二张输入图像路径。 |
| `output_dir` | 输出目录。 |
| `draw_keypoints` | 是否保存关键点可视化。 |
| `draw_matches` | 是否保存匹配可视化。 |
| `draw_inliers_only` | 匹配图是否只绘制内点。 |
| `max_matches_drawn` | 匹配图最大绘制数量。 |
| `warp` | 是否执行图像变换。 |
| `show_source_window` | 是否显示源图窗口。 |
| `show_target_window` | 是否显示目标图窗口。 |
| `show_warped_window` | 是否显示变换后图像窗口。 |
| `wait_key` | OpenCV 窗口等待时间。 |

### `Config`

配置文件加载和路径解析工具。

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `load(const std::filesystem::path& path)` | 读取 YAML 文件。 | `path`：YAML 文件路径。 | `YAML::Node`；失败时抛出异常。 |
| `loadPipeline(const std::filesystem::path& path)` | 读取单次 pipeline YAML 并解析子配置路径。 | `path`：pipeline YAML 路径。 | `PipelineConfig`。 |
| `resolvePath(const std::filesystem::path& base_dir, const std::string& relative_or_absolute)` | 将相对或绝对路径解析为可用路径。 | `base_dir`：基准目录；`relative_or_absolute`：路径字符串。 | 解析后的路径。 |

### `RegistrationContext`

贯穿整条配准流程的共享上下文。

成员变量：

| 变量 | 用途 |
|---|---|
| `feature_data` | 特征阶段数据。 |
| `match_data` | 匹配和过滤阶段数据。 |
| `geometry_data` | 几何估计结果。 |
| `transform_data` | 图像变换结果。 |
| `evaluation` | 评测指标结果。 |
| `result` | 本次运行摘要。 |
| `image1_path` | 第一张输入图像路径。 |
| `image2_path` | 第二张输入图像路径。 |
| `output_dir` | 输出目录。 |
| `warped_image` | 变换后的图像。 |

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `RegistrationContext()` | 构造空上下文。 | 无。 | 新上下文对象。 |
| `reset()` | 清空运行时数据，保留路径配置。 | 无。 | 无。 |

### `RegistrationResult`

一次配准运行的摘要结果。

成员变量：

| 变量 | 用途 |
|---|---|
| `success` | 运行是否成功。 |
| `message` | 状态或错误信息。 |
| `num_keypoints_first` | 第一张图关键点数量。 |
| `num_keypoints_second` | 第二张图关键点数量。 |
| `num_raw_matches` | 原始匹配数量。 |
| `num_filtered_matches` | 过滤后匹配数量。 |
| `num_inliers` | 几何估计内点数量。 |
| `inlier_ratio` | 内点比例。 |
| `mean_reproj_error` | 平均重投影误差。 |
| `t_load_ms` | 加载图像耗时。 |
| `t_extract_ms` | 特征提取耗时。 |
| `t_match_ms` | 匹配耗时。 |
| `t_filter_ms` | 过滤耗时。 |
| `t_geometry_ms` | 几何估计耗时。 |
| `t_warp_ms` | 图像变换耗时。 |
| `t_total_ms` | 总耗时。 |

### `Factory`

根据 YAML 类型创建具体组件。

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `createFeatureExtractor(const YAML::Node& cfg)` | 创建特征提取器。 | `cfg`：feature YAML。 | `std::shared_ptr<IFeatureExtractor>`。 |
| `createMatcher(const YAML::Node& cfg)` | 创建匹配器。 | `cfg`：matcher YAML。 | `std::shared_ptr<IMatcher>`。 |
| `createFilter(const YAML::Node& cfg)` | 创建过滤器。 | `cfg`：filter YAML。 | `std::shared_ptr<IFilter>`。 |
| `createGeometryEstimator(const YAML::Node& cfg)` | 创建几何估计器。 | `cfg`：geometry YAML。 | `std::shared_ptr<IGeometryEstimator>`。 |

### `Registration`

顶层配准对象，持有 pipeline 并执行。

成员变量：

| 变量 | 用途 |
|---|---|
| `_cfg` | 最近一次应用的 pipeline 配置。 |
| `_pipeline` | 实际执行配准流程的 pipeline。 |

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `Registration()` | 构造未配置实例。 | 无。 | 新实例。 |
| `Registration(std::shared_ptr<IPipeline> pipeline)` | 构造并绑定 pipeline。 | `pipeline`：流水线实例。 | 新实例。 |
| `name() const` | 返回实例名称。 | 无。 | 字符串名称。 |
| `configure(const PipelineConfig& cfg)` | 保存配置并初始化 pipeline。 | `cfg`：pipeline 配置。 | 是否成功。 |
| `run(RegistrationContext& ctx)` | 执行配准流程。 | `ctx`：运行上下文。 | 是否成功。 |
| `config() const` | 获取当前配置。 | 无。 | `const PipelineConfig&`。 |
| `pipeline() const` | 获取当前 pipeline。 | 无。 | `std::shared_ptr<IPipeline>`。 |

## 3. Data

### `FeatureImageData`

单张图像的特征数据。

成员变量：

| 变量 | 用途 |
|---|---|
| `image` | 原始 BGR 图像。 |
| `gray` | 灰度图。 |
| `keypoints` | 关键点列表。 |
| `descriptors` | 描述子矩阵。 |

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `clear()` | 清空图像和特征数据。 | 无。 | 无。 |
| `empty() const` | 判断是否缺少有效特征。 | 无。 | `true` 表示关键点或描述子为空。 |

### `FeatureData`

两张图像的特征数据集合。

成员变量：

| 变量 | 用途 |
|---|---|
| `first` | 第一张图的特征数据。 |
| `second` | 第二张图的特征数据。 |
| `type` | 特征类型。 |
| `norm_type` | 描述子匹配距离类型。 |

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `clear()` | 清空两张图的特征和类型信息。 | 无。 | 无。 |
| `empty() const` | 判断是否至少一张图缺少有效特征。 | 无。 | `true` 表示不可用于匹配。 |

### `MatchData`

匹配阶段数据。

成员变量：

| 变量 | 用途 |
|---|---|
| `raw_knn` | 原始 KNN 匹配结果；1-NN 匹配会保存为每行一个匹配。 |
| `filtered` | 过滤后的匹配结果。 |
| `inlier_mask` | 几何估计得到的内点掩码。 |
| `inliers` | 最终内点匹配。 |

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `clear()` | 清空所有匹配数据。 | 无。 | 无。 |

### `GeometryData`

几何估计结果。

成员变量：

| 变量 | 用途 |
|---|---|
| `type` | 几何模型类型。 |
| `H` | 3x3 单应矩阵或统一矩阵。 |
| `A` | 2x3 仿射类矩阵。 |
| `valid` | 几何估计是否有效。 |
| `num_inliers` | 内点数量。 |
| `inlier_ratio` | 内点比例。 |
| `message` | 估计状态说明。 |

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `clear()` | 清空几何估计结果。 | 无。 | 无。 |

### `TransformData`

图像变换阶段数据。

成员变量：

| 变量 | 用途 |
|---|---|
| `type` | 变换类型。 |
| `M` | 统一使用的 3x3 变换矩阵。 |
| `valid` | 变换是否有效。 |

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `clear()` | 清空变换结果。 | 无。 | 无。 |

### `MetricResult`

单个评测指标结果。

成员变量：

| 变量 | 用途 |
|---|---|
| `name` | 指标名称。 |
| `value` | 指标数值。 |
| `valid` | 指标是否有效。 |
| `note` | 附加说明。 |

### `EvaluationData`

一次运行的全部评测结果。

成员变量：

| 变量 | 用途 |
|---|---|
| `metrics` | 指标结果列表。 |

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `find(const std::string& name) const` | 按名称查找指标。 | `name`：指标名称。 | 找到则返回指针，否则返回 `nullptr`。 |
| `asMap() const` | 将有效指标转为键值表。 | 无。 | `std::map<std::string, double>`。 |
| `clear()` | 清空指标列表。 | 无。 | 无。 |

## 4. Interfaces

### `IFeatureExtractor`

特征提取器接口。

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `~IFeatureExtractor()` | 虚析构。 | 无。 | 无。 |
| `name() const` | 返回提取器名称。 | 无。 | 字符串名称。 |
| `type() const` | 返回特征类型。 | 无。 | `FeatureType`。 |
| `normType() const` | 返回推荐匹配距离类型。 | 无。 | `NormType`。 |
| `extract(RegistrationContext& ctx)` | 提取两张图的关键点和描述子。 | `ctx`：上下文。 | 是否成功。 |

### `IMatcher`

描述子匹配器接口。

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `~IMatcher()` | 虚析构。 | 无。 | 无。 |
| `name() const` | 返回匹配器名称。 | 无。 | 字符串名称。 |
| `match(RegistrationContext& ctx)` | 执行描述子匹配。 | `ctx`：上下文。 | 是否成功。 |

### `IFilter`

匹配过滤器接口。

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `~IFilter()` | 虚析构。 | 无。 | 无。 |
| `name() const` | 返回过滤器名称。 | 无。 | 字符串名称。 |
| `apply(RegistrationContext& ctx)` | 对匹配结果执行过滤。 | `ctx`：上下文。 | 是否成功或是否有有效结果。 |

### `IGeometryEstimator`

几何估计器接口。

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `~IGeometryEstimator()` | 虚析构。 | 无。 | 无。 |
| `name() const` | 返回估计器名称。 | 无。 | 字符串名称。 |
| `type() const` | 返回几何模型类型。 | 无。 | `GeometryType`。 |
| `estimate(RegistrationContext& ctx)` | 从匹配点估计几何模型。 | `ctx`：上下文。 | 是否成功。 |

### `IPipeline`

配准流水线接口。

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `~IPipeline()` | 虚析构。 | 无。 | 无。 |
| `name() const` | 返回流水线名称。 | 无。 | 字符串名称。 |
| `configure(const PipelineConfig& cfg)` | 根据配置初始化流水线。 | `cfg`：pipeline 配置。 | 是否成功。 |
| `run(RegistrationContext& ctx)` | 执行完整流水线。 | `ctx`：上下文。 | 是否成功。 |

### `IRegistration`

顶层配准任务接口。

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `~IRegistration()` | 虚析构。 | 无。 | 无。 |
| `name() const` | 返回任务名称。 | 无。 | 字符串名称。 |
| `configure(const PipelineConfig& cfg)` | 应用 pipeline 配置。 | `cfg`：pipeline 配置。 | 是否成功。 |
| `run(RegistrationContext& ctx)` | 执行一次配准。 | `ctx`：上下文。 | 是否成功。 |

### `IWarper`

图像变换器接口。

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `~IWarper()` | 虚析构。 | 无。 | 无。 |
| `name() const` | 返回变换器名称。 | 无。 | 字符串名称。 |
| `warp(RegistrationContext& ctx)` | 执行图像变换。 | `ctx`：上下文。 | 是否成功。 |

## 5. Pipeline

### `BasePipeline`

基础配准流水线。

成员变量：

| 变量 | 用途 |
|---|---|
| `_config` | 当前 pipeline 配置。 |
| `_extractor` | 特征提取器。 |
| `_matcher` | 匹配器。 |
| `_filters` | 过滤器列表。 |
| `_geometry` | 几何估计器。 |
| `_warper` | 图像变换器。 |

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `BasePipeline()` | 构造空流水线。 | 无。 | 新实例。 |
| `configure(const PipelineConfig& cfg)` | 创建并初始化各阶段组件。 | `cfg`：pipeline 配置。 | 是否成功。 |
| `run(RegistrationContext& ctx)` | 执行完整配准流程。 | `ctx`：上下文。 | 是否成功。 |
| `loadImages(RegistrationContext& ctx)` | 加载输入图像并生成灰度图。 | `ctx`：上下文。 | 是否成功。 |
| `runExtract(RegistrationContext& ctx)` | 执行特征提取。 | `ctx`：上下文。 | 是否成功。 |
| `runMatch(RegistrationContext& ctx)` | 执行描述子匹配。 | `ctx`：上下文。 | 是否成功。 |
| `runFilters(RegistrationContext& ctx)` | 按顺序执行过滤器。 | `ctx`：上下文。 | 是否成功。 |
| `runGeometry(RegistrationContext& ctx)` | 执行几何估计。 | `ctx`：上下文。 | 是否成功。 |
| `runWarp(RegistrationContext& ctx)` | 执行图像变换。 | `ctx`：上下文。 | 是否成功。 |
| `saveOutputs(RegistrationContext& ctx)` | 保存可视化和结果图像。 | `ctx`：上下文。 | 是否成功。 |
| `showWindows(RegistrationContext& ctx)` | 根据配置显示 OpenCV 窗口。 | `ctx`：上下文。 | 是否成功。 |

### `FeaturePipeline`

当前默认特征配准流水线。

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `FeaturePipeline()` | 构造默认流水线。 | 无。 | 新实例。 |
| `name() const` | 返回流水线名称。 | 无。 | `"FeaturePipeline"`。 |

## 6. Feature Extractors

### 通用说明

所有特征提取器都实现 `IFeatureExtractor`，构造函数读取 YAML 参数，`extract` 执行 OpenCV 特征提取并写入 `ctx.feature_data`。

### `SiftExtractor`

成员变量：

| 变量 | 用途 |
|---|---|
| `_nfeatures` | 最多保留的特征点数量。 |
| `_nOctaveLayers` | 每个 octave 的层数。 |
| `_contrastThreshold` | 对比度阈值。 |
| `_edgeThreshold` | 边缘阈值。 |
| `_sigma` | 初始高斯尺度。 |
| `_impl` | OpenCV SIFT 实例。 |

成员函数：`SiftExtractor(const YAML::Node& cfg)` 初始化参数；`name()` 返回 `"SIFT"`；`type()` 返回 `SIFT`；`normType()` 返回 `L2`；`extract(ctx)` 提取 SIFT 特征。

### `SurfExtractor`

成员变量：`_hessianThreshold` 控制响应阈值；`_nOctaves` 和 `_nOctaveLayers` 控制尺度空间；`_extended` 控制描述子长度；`_upright` 控制是否忽略方向；`_impl` 为 OpenCV SURF 实例。

成员函数：构造函数初始化参数；`name()` 返回 `"SURF"`；`type()` 返回 `SURF`；`normType()` 返回 `L2`；`extract(ctx)` 提取 SURF 特征。

### `OrbExtractor`

成员变量：`_nfeatures`、`_scaleFactor`、`_nlevels`、`_edgeThreshold`、`_firstLevel`、`_wtaK`、`_scoreType`、`_patchSize`、`_fastThreshold` 控制 ORB 参数；`_impl` 为 OpenCV ORB 实例。

成员函数：构造函数初始化参数；`name()` 返回 `"ORB"`；`type()` 返回 `ORB`；`normType()` 返回 `HAMMING`；`extract(ctx)` 提取 ORB 特征。

### `BriskExtractor`

成员变量：`_thresh` 为 FAST 阈值；`_octaves` 为尺度层数；`_patternScale` 为采样模式尺度；`_impl` 为 OpenCV BRISK 实例。

成员函数：构造函数初始化参数；`name()` 返回 `"BRISK"`；`type()` 返回 `BRISK`；`normType()` 返回 `HAMMING`；`extract(ctx)` 提取 BRISK 特征。

### `KazeExtractor`

成员变量：`_extended`、`_upright`、`_threshold`、`_nOctaves`、`_nOctaveLayers`、`_diffusivity` 控制 KAZE 参数；`_impl` 为 OpenCV KAZE 实例。

成员函数：构造函数初始化参数；`name()` 返回 `"KAZE"`；`type()` 返回 `KAZE`；`normType()` 返回 `L2`；`extract(ctx)` 提取 KAZE 特征。

### `AkazeExtractor`

成员变量：`_descriptorType`、`_descriptorSize`、`_descriptorChannels` 控制描述子；`_threshold`、`_nOctaves`、`_nOctaveLayers`、`_diffusivity` 控制检测；`_norm` 保存匹配距离类型；`_impl` 为 OpenCV AKAZE 实例。

成员函数：构造函数初始化参数；`name()` 返回 `"AKAZE"`；`type()` 返回 `AKAZE`；`normType()` 返回 `_norm`；`extract(ctx)` 提取 AKAZE 特征。

## 7. Matchers

### `BfMatcher`

暴力匹配器，可通过 `params.method` 选择 `match` / `knn` / `radius`。

成员变量：

| 变量 | 用途 |
|---|---|
| `_normType` | 匹配距离类型，`AUTO` 时使用特征阶段给出的类型。 |
| `_method` | 具体匹配接口，支持 `match` / `knn` / `radius`。 |
| `_knnK` | KNN 的 K 值。 |
| `_radius` | 半径匹配阈值。 |
| `_crossCheck` | 是否启用 OpenCV 内置交叉验证。 |

成员函数：

| 函数 | 作用 | 参数 | 返回值 |
|---|---|---|---|
| `BfMatcher(const YAML::Node& cfg)` | 从 YAML 初始化距离类型、匹配方法、KNN K 值、半径阈值和 crossCheck。 | `cfg`：matcher 配置。 | 新实例。 |
| `name() const` | 返回匹配器名称。 | 无。 | `"BFMatcher"`。 |
| `match(RegistrationContext& ctx)` | 按配置执行 `cv::BFMatcher::match` / `knnMatch` / `radiusMatch`，写入 `raw_knn`，`match` 模式还会直接写入 `filtered`。 | `ctx`：上下文。 | 是否产生匹配。 |

### `FlannMatcher`

FLANN KNN 匹配器。

成员变量：

| 变量 | 用途 |
|---|---|
| `_normType` | 匹配距离类型。 |
| `_knnK` | KNN 的 K 值。 |
| `_kdTrees` | KDTree 树数量。 |
| `_lshTableNumber` | LSH 表数量。 |
| `_lshKeySize` | LSH key 大小。 |
| `_lshMultiProbeLevel` | LSH 多探测层级。 |
| `_searchChecks` | FLANN 搜索检查次数。 |
| `_searchEps` | 搜索 eps 参数。 |
| `_searchSorted` | 搜索结果是否排序。 |

成员函数：构造函数读取 FLANN 参数；`name()` 返回 `"FlannMatcher"`；`match(ctx)` 执行 FLANN KNN 匹配。

## 8. Filters

### `RatioTestFilter`

成员变量：`_ratio` 保存 Lowe ratio 阈值。

成员函数：构造函数读取阈值；`name()` 返回 `"RatioTest"`；`apply(ctx)` 从 `raw_knn` 中执行比值检验并写入 `filtered`。

### `CrossCheckFilter`

成员变量：`_enabled` 表示是否启用过滤器。

成员函数：构造函数读取开关；`name()` 返回 `"CrossCheck"`；`apply(ctx)` 对匹配结果执行双向一致性检查并更新 `filtered`。

### `GmsFilter`

成员变量：`_withRotation` 控制旋转一致性；`_withScale` 控制尺度一致性；`_thresholdFactor` 控制 GMS 阈值。

成员函数：构造函数读取 GMS 参数；`name()` 返回 `"GMS"`；`apply(ctx)` 执行空间一致性过滤并写入 `filtered`。

## 9. Geometry Estimators

### `HomographyEstimator`

成员变量：`_method` 为鲁棒估计方法；`_ransacReprojThreshold` 为 RANSAC 重投影阈值；`_maxIters` 为最大迭代次数；`_confidence` 为置信度；`_minInliers` 为最小内点数。

成员函数：构造函数读取参数；`name()` 返回 `"Homography"`；`type()` 返回 `HOMOGRAPHY`；`estimate(ctx)` 估计 3x3 单应矩阵。

### `AffineEstimator`

成员变量：`_method`、`_ransacReprojThreshold`、`_maxIters`、`_confidence`、`_refineIters`、`_minInliers` 控制仿射估计和结果有效性。

成员函数：构造函数读取参数；`name()` 返回 `"Affine2D"`；`type()` 返回 `AFFINE`；`estimate(ctx)` 估计 2x3 仿射矩阵。

### `RigidEstimator`

成员变量：`_method`、`_ransacReprojThreshold`、`_maxIters`、`_confidence`、`_refineIters`、`_minInliers` 控制刚体变换估计。

成员函数：构造函数读取参数；`name()` 返回 `"Rigid2D"`；`type()` 返回 `RIGID`；`estimate(ctx)` 估计刚体变换矩阵。

### `SimilarityEstimator`

成员变量：`_method`、`_ransacReprojThreshold`、`_maxIters`、`_confidence`、`_refineIters`、`_minInliers` 控制相似变换估计。

成员函数：构造函数读取参数；`name()` 返回 `"Similarity2D"`；`type()` 返回 `SIMILARITY`；`estimate(ctx)` 估计相似变换矩阵。

## 10. Transform

### `PerspectiveWarper`

成员函数：构造函数创建实例；`name()` 返回 `"PerspectiveWarper"`；`warp(ctx)` 使用 3x3 变换矩阵生成 `ctx.warped_image`。

### `AffineWarper`

成员函数：构造函数创建实例；`name()` 返回 `"AffineWarper"`；`warp(ctx)` 使用 2x3 仿射矩阵生成 `ctx.warped_image`。

## 11. Dataset

### `Sample`

成员变量：

| 变量 | 用途 |
|---|---|
| `name` | 样本名称。 |
| `source_path` | 源图路径。 |
| `target_path` | 目标图路径。 |
| `H_gt` | 可选真值单应矩阵。 |

成员函数：`has_ground_truth() const` 判断是否携带真值矩阵。

### `DatasetLoader::Options`

成员变量：`root` 为数据集根目录；`pattern_source` 为源图文件名前缀；`pattern_target` 为目标图文件名前缀；`include` 为样本白名单；`extensions` 为可接受图像扩展名。

### `DatasetLoader`

成员变量：`_opt` 保存扫描配置。

成员函数：构造函数保存配置；`load() const` 扫描并返回样本列表；`resolveImage(...) const` 在目录中查找图像；`tryLoadGroundTruth(...) const` 读取真值矩阵。

## 12. Evaluator

### `IMetric`

成员函数：虚析构；`name() const` 返回指标名；`compute(ctx, sample)` 计算指标并返回 `MetricResult`。

### `Evaluator`

成员变量：`_metrics` 保存已加载指标实例。

成员函数：`loadFromYaml(path)` 从文件加载指标；`loadFromNode(root)` 从 YAML 节点加载指标；`clear()` 清空指标；`add(m)` 添加指标；`evaluate(ctx, sample)` 执行全部指标；`createMetric(name, params)` 创建指标实例；`metrics() const` 返回指标列表。

### `MetricStats`

成员变量：`count` 为样本数；`mean` 为均值；`median` 为中位数；`stddev` 为标准差；`minv` 为最小值；`maxv` 为最大值。

### `Statistics`

成员变量：`_raw` 保存每个指标的原始数值列表。

成员函数：`push(ev)` 加入一次评测；`summary() const` 计算统计摘要；`raw() const` 返回原始数据；`clear()` 清空数据。

### `Benchmark::Config`

成员变量：`dataset` 数据集配置；`pipeline_yamls` pipeline 列表；`metrics_yaml` 指标配置；`output_root` 输出根目录；`csv_dir` CSV 目录；`reports_dir` 报告目录；`benchmark_dir` benchmark 目录；`save_visuals` 是否保存可视化。

### `Benchmark`

成员函数：`loadConfig(yaml_path)` 加载 benchmark 配置；`run(cfg)` 执行批量评测；`writePerPipelineCsv(...)` 写单 pipeline CSV；`writeSummary(...)` 写汇总报告。

### 指标类

| 类 | 成员变量 | 成员函数 |
|---|---|---|
| `RepeatabilityMetric` | `_pixelThreshold`：关键点重复判断阈值。 | 构造函数读取参数；`name()` 返回 `"REPEATABILITY"`；`compute(ctx, sample)` 计算重复率。 |
| `InlierRatioMetric` | 无成员变量。 | 构造函数接收参数但不使用；`name()` 返回 `"INLIER_RATIO"`；`compute(ctx, sample)` 计算内点比例。 |
| `ReprojectionErrorMetric` | `_symmetric`：是否使用对称误差。 | 构造函数读取参数；`name()` 返回 `"REPROJECTION_ERROR"`；`compute(ctx, sample)` 计算重投影误差。 |
| `PsnrMetric` | `_maxValue`：像素最大值。 | 构造函数读取参数；`name()` 返回 `"PSNR"`；`compute(ctx, sample)` 计算 PSNR。 |
| `RmseMetric` | 无成员变量。 | 构造函数接收参数但不使用；`name()` 返回 `"RMSE"`；`compute(ctx, sample)` 计算 RMSE。 |
| `SsimMetric` | `_window`：窗口大小；`_sigma`：高斯 sigma。 | 构造函数读取参数；`name()` 返回 `"SSIM"`；`compute(ctx, sample)` 计算 SSIM。 |

## 13. Visualization

### `DrawMatches::Options`

成员变量：`draw_inliers_only` 控制是否只画内点；`max_matches` 控制最大绘制数；`match_color` 为匹配线颜色；`single_point` 为单点颜色。

### `DrawMatches`

成员函数：`render(ctx, opt)` 绘制匹配图并返回 `cv::Mat`。

### `DrawInliers::Options`

成员变量：`max_inliers` 控制最大内点数；`inlier_color` 为内点颜色；`non_inlier_color` 为外点颜色；`draw_outliers` 控制是否绘制外点。

### `DrawInliers`

成员函数：`render(ctx, opt)` 绘制内点图并返回 `cv::Mat`。

### `DrawOverlay::Options`

成员变量：`alpha` 控制叠加透明度。

### `DrawOverlay`

成员函数：`render(ctx, opt)` 绘制 warped 图与目标图的叠加结果。

### `DrawDiff::Options`

成员变量：`heatmap` 控制是否输出热力图；`scale` 控制差异缩放。

### `DrawDiff`

成员函数：`render(ctx, opt)` 绘制差异图。

### `VisualizationManager::Options`

成员变量：`draw_matches`、`draw_inliers`、`draw_overlay`、`draw_diff`、`save_warped` 控制输出类型；`max_matches` 和 `max_inliers` 控制绘制数量。

### `VisualizationManager`

成员函数：`saveAll(ctx, output_root, stem)` 使用默认选项保存全部可视化；`saveAll(ctx, output_root, stem, opt)` 使用指定选项保存全部可视化。

## 14. Utils

### `Logger`

成员变量：`_level` 保存最低日志级别；`_mu` 保证多线程输出安全。

成员函数：`instance()` 返回全局日志器；`setLevel(lv)` 设置日志级别；`level() const` 返回当前级别；`log(lv, args...)` 输出日志；`prefix(lv)` 生成日志前缀；`appendAll(...)` 拼接日志内容。

### `Timer`

成员变量：`_start` 保存计时起点。

成员函数：构造函数开始计时；`reset()` 重置起点；`elapsedMs() const` 返回 elapsed 毫秒。

### `ScopedTimer`

成员变量：`_out` 引用外部耗时变量；`_t` 为内部计时器。

成员函数：构造函数开始计时；析构函数写回耗时；拷贝构造和赋值被禁用。

### `file_utils`

函数：`ensureDirectory(dir)` 确保目录存在；`fileExists(path)` 判断文件存在；`readWholeFile(path)` 读取全文；`writeWholeFile(path, content)` 写入全文；`csvEscape(s)` 转义 CSV 字段；`makeStem(sample_name, pipeline_name)` 生成安全文件名；`listSubdirectories(root)` 枚举子目录。

### `image_utils`

函数：`toGrayFloat(src)` 转灰度浮点图；`warpedValidMask(src_size, H, dst_size)` 生成变换有效区域 mask；`nonZeroMask(warped)` 生成非零 mask；`cropToMask(a, b, mask, a_out, b_out)` 按 mask 裁剪共同有效区域。

### `yaml_utils`

函数：`get<T>(node, key, fallback)` 安全读取指定类型；`getString`、`getInt`、`getDouble`、`getFloat`、`getBool` 读取常用类型；`getVec<T>` 读取数组。

## 15. 枚举与转换函数

| API | 作用 |
|---|---|
| `FeatureType` | 特征类型枚举。 |
| `toString(FeatureType)` | 特征类型转字符串。 |
| `featureTypeFromString(std::string)` | 字符串转特征类型。 |
| `NormType` | 描述子距离类型枚举。 |
| `toString(NormType)` | 距离类型转字符串。 |
| `normTypeFromString(std::string)` | 字符串转距离类型。 |
| `toCvNorm(NormType)` | 项目距离类型转 OpenCV norm 常量。 |
| `GeometryType` | 几何模型类型枚举。 |
| `toString(GeometryType)` | 几何类型转字符串。 |
| `geometryTypeFromString(std::string)` | 字符串转几何类型。 |
| `ImageIndex` | 输入图像索引枚举。 |
| `robustMethodFromString(std::string)` | 鲁棒估计方法字符串转 OpenCV 常量。 |
| `TransformType` | 图像变换类型枚举。 |
| `toTransformType(GeometryType)` | 几何类型转变换类型。 |
| `toString(TransformType)` | 变换类型转字符串。 |
