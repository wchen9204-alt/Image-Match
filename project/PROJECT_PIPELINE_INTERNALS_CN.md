# 平台内部流水线与数据流说明

本文由粘贴的内部说明文本整理而来，目标是说明平台内部一次配准任务如何执行、数据如何在各阶段流转、YAML 如何创建算法组件，以及结构线匹配为什么能复用通用几何估计流程。

本文偏向开发者阅读，和 `EXPERIMENT_PLATFORM_DEVELOPMENT_USAGE_REPORT_CN.md` 的导师汇报定位不同。

## 1. 一次完整配准的执行顺序

平台的公共执行顺序由 `BasePipeline::run()` 定义。无论点特征法、结构法、直接法还是深度学习匹配法，都复用这套主流程，只是在部分阶段由各自的 Pipeline 子类实现不同逻辑。

```text
loadImages()
  -> runExtraction()
  -> runAssociation()
  -> runEstimation()
  -> runWarp()
  -> validateWarpQuality()
  -> saveOutputs()
```

任一关键阶段返回 `false` 时，本次运行会被判定为失败；平台会写入失败原因，并尽量保存已经产生的输出，便于调试。

## 2. 七个阶段说明

### 2.1 loadImages：图像读取

对应文件：`src/pipeline/base_pipeline.cpp`

主要职责：

1. 检查 `ctx.image1_path` 和 `ctx.image2_path` 是否为空、文件是否存在。
2. 读取 source 和 target 图像。
3. 对普通图像读取 BGR 图和 8-bit 灰度图。
4. 对 TIFF 等高位深图像使用 `IMREAD_UNCHANGED`，先保留原始位深，再归一化到 8-bit 灰度图。
5. 写入 `ctx.images.first`、`ctx.images.first_gray`、`ctx.images.second`、`ctx.images.second_gray`。

这样设计的原因是：后续关键点、结构、直接法和学习方法都可以读取统一格式的灰度图，不需要每个算法重复处理图像通道和位深；BGR 图则保留给可视化、warping 和 blend 输出使用。

### 2.2 runExtraction：提取阶段

不同方法族在该阶段的含义不同。

| 方法族 | 执行内容 | 写入数据 |
|---|---|---|
| 点特征法 | 调用 `IKeypointExtractor::extract(ctx)`，检测关键点并计算描述子。 | `ctx.keypoint_data` |
| 结构法 | 调用 `IStructureExtractor::extract(ctx)`，提取直线或轮廓。 | `ctx.structure_data` |
| 直接法 | 当前无独立提取阶段，主要清空和初始化统计值。 | `ctx.result` |
| 学习法 | 当前无独立 C++ 提取阶段，匹配点由 Python matcher 在下一阶段产生。 | `ctx.result` |

点特征提取后，`KeypointImageData` 主要包含：

```text
keypoints     vector<cv::KeyPoint>
descriptors   cv::Mat
type          KeypointType
norm_type     NormType
```

结构提取后，`StructureImageData` 按结构类型保存：

| 结构类型 | 输出内容 |
|---|---|
| Line | 线段数组 `lines` 和线响应图 |
| Contour | 轮廓数组 `contours` 和轮廓响应图 |

### 2.3 LineExtractor 内部流程

对应文件：`src/structure/line_extractor.cpp`

直线结构提取支持多种检测方法：

| 方法 | OpenCV 接口 | 输出特点 |
|---|---|---|
| `HOUGH_LINES` | `cv::HoughLines` + 裁剪 | 标准霍夫直线，需要裁剪到图像边界。 |
| `HOUGH_LINES_P` | `cv::HoughLinesP` | 概率霍夫，直接输出有限线段。 |
| `LSD` | 优先使用 `cv::line_descriptor::LSDDetector`，必要时回退到 `cv::LineSegmentDetector` | 端点精度较高，适合线描述子。 |
| `FLD` | `cv::ximgproc::FastLineDetector` | 快速线段检测，需要 OpenCV ximgproc。 |

线段后处理包括：

1. 去重：按长度排序，依据角度差和线段间距剔除重复线段。
2. 限数：保留最长的若干线段，默认上限由配置控制。
3. 渲染：在线响应图上绘制线段，供结构匹配和可视化使用。

## 3. runAssociation：匹配与关联

### 3.1 点特征法

点特征法先执行描述子匹配，再执行过滤链。

```text
matcher->match(ctx)
  -> ctx.keypoint_match_data.raw_matches_by_query
  -> filter1.apply(ctx)
  -> filter2.apply(ctx)
  -> ctx.keypoint_match_data.filtered_matches
```

当前匹配器包括：

| 匹配器 | 说明 |
|---|---|
| `BfMatcher` | 暴力匹配，支持 L1、L2、Hamming、Hamming2。 |
| `FlannMatcher` | FLANN 近似最近邻，适合浮点描述子和较大规模匹配。 |

### 3.2 结构法

结构法先由关联器生成结构匹配，再按需执行通用过滤链。

```text
associator->associate(ctx)
  -> ctx.structure_match_data.raw_matches_knn / filtered_matches
  -> filter.apply(ctx)
  -> ctx.structure_match_data.line_matches
```

常见结构关联器：

| 关联器 | 原理 | 适用场景 |
|---|---|---|
| `PhaseCorrelateAssociator` | 频域相位相关 | 平移为主的轮廓响应图。 |
| `ChamferAssociator` | 距离变换 + 平均距离 | 轮廓响应图。 |
| `HausdorffAssociator` | 分位数 Hausdorff 距离 | 含噪声结构点集。 |
| `IcpAssociator` | 迭代最近点 | 结构点集。 |
| `LineSegmentAssociator` | 角度、长度、中心位移几何投票 | 线段几何 baseline。 |
| `LineDescriptorAssociator` | LBD/MSLD/Line-SIFT 等线描述子 + 几何一致性 | 线段描述子匹配。 |
| `ContourDescriptorAssociator` | Hu、Shape Context、FD、EFD 等轮廓描述子 | 轮廓形状匹配。 |

### 3.3 直接法

直接法的核心计算发生在 `DirectPipeline::runEstimation()` 中，而不是传统意义上的描述子匹配阶段。`IDirectAligner::align(ctx)` 会直接估计变换，或者生成可视化和统计用点对。

```text
DirectPipeline::runEstimation()
  -> _aligner->align(ctx)
  -> ctx.direct_data
  -> ctx.geometry_data
```

直接法输出通过 `DirectData` 保存，包括仿射矩阵、单应矩阵、光流、点对、伪匹配、内点掩码、score 和诊断指标。

### 3.4 学习法

学习法通过 `PythonLearningMatcher` 调用 Python 后端，读取统一 matches JSON，再把结果转换为 C++ 侧的点对数据。

```text
PythonLearningMatcher::match(ctx)
  -> Python backend
  -> matches JSON
  -> ctx.keypoint_data
  -> ctx.keypoint_match_data
```

后续几何估计、warping、评测和输出都复用 C++ 主流程。

## 4. IFilter 过滤链

过滤器都实现 `IFilter::apply(RegistrationContext& ctx)`，通过读取 `RegistrationContext` 中的匹配数据进行筛选。

| 过滤器 | 原理 | 点特征 | 结构线匹配 |
|---|---|:---:|:---:|
| `RatioTest` | Lowe ratio test，要求最近邻显著优于次近邻。 | 支持 | 支持 |
| `DistanceThresholdFilter` | 描述子距离不超过固定阈值。 | 支持 | 支持 |
| `MinDistanceFilter` | 根据最小距离和倍率自适应阈值。 | 支持 | 支持 |
| `DistanceDistributionFilter` | 根据均值、标准差或分位数过滤。 | 支持 | 支持 |
| `CrossCheck` | 双向一致性检查。 | 支持 | 依赖描述子矩阵场景，结构法中需谨慎使用。 |
| `GmsFilter` | 网格运动统计。 | 支持 | 主要面向关键点坐标。 |

过滤器链的意义是把算法专属的候选匹配和通用质量筛选分开。关联器负责产出候选，过滤器负责按配置继续筛选。

## 5. runEstimation：几何估计

### 5.1 点特征法

点特征法在 `KeypointPipeline::runEstimation()` 中调用几何估计器：

```text
ctx.correspondence_source = "KEYPOINT"
_geometry->estimate(ctx)
```

几何估计器从 `ctx.keypoint_data` 和 `ctx.keypoint_match_data.filtered_matches` 中读取点对，通过 RANSAC 或 OpenCV 估计接口得到：

```text
ctx.geometry_data.H   3x3 单应矩阵
ctx.geometry_data.A   2x3 仿射族矩阵
ctx.geometry_data.inlier_mask
ctx.geometry_data.num_inliers
ctx.geometry_data.inlier_ratio
```

### 5.2 结构法

当前结构法已经通过 `CorrespondenceView` 统一对应点来源，不再需要把线段端点临时伪装成 keypoint match。

```text
StructurePipeline::runEstimation()
  -> ctx.correspondence_source = "STRUCTURE"
  -> buildStructureCorrespondenceView(ctx)
  -> _geometry->estimate(ctx)
  -> promoteStructureInliersFromGeometryMask(ctx)
```

其中，结构匹配会被转换成几何估计器可读的点对视图：

| 结构类型 | 点对来源 |
|---|---|
| Line | 线段两个端点，必要时处理方向一致性。 |
| Contour | 轮廓质心或描述子匹配对应点。 |
| Response | 结构响应图关联得到的点或平移结果。 |

几何估计后，`promoteStructureInliersFromGeometryMask()` 会把点级内点提升回结构级内点。例如线段匹配要求两个端点都成为 RANSAC 内点，才认为这条线是内点线匹配。

### 5.3 直接法和学习法

直接法显式设置：

```text
ctx.correspondence_source = "DIRECT"
```

学习法显式设置：

```text
ctx.correspondence_source = "LEARNING"
```

这样几何估计和可视化阶段可以根据来源读取不同数据域，避免把深度学习点对或直接法点对误判为传统关键点来源。

## 6. runWarp：图像变换

对应文件：`src/pipeline/base_pipeline.cpp`

`BasePipeline::runWarp()` 根据 `GeometryData` 中的模型类型选择变换方式：

| 几何类型 | Warper | OpenCV 接口 |
|---|---|---|
| `HOMOGRAPHY` | `PerspectiveWarper` | `warpPerspective` |
| `AFFINE` | `AffineWarper` | `warpAffine` |
| `RIGID` | `AffineWarper` | `warpAffine` |
| `SIMILARITY` | `AffineWarper` | `warpAffine` |

如果配置关闭 `warp`，或几何结果不可用于重采样，则该阶段跳过。

## 7. validateWarpQuality：质量验证

该阶段用于判断配准结果是否只是“算出了矩阵”，还是确实把图像对齐到了可接受程度。

### 7.1 Containment 检查

warp overlap 现在使用 `min_containment` 做前景局部包含率验证，适合一张图可能是另一张图局部的场景。

计算方式是：

```text
containment =
  intersection(warped source foreground, target foreground)
  / min(source 原始前景面积, target 前景面积)
```

这里的 source 面积使用 warp 前的原始前景面积，而不是被目标画布裁剪后的 warped 面积。这样可以避免“只贴上一小截局部区域”时因为 warped 面积已经变小而得到虚高分。

当前配置写法：

```yaml
validation:
  warp_overlap:
    enabled: true
    min_containment: 0.70
    foreground_threshold: 10
```

### 7.2 NMAD 检查

NMAD 用于衡量重叠区域的归一化平均绝对灰度差。

```text
NMAD = mean(abs(warped - target)) / 255
```

配置项：

```yaml
validation:
  photometric:
    enabled: true
    max_nmad: 0.15
```

任一启用的质量检查不达标，本次配准会被判定为失败。两项都关闭时，该阶段直接通过。

### 7.3 Bidirectional Coverage 检查

`warp bidirectional coverage` 用于衡量局部图场景下，是否至少有一个方向能说明“小图被完整保留”。

它会输出三个量：

- `source_coverage`
- `target_coverage`
- `bidirectional_coverage = max(source_coverage, target_coverage)`

具体计算方式是：

```text
source_coverage =
  countNonZero(warp(source foreground mask) 落在 target 画布内的部分)
  / countNonZero(source foreground mask)

target_coverage =
  countNonZero(inverse-warp(target foreground mask) 落在 source 画布内的部分)
  / countNonZero(target foreground mask)

bidirectional_coverage =
  max(source_coverage, target_coverage)
```

这和 `min_containment` 不是一回事：

- `min_containment` 看的是 warped source 与 target 的交集能否覆盖较小前景区域。
- `bidirectional coverage` 看的是 source / target 是否至少有一个方向在 warp 后仍被完整保留，适合发现“只贴上局部一截”的错误解。

它的用途尤其适合点特征法里的 `Test04`、`Test06` 这类样本：

- 如果 source 是 target 的局部，通常 `source_coverage` 会高。
- 如果 target 是 source 的局部，通常 `target_coverage` 会高。
- 如果错误模型只是把左右端局部平移贴上，两个方向的 coverage 往往都会偏低。

当前配置里可这样写：

```yaml
validation:
  warp_overlap:
    enabled: true
    min_containment: 0.70
    min_bidirectional_coverage: 0.80
    foreground_threshold: 10
```

现版本实现中，`source_coverage`、`target_coverage`、`bidirectional_coverage` 和 `foreground_threshold` 共用同一套前景 mask 语义，不把黑背景算进去。

## 8. saveOutputs：输出保存

输出分为两层：Pipeline 负责保存图像类结果，`RegistrationApp` 负责保存运行摘要和 CSV 统计表。

`BasePipeline::saveOutputs()` 保存通用图像输出：

| 目录/文件 | 内容 |
|---|---|
| `originals/` | 原始 source 和 target。 |
| `warped/` | 变换后的 source。 |
| `blend/` | warped source 与 target 的叠加图。 |
| `false_color_overlay/` | 伪彩色配准误差叠加图：warped source 为红色，target 为绿色，重合区域为黄色。 |

方法族专属输出包括：

| 方法族 | 典型目录 | 内容 |
|---|---|---|
| 点特征法 | `keypoints/`、`all_match/`、`inlier_match/` | 关键点图、全部匹配图、内点匹配图。 |
| 结构法 | `structures/`、`matches/` | 结构响应图、结构匹配图。 |
| 直接法 | `direct/` | 直接法点对、warp 差异热力图。 |
| 学习法 | `learning/` | 学习匹配图、学习内点图。 |

`RegistrationApp` 在 pipeline 运行结束后保存摘要和统计表：

| 文件 | 写入位置 | 内容 |
|---|---|---|
| `run_summary.txt` | 当前样本输出目录 | 单次运行文本摘要。 |
| `run_summary.json` | 当前样本输出目录 | 单次运行 JSON 摘要。 |
| `summary.csv` | 单次输出目录或批量 pipeline 汇总目录 | 单次/批量统计表。 |
| `comparison.csv` | compare 输出目录 | 多方法横向对比统计表。 |

### 8.1 直接法 summary.csv 字段说明

直接法的 `summary.csv` 由 `apps/summary_csv_writer.cpp` 写出。当前表头这一行改为中文，
并且不再输出 `feature_initializer_attempted`、`feature_initializer_used`、
`feature_initializer_method`、`num_correspondences` 和
`feature_initializer_warp_edge_alignment_iou`；这些字段要么偏流程诊断，要么不再属于当前
直接法汇总表的核心判读信息。

| 字段 | 作用 |
|---|---|
| `sample_name` / `样本名` | 样本名称，通常来自数据集子目录名或单次运行的 source/target 文件名组合。 |
| `success` / `是否成功` | 本样本最终是否通过配准与验证；`1` 表示成功，`0` 表示失败。 |
| `message` / `结果说明` | 成功或失败原因说明，便于定位是哪一阶段或哪条质量门槛影响结果。 |
| `direct_confidence` / `直接法置信度` | 直接法算法自身的置信度、响应值或 score；不同直接法的具体含义由 aligner 定义。 |
| `final_validation_source` / `最终采用来源` | 最终验证采用的结果来源，常见为 `DIRECT` 或 `INITIALIZER`；为空表示没有可接受的最终结果。 |
| `feature_initializer_inliers` / `初始值内点数` | 已接受点特征初始值的内点数；没有可用初始值时通常为 `0`。 |
| `feature_initializer_inlier_ratio` / `初始值内点率` | 已接受点特征初始值的内点率；不可用时为 `-1`。 |
| `feature_initializer_spatial_coverage` / `初始值空间覆盖率` | 已接受点特征初始值的内点空间覆盖率，用于判断初值点分布是否足够分散；不可用时为 `-1`。 |
| `feature_initializer_warp_photometric_error` / `初始值光度误差` | 已接受点特征初始值临时 warp 后的 NMAD 灰度误差；越小越好，不可用时为 `-1`。 |
| `warp_overlap_containment` / `重叠包含率` | 最终 warped source 与 target 的局部包含率，用于局部图场景下判断较小前景是否被覆盖。 |
| `warp_source_coverage` / `源图覆盖率` | source 前景经最终变换后仍落在 target 画布内的比例。 |
| `warp_target_coverage` / `目标图覆盖率` | target 前景经逆变换后仍落在 source 画布内的比例。 |
| `warp_bidirectional_coverage` / `双向覆盖率` | `warp_source_coverage` 与 `warp_target_coverage` 的较大值，用于双向覆盖质量判断。 |
| `warp_edge_alignment_iou` / `边缘对齐IoU` | 最终 warped source 与 target 在重叠区域内的边缘对齐 IoU；越大表示边缘越一致。 |
| `warp_photometric_error` / `光度误差` | 最终 warped source 与 target 重叠区域的 NMAD 灰度误差；越小表示光度越一致。 |
| `t_load_ms` / `加载耗时_ms` | 图像读取和预处理耗时，单位毫秒。 |
| `t_geometry_ms` / `几何阶段耗时_ms` | 直接法估计、几何同步和相关估计阶段耗时，单位毫秒。 |
| `t_warp_ms` / `变换耗时_ms` | 最终图像 warp 阶段耗时，单位毫秒。 |
| `t_total_ms` / `总耗时_ms` | 本样本完整流水线总耗时，单位毫秒。 |
| `metric_*` / `指标_*` | Evaluator 动态追加的评估指标列，例如 `metric_REPEATABILITY` 或 `指标_REPEATABILITY`；只有当前批次中至少出现过一次有效值的指标才会写入。 |

历史批次的旧 `summary.csv` 在删除三列后仍可能保留下面这些列；它们不属于当前新生成的直接法核心 schema，但含义如下：

| 旧字段 | 作用 |
|---|---|
| `feature_initializer_candidates` | 旧版记录的点特征初始化候选数量，用来观察初始化器尝试了多少候选。 |
| `feature_initializer_accepted` | 旧版记录是否存在通过初始值阶段自检的点特征初始化结果；`1` 表示通过，`0` 表示未通过。 |
| `num_correspondences` | 旧版记录的直接法对应点数量；稀疏/稠密直接法中更有意义，当前新表已删除。 |
| `mean_reproj_error` | 旧版沿用通用几何重投影误差；对多数直接法没有稳定通用语义，当前直接法新表不再输出。 |
| `inlier_spatial_coverage` | 旧版沿用通用内点空间覆盖率；当前直接法主判定以 warp 质量为主，新表不再输出该列。 |
| `feature_initializer_warp_edge_alignment_iou` | 旧版记录初始值临时 warp 的边缘对齐 IoU；当前新表已删除。 |
| `t_align_ms` | 旧版直接法对齐耗时列；历史实现中实际写入的是 `t_match_ms`，含义不够准确，当前新表不再输出。 |

批量输出路径形式：

```text
outputs/batch/<method_family>/<pipeline>/<sample>/
```

单次输出路径形式：

```text
outputs/single/<method_family>/<pipeline>/
```

## 9. RegistrationContext 数据流

`RegistrationContext` 是贯穿整条流水线的数据总线。各阶段不直接互相调用，而是通过它读写中间结果。

```text
ImagePairData
  -> KeypointData / StructureData / DirectData
  -> KeypointMatchData / StructureMatchData / CorrespondenceView
  -> GeometryData
  -> TransformData
  -> EvaluationData
  -> RegistrationResult
```

关键结构速查：

| 结构 | 核心字段 | 写入阶段 | 读取阶段 |
|---|---|---|---|
| `ImagePairData` | BGR 图、灰度图 | `loadImages` | 后续所有阶段 |
| `KeypointData` | keypoints、descriptors、type、norm_type | 点特征提取、学习匹配 | 匹配、几何估计、可视化 |
| `KeypointMatchData` | raw_matches_by_query、filtered、inlier_mask、inliers | 匹配、过滤、几何估计 | 几何估计、输出 |
| `StructureData` | response、lines、contours | 结构提取 | 结构关联、输出 |
| `StructureMatchData` | raw_matches_knn、filtered_matches、line_matches、inlier_line_matches、translation、affine、score | 结构关联、过滤、几何估计 | 几何估计、输出 |
| `DirectData` | A、H、flow、points1、points2、matches、inlier_mask、score、diagnostics | 直接法 aligner | 几何同步、可视化、摘要 |
| `GeometryData` | type、H、A、inlier_mask、num_inliers、inlier_ratio | 几何估计 | warp、验证、摘要 |
| `TransformData` | type、M、valid | warp | 验证、摘要 |
| `EvaluationData` | metrics | Evaluator | 摘要、CSV |
| `RegistrationResult` | success、message、数量统计、质量指标、耗时 | 各阶段 | 终端输出、摘要、CSV |

每次 `BasePipeline::run()` 开始时会调用 `ctx.reset()` 清空运行时数据，避免批量运行时前一个样本污染后一个样本。

## 10. Factory 与 YAML

平台使用 Factory 模式创建算法组件。业务流程只依赖接口，不直接构造具体类。

常见创建函数包括：

```cpp
Factory::createKeypointExtractor(...)
Factory::createStructureExtractor(...)
Factory::createStructureAssociator(...)
Factory::createMatcher(...)
Factory::createFilter(...)
Factory::createDirectAligner(...)
Factory::createGeometryEstimator(...)
```

YAML 中的 `type` 或 `method` 字段会被归一化为稳定 key，再映射到具体实现类。这样配置可以写得相对宽松，例如大小写和下划线差异不会轻易导致无法识别。

配置加载的基本流程：

```text
RegistrationApp::run()
  -> Config::loadPipeline(pipeline_yaml)
  -> 解析 method_family / io / visualization / validation / filters
  -> resolvePath() 得到各子配置绝对路径
  -> createPipelineForConfig()
  -> pipeline->configure()
  -> Factory 创建各阶段组件
```

## 11. LineDescriptorAssociator 说明

`LineDescriptorAssociator` 是线结构法中的线描述子匹配器，适合说明结构法如何从线段走到通用几何估计。

完整流程：

```text
LineExtractor
  -> ctx.structure_data.first.lines / second.lines
  -> Vec4i 转 KeyLine
  -> 计算线描述子
  -> KNN 或 MATCH 匹配
  -> 方向、长度、位移一致性过滤
  -> 写入 StructureMatchData
```

几何一致性过滤的核心是：真实匹配线段在方向、长度比例和中心位移上应当具有一致性。描述子匹配提供候选，几何一致性负责剔除明显不合理的候选。

常用参数位于 `configs/structure/line.yaml`：

```yaml
association:
  params:
    line_descriptor:
      descriptor: LBD
      matcher: BF
      match_mode: MATCH
      min_matches: 2
      geometric_filter: true
      angle_threshold_deg: 30.0
      min_length_ratio: 0.30
      shift_consistency_threshold: 30.0
```

运行时可重点观察以下日志：

```text
LineDescriptorAssociator input: srcLines=N, dstLines=M
LineDescriptorAssociator LBD: srcDesc=R x C (keys=K), dstDesc=...
LineDescriptorAssociator KNN: raw_matches_knn groups=G
```

如果 `keys` 明显小于输入线段数，说明部分线段在转换或描述子计算时被过滤；如果 `raw_matches_knn` 很少，说明描述子可匹配性较差或阈值过严。

## 12. StructurePipeline 几何路径

当前结构法几何估计路径的核心是 `CorrespondenceView`。

```text
StructurePipeline::runEstimation()
  -> buildStructureCorrespondenceView(ctx)
  -> _geometry->estimate(ctx)
  -> promoteStructureInliersFromGeometryMask(ctx)
```

这意味着结构法不再由关联器内部硬编码某个几何模型，而是由 pipeline YAML 中的 `geometry` 配置决定使用 Rigid、Affine、Similarity 还是 Homography。

结构法的好处是：结构关联器只负责产生结构对应关系，几何估计器只负责根据对应点估计矩阵，两者职责分离。

## 13. 批量运行流程

批量运行由 `RegistrationApp::runBatch()` 调度。

```text
runBatch(batch_yaml)
  -> loadBatchConfig()
  -> Config::loadPipeline(batch.pipeline_yaml)
  -> DatasetLoader::load()
  -> for each sample:
       createPipelineForConfig()
       pipeline->configure(base_cfg)
       设置 sample 的 image1/image2/output_dir
       pipeline->run(ctx)
       收集 RegistrationResult 和 EvaluationData
  -> writeSummaryCsv()
```

批量配置通常包含：

```yaml
pipeline: ../keypoint/sift_pipeline.yaml

dataset:
  root: ../../../datasets
  pattern_sources: [source, moving]
  pattern_targets: [target, reference]
  include: []

output:
  root: ../../../outputs
  save_visuals: true
  summary_csv: true
```

`include` 为空时会扫描数据集根目录下全部样本；填写样本名时只运行指定样本。

