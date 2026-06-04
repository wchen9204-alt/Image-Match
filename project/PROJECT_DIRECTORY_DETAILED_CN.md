# 项目目录结构详细说明

本文按项目目录顺序说明文件功能、主要类、核心函数以及实现逻辑。适合代码走读、项目交接和向导师解释工程组织方式。

## 1. 根目录

| 文件或目录 | 作用 |
|---|---|
| `CMakeLists.txt` | CMake 构建脚本，配置 C++17、OpenCV、yaml-cpp、核心静态库和可执行程序。 |
| `main.cpp` | 程序入口，创建 `RegistrationApp` 并转交命令行参数。 |
| `README.md` | 英文版项目简介、构建方式和运行示例。 |
| `PROJECT_QUICK_INTRO_CN.md` | 中文快速介绍文档。 |
| `API_REFERENCE_CN.md` | 中文 API 查阅文档。 |
| `apps/` | 命令行应用层。 |
| `configs/` | YAML 配置目录。 |
| `datasets/` | 实验样本目录。 |
| `include/` | 头文件目录，定义公共接口和类声明。 |
| `src/` | 源文件目录，存放具体实现。 |
| `outputs/` | 实验输出目录。 |

### `main.cpp`

功能：最小化入口文件。

核心逻辑：

```text
main(argc, argv)
  -> ir::RegistrationApp app
  -> app.run(argc, argv)
```

它不直接处理图像或算法，只负责启动应用层，保证主流程集中在 `RegistrationApp` 中。

## 2. `apps/`

### `registration_app.h / registration_app.cpp`

功能：命令行应用入口，负责区分单次实验和批量实验。

核心类：`RegistrationApp`

主要职责：

| 函数 | 功能 | 核心逻辑 |
|---|---|---|
| `run(int argc, char** argv)` | 从命令行参数启动程序 | 解析 pipeline YAML、可选图像路径和输出目录 |
| `run(const Args& args)` | 根据已解析参数运行 | 先判断 YAML 是单次配置还是批处理配置 |
| `runSingle(...)` | 单样本配准 | 加载 pipeline、创建 pipeline 对象、执行、保存摘要和 CSV |
| `runBatch(...)` | 批量配准 | 读取批处理配置、扫描数据集、逐样本执行同一 pipeline |
| `loadBatchConfig(...)` | 读取批处理 YAML | 解析 dataset、pipeline、output 配置 |
| `writeSummaryCsv(...)` | 写出 CSV 汇总 | 把每个样本的 `RegistrationResult` 展开成表格 |
| `buildOutputDir(...)` | 生成输出目录 | 按 `single/batch + 方法族 + pipeline + 样本名` 分层 |

重要设计：

1. 应用层不直接实现算法，只选择 `KeypointPipeline` 或 `StructurePipeline`。
2. 单次与批量复用同一条 pipeline，批量只替换输入图像与输出目录。
3. 输出路径由应用层统一生成，避免各模块各自拼路径。

## 3. `configs/`

配置目录把实验参数从代码中分离出来，是平台可扩展性的关键。

### `configs/keypoint/`

包含点特征提取器配置：

| 文件 | 方法 |
|---|---|
| `sift.yaml` | SIFT |
| `surf.yaml` | SURF |
| `orb.yaml` | ORB |
| `brisk.yaml` | BRISK |
| `kaze.yaml` | KAZE |
| `akaze.yaml` | AKAZE |

这些文件通常包含：

| 字段 | 作用 |
|---|---|
| `type` | 特征类型 |
| `descriptor_norm` | 描述子默认距离类型 |
| `params` | OpenCV 构造参数 |

### `configs/matcher/`

| 文件 | 功能 |
|---|---|
| `bf.yaml` | 暴力匹配器，支持 match / knn / radius |
| `flann.yaml` | FLANN 近似最近邻匹配 |

### `configs/filter/`

| 文件 | 功能 |
|---|---|
| `ratio_test.yaml` | Lowe ratio test |
| `cross_check.yaml` | 双向一致性检查 |
| `gms.yaml` | GMS 网格运动统计过滤 |
| `min_distance.yaml` | 最小距离比例过滤 |
| `distance_threshold.yaml` | 固定距离阈值过滤 |
| `distance_distribution.yaml` | 基于距离分布的自适应过滤 |

### `configs/geometry/`

| 文件 | 几何模型 |
|---|---|
| `homography.yaml` | 单应矩阵 |
| `affine.yaml` | 仿射变换 |
| `rigid.yaml` | 刚体变换 |
| `similarity.yaml` | 相似变换 |

### `configs/structure/`

| 文件 | 功能 |
|---|---|
| `edge.yaml` | 边缘结构配置 |
| `line.yaml` | 直线结构配置，包括线提取器和线关联器 |
| `contour.yaml` | 轮廓结构配置 |

`line.yaml` 当前包含：

1. `extractor`：Hough、LSD、FLD 等线检测方式。
2. `association`：LINE_DESCRIPTOR、LINE_SEGMENT、PHASE_CORRELATE 等线关联方式。
3. `line_descriptor`：LBD 参数、KeyLine 来源、KNN 和几何一致性过滤阈值。

### `configs/pipeline/`

| 子目录 | 功能 |
|---|---|
| `keypoint/` | 点特征单次 pipeline |
| `structure/` | 结构法单次 pipeline |
| `batch/` | 批处理入口 |

单次 pipeline 负责组合子配置，批处理配置负责指定复用哪个单次 pipeline 和扫描哪个数据集。

## 4. `include/core/` 与 `src/core/`

核心模块保存全局类型、配置、上下文、工厂和顶层配准类。

### `types.h / types.cpp`

功能：定义项目通用枚举和字符串转换。

核心内容：

| 类型 | 功能 |
|---|---|
| `KeypointType` | SIFT、SURF、ORB、BRISK、KAZE、AKAZE |
| `StructureType` | EDGE、LINE、CONTOUR |
| `NormType` | L1、L2、HAMMING、HAMMING2 |
| `MatchMethod` | MATCH、KNN、RADIUS |
| `GeometryType` | HOMOGRAPHY、AFFINE、RIGID、SIMILARITY |
| `ImageIndex` | 第一张图或第二张图 |

核心函数：

| 函数 | 功能 |
|---|---|
| `toString(...)` | 枚举转字符串 |
| `xxxFromString(...)` | YAML 字符串转枚举 |
| `toCvNorm(NormType)` | 转成 OpenCV 距离范数 |
| `robustMethodFromString(...)` | RANSAC、LMEDS 等字符串转 OpenCV 参数 |

### `config.h / config.cpp`

核心类：`Config`

功能：加载 YAML，并解析 pipeline 中引用的相对路径。

核心函数：

| 函数 | 功能 | 逻辑 |
|---|---|---|
| `load(path)` | 读取 YAML | 调用 yaml-cpp，失败时抛异常 |
| `resolvePath(base, value)` | 路径解析 | 绝对路径直接使用，相对路径基于 YAML 所在目录 |
| `loadPipeline(path)` | 加载 pipeline | 读取 keypoint/structure/matcher/filter/geometry/io/visualization |

核心结构：`PipelineConfig`

保存 pipeline 名称、子配置路径、输入输出路径、可视化开关和 warp 质量校验参数。

### `context.h`

核心类：`RegistrationContext`

功能：整条流水线共享的运行上下文。

主要成员：

| 成员 | 作用 |
|---|---|
| `images` | source / target 原图和灰度图 |
| `keypoint_data` | 点特征提取结果 |
| `keypoint_match_data` | 点特征匹配和过滤结果 |
| `structure_data` | 结构提取结果 |
| `structure_match_data` | 结构关联结果 |
| `geometry_data` | 几何估计结果 |
| `evaluation` | 评测指标 |
| `result` | 运行摘要 |
| `warped_image` | warp 后图像 |

核心函数：`reset()`，清空一次运行的中间数据。

### `factory.h / factory.cpp`

核心类：`Factory`

功能：根据 YAML 中的 `type` 或 `method` 创建具体算法对象。

主要创建函数：

| 函数 | 创建对象 |
|---|---|
| `createKeypointExtractor` | SIFT、SURF、ORB、BRISK、KAZE、AKAZE |
| `createStructureExtractor` | Edge、Line、Contour |
| `createStructureAssociator` | PhaseCorrelate、Chamfer、Hausdorff、ICP、LineSegment、LineDescriptor |
| `createMatcher` | BF、FLANN |
| `createFilter` | Ratio、CrossCheck、GMS 等 |
| `createGeometryEstimator` | Homography、Affine、Rigid、Similarity |

核心逻辑：把 YAML 字符串归一化后分发到具体类，新增算法时需要在这里注册。

### `registration.h / registration.cpp`

核心类：`Registration`

功能：顶层配准封装，持有 `IPipeline`，提供统一 `configure/run` 风格接口。当前命令行应用主要直接使用 pipeline，此类为后续库式调用预留。

### `result.h`

结构：`RegistrationResult`

功能：保存运行摘要，包括成功状态、消息、特征数量、匹配数量、内点数量、质量指标和各阶段耗时。

## 5. `include/data/`

数据模块定义各阶段的输入输出结构。

| 文件 | 结构 | 功能 |
|---|---|---|
| `image_data.h` | `ImagePairData` | 保存两张图的彩色图和灰度图 |
| `keypoint_data.h` | `KeypointImageData`, `KeypointData` | 保存关键点、描述子、类型和距离范数 |
| `keypoint_match_data.h` | `KeypointMatchData` | 保存 KNN 原始匹配、过滤匹配、内点匹配 |
| `structure_data.h` | `StructureImageData`, `StructureData` | 保存结构响应图、线段、轮廓和结构类型 |
| `structure_match_data.h` | `StructureMatchData` | 保存结构平移、仿射、线段匹配、内点线段匹配 |
| `geometry_data.h` | `GeometryData` | 保存 H、A、几何类型、内点数、内点比例和状态 |
| `transform_data.h` | `TransformData` | 保存统一变换矩阵和类型 |
| `evaluation_data.h` | `MetricResult`, `EvaluationData` | 保存评测指标集合 |

这些结构一般只保存数据和 `clear()` 方法，不直接实现复杂算法。

## 6. `include/interfaces/`

接口层定义模块契约，使 pipeline 可以通过多态调用不同算法。

| 接口 | 核心函数 | 功能 |
|---|---|---|
| `IKeypointExtractor` | `extract(ctx)` | 检测关键点并计算描述子 |
| `IMatcher` | `match(ctx)` | 根据描述子生成候选匹配 |
| `IFilter` | `apply(ctx)` | 过滤候选匹配 |
| `IGeometryEstimator` | `estimate(ctx)` | 根据匹配估计几何模型 |
| `IStructureExtractor` | `extract(ctx)` | 提取边缘、线、轮廓等结构 |
| `IStructureAssociator` | `associate(ctx)` | 对结构响应或结构元素建立关联 |
| `IPipeline` | `configure/run/showWindows` | 完整流水线接口 |
| `IRegistration` | `configure/run` | 顶层配准接口 |

接口的意义：新增算法只要实现对应接口，就能被 Factory 注册并接入 pipeline。

## 7. `include/pipeline/` 与 `src/pipeline/`

### `BasePipeline`

功能：通用配准骨架。

核心函数：

| 函数 | 作用 | 逻辑 |
|---|---|---|
| `configure(cfg)` | 保存配置并创建公共 warper | 调用子类 `configureStages` |
| `run(ctx)` | 执行完整流程 | reset -> load -> extract -> associate -> estimate -> warp -> validate -> save |
| `loadImages(ctx)` | 读图 | 支持普通图与高位深图，准备灰度图 |
| `runWarp(ctx)` | 执行图像变换 | 检查几何类型和有效性后调用 warper |
| `validateWarpQuality(ctx)` | 可选重合校验 | 计算 warped source 与 target 前景 IoU |
| `saveOutputs(ctx)` | 保存通用输出 | originals、warped、blend |
| `showWindows(ctx)` | 显示窗口 | 根据配置调用 OpenCV 窗口 |

### `KeypointPipeline`

功能：点特征配准流水线。

成员：

| 成员 | 作用 |
|---|---|
| `_extractor` | 点特征提取器 |
| `_matcher` | 描述子匹配器 |
| `_filters` | 过滤器链 |
| `_geometry` | 几何估计器 |

核心函数：

| 函数 | 逻辑 |
|---|---|
| `configureStages` | 读取 keypoint、matcher、filter、geometry 配置并创建对象 |
| `runExtraction` | 调用 extractor，统计关键点数量 |
| `runAssociation` | 先匹配，再按顺序执行过滤器 |
| `runEstimation` | 调用 geometry estimator，写回内点统计 |
| `saveOutputs` | 保存关键点图、匹配图、内点图，再委托基类保存 warp 输出 |

### `StructurePipeline`

功能：结构特征配准流水线。

成员：

| 成员 | 作用 |
|---|---|
| `_extractor` | 结构提取器 |
| `_associator` | 结构关联器 |
| `_geometry` | 几何估计器，作为必要时的回退 |

核心函数：

| 函数 | 逻辑 |
|---|---|
| `configureStages` | 加载 structure 配置，创建 extractor 和 associator；加载 geometry 配置 |
| `runExtraction` | 提取结构响应并统计结构数量 |
| `runAssociation` | 执行结构关联，统计匹配数量 |
| `runEstimation` | 优先使用 associator 给出的仿射；没有仿射时把线段端点转点对再估计 |
| `saveOutputs` | 保存结构响应图和结构匹配图，再委托基类 |

重要细节：

1. 线描述子关联器会直接给出基于线段中心的平移仿射。
2. `runEstimation` 只有在 `structure_match_data.affine` 为空时才回退到端点点对估计。
3. 线段匹配图优先使用 `cv::line_descriptor::drawLineMatches`，并显式传入 matches mask。

## 8. `include/keypoint/` 与 `src/keypoint/`

点特征提取器都继承 `IKeypointExtractor`。

| 类 | 文件 | 功能 | 核心逻辑 |
|---|---|---|---|
| `SiftExtractor` | `sift_extractor.*` | SIFT 特征 | 创建 `cv::SIFT`，调用 `detectAndCompute` |
| `SurfExtractor` | `surf_extractor.*` | SURF 特征 | 创建 `cv::xfeatures2d::SURF` |
| `OrbExtractor` | `orb_extractor.*` | ORB 特征 | 创建 `cv::ORB`，输出二进制描述子 |
| `BriskExtractor` | `brisk_extractor.*` | BRISK 特征 | 创建 `cv::BRISK` |
| `KazeExtractor` | `kaze_extractor.*` | KAZE 特征 | 创建 `cv::KAZE`，输出浮点描述子 |
| `AkazeExtractor` | `akaze_extractor.*` | AKAZE 特征 | 创建 `cv::AKAZE`，支持不同描述子类型 |

共同流程：

```text
读取 YAML 参数
创建 OpenCV extractor
对 first_gray / second_gray 调用 detectAndCompute
写入 ctx.keypoint_data
设置 keypoint type 和 descriptor norm
```

## 9. `include/matcher/keypoint/` 与 `src/matcher/keypoint/`

### `BfMatcher`

功能：OpenCV BFMatcher 封装。

核心参数：

| 参数 | 作用 |
|---|---|
| `method` | MATCH、KNN、RADIUS |
| `crossCheck` | 是否启用 OpenCV 内置交叉检查 |
| `knnK` | KNN 邻居数 |
| `radius` | 半径匹配阈值 |

逻辑：

1. 根据 `ctx.keypoint_data.norm` 选择 OpenCV norm。
2. 创建 `cv::BFMatcher`。
3. 根据配置调用 `match`、`knnMatch` 或 `radiusMatch`。
4. 写入 `ctx.keypoint_match_data.raw_knn`。

### `FlannMatcher`

功能：OpenCV FLANN 匹配器封装。

逻辑：

1. 浮点描述子直接使用 FLANN。
2. 二进制描述子通常需要转换或使用合适索引。
3. 输出 KNN 匹配结果供后续过滤器处理。

## 10. `include/filter/` 与 `src/filter/`

过滤器都继承 `IFilter`，输入是 `ctx.keypoint_match_data.raw_knn` 或上一个过滤结果，输出是 `filtered`。

| 类 | 方法 | 核心逻辑 |
|---|---|---|
| `RatioTestFilter` | Lowe ratio test | 比较最近邻和次近邻距离，保留足够显著的匹配 |
| `CrossCheckFilter` | 双向一致 | source->target 与 target->source 必须互相对应 |
| `GmsFilter` | GMS | 使用 OpenCV xfeatures2d 网格运动统计剔除误匹配 |
| `MinDistanceFilter` | 最小距离比例 | 根据全局最小距离设置动态阈值 |
| `DistanceThresholdFilter` | 固定阈值 | 匹配距离小于阈值才保留 |
| `DistanceDistributionFilter` | 分布阈值 | 根据距离均值和标准差筛选 |

## 11. `include/geometry/` 与 `src/geometry/`

几何估计器都继承 `IGeometryEstimator`。

| 类 | OpenCV 方法 | 输出 |
|---|---|---|
| `HomographyEstimator` | `cv::findHomography` | 3x3 单应矩阵 H |
| `AffineEstimator` | `cv::estimateAffine2D` | 2x3 仿射矩阵 A |
| `RigidEstimator` | `cv::estimateAffinePartial2D` 后提取刚体成分 | 2x3 刚体矩阵 |
| `SimilarityEstimator` | `cv::estimateAffinePartial2D` | 旋转、统一尺度、平移 |

共同逻辑：

1. 从关键点和过滤匹配中生成 `srcPoints` / `dstPoints`。
2. 检查最小匹配数。
3. 调用 OpenCV 鲁棒估计函数，通常使用 RANSAC。
4. 根据 inlier mask 生成内点匹配。
5. 写回 `ctx.geometry_data` 和运行摘要。

`partial_affine_utils.h` 提供部分仿射矩阵辅助函数，用于刚体和相似变换的公共处理。

## 12. `include/structure/` 与 `src/structure/`

### `EdgeExtractor`

功能：提取边缘响应图。

支持方法：

| 方法 | 逻辑 |
|---|---|
| CANNY | 调用 `cv::Canny` |
| SOBEL | Sobel 梯度幅值 |
| LOG | 高斯模糊后 Laplacian |
| LAPLACIAN | 直接 Laplacian |

输出：`ctx.structure_data.first/second.response`。

### `LineExtractor`

功能：提取线段结构。

支持方法：

| 方法 | OpenCV 接口 |
|---|---|
| HOUGH_LINES | `cv::HoughLines`，无限直线裁剪到图像边界 |
| HOUGH_LINES_P | `cv::HoughLinesP` |
| LSD | `cv::createLineSegmentDetector` |
| FLD | `cv::ximgproc::createFastLineDetector` |

核心逻辑：

1. 读取 YAML 中的检测器类型和参数。
2. 对两张灰度图分别检测线段。
3. 过滤短线段。
4. 可选执行近重复线段去重。
5. 按长度保留最多 `maxLines` 条。
6. 用 `cv::line` 绘制结构响应图。

当前已优先使用 OpenCV API：

| 功能 | 使用方法 |
|---|---|
| 线段长度 | `cv::norm` |
| 点到线段距离 | `cv::pointPolygonTest` |

### `ContourExtractor`

功能：提取轮廓结构。

核心逻辑：

1. 灰度图阈值化或边缘化。
2. 调用 `cv::findContours`。
3. 根据面积、长度等参数过滤轮廓。
4. 输出轮廓集合和响应图。

## 13. `include/matcher/structure/` 与 `src/matcher/structure/`

结构关联器都继承 `IStructureAssociator`。

### `LineDescriptorAssociator`

功能：基于线描述子的线段匹配。

当前实现：

| 步骤 | 方法 |
|---|---|
| KeyLine 来源 | OpenCV `cv::line_descriptor::LSDDetector` 或外部线段转换 |
| 描述子 | `cv::line_descriptor::BinaryDescriptor`，即 LBD |
| 匹配器 | `cv::line_descriptor::BinaryDescriptorMatcher` |
| 初筛 | KNN + ratio / top-k |
| 几何一致性 | 方向差、长度比例、中点位移一致性 |
| 输出 | `line_matches`、`inlier_line_matches`、`translation`、`affine` |

注意：LBD 已使用专属匹配器，因此 YAML 中不再配置 `matcher: BF`。

### `LineSegmentAssociator`

功能：不计算描述子，仅根据线段几何属性建立 baseline 匹配。

逻辑：

1. 计算线段中心、方向、长度。
2. 根据方向差、长度比例和中点位移生成候选。
3. 用中点位移一致性投票选择稳定匹配。
4. 输出平均平移仿射。

### `PhaseCorrelateAssociator`

功能：基于结构响应图的相位相关平移估计。

逻辑：把结构响应图转为浮点图，模糊后调用 `cv::phaseCorrelate`，根据响应值判断有效性。

### `ChamferAssociator`

功能：基于距离变换的 Chamfer 匹配。

逻辑：对 target 结构响应图计算距离图，在平移搜索窗口内寻找 source 点集到 target 的平均距离最小位置。

### `HausdorffAssociator`

功能：基于 Hausdorff 距离的结构点集匹配。

逻辑：采样结构点集，在平移搜索窗口内计算正向和反向距离，并用分位数降低离群点影响。

### `IcpAssociator`

功能：基于 ICP 思想估计结构点集平移。

逻辑：采样点集，寻找最近邻对应，迭代更新平移，直到位移增量小于阈值。

### `structure_point_set`

功能：结构点集工具函数。

主要函数：

| 函数 | 功能 |
|---|---|
| `toGray32F` | 响应图转浮点灰度 |
| `toBinaryMask` | 响应图转二值 mask |
| `prepareDistanceMap` | 计算距离变换图 |
| `collectPoints` | 从响应图采样结构点 |
| `centroid` | 计算点集质心 |

## 14. `include/transform/` 与 `src/transform/`

| 类 | 功能 |
|---|---|
| `IWarper` | 图像变换接口 |
| `Warper` | 通用变换相关基类或工具 |
| `PerspectiveWarper` | 当前主要使用的 warp 实现，支持 H 和 A |
| `AffineWarper` | 仿射变换实现 |

`PerspectiveWarper` 逻辑：

1. 从 `ctx.geometry_data` 读取 H 或 A。
2. 如果是 2x3 仿射矩阵，扩展为可用于透视 warper 的形式或调用仿射 warp。
3. 调用 OpenCV 重采样函数生成 `ctx.warped_image`。

## 15. `include/dataset/` 与 `src/dataset/`

### `Sample`

保存一个样本的信息：样本名、source 路径、target 路径。

### `DatasetLoader`

功能：批量扫描数据集。

逻辑：

1. 读取 dataset root。
2. 遍历子目录。
3. 根据 `pattern_source` 和 `pattern_target` 找到配对图像。
4. 如果 `include` 非空，只保留指定样本。
5. 返回 `std::vector<Sample>`。

## 16. `include/evaluator/` 与 `src/evaluator/`

评测模块为后续系统评价预留。

| 文件 | 功能 |
|---|---|
| `evaluator.h` | 定义 `IMetric` 和 `Evaluator` |
| `benchmark.h` | benchmark 运行组织 |
| `statistics.h` | 指标统计 |
| `metrics/image/*.h` | PSNR、RMSE、SSIM |
| `metrics/geometric/*.h` | InlierRatio、ReprojectionError |
| `metrics/keypoint/*.h` | Repeatability |

典型逻辑：

1. metric 从 `RegistrationContext` 读取必要数据。
2. 计算单个指标。
3. 写入 `EvaluationData`。
4. 批量时汇总到 CSV。

## 17. `include/utils/` 与 `src/utils/`

工具模块提供通用能力。

| 文件 | 功能 |
|---|---|
| `logger.*` | 日志输出，支持 INFO、WARN、ERROR |
| `timer.*` | 计时器和作用域计时 |
| `yaml_utils.*` | YAML 参数读取工具，支持默认值 |
| `file_utils.*` | 文件读写、目录创建、CSV 转义 |
| `image_utils.*` | 灰度转换、mask、裁剪等图像工具 |
| `visualization/*` | 匹配、内点、叠加、差异等可视化工具 |

## 18. 代码阅读建议

建议按以下顺序阅读：

1. `main.cpp`
2. `apps/registration_app.cpp`
3. `include/core/config.h` 与 `src/core/config.cpp`
4. `include/pipeline/base_pipeline.h` 与 `src/pipeline/base_pipeline.cpp`
5. `src/pipeline/keypoint_pipeline.cpp`
6. `src/pipeline/structure_pipeline.cpp`
7. `src/core/factory.cpp`
8. 具体算法模块，例如 `src/keypoint/sift_extractor.cpp` 或 `src/matcher/structure/line_descriptor_associator.cpp`

这条阅读路径可以从入口、配置、调度、工厂再到算法实现，避免一开始陷入单个 OpenCV 函数细节。
