# 项目快速介绍文档

本文面向项目使用者，帮助快速理解这个图像配准项目的模块划分、运行流程和核心类职责。

## 1. 项目定位

本项目是一个基于 OpenCV 和 YAML 配置驱动的二维图像配准框架。它把一次配准任务拆成多个可替换模块：

1. 读取图像
2. 提取特征
3. 匹配描述子
4. 过滤匹配点
5. 估计几何模型
6. 变换图像
7. 保存可视化和统计结果

用户主要通过 `configs/pipeline/*.yaml` 组合算法链，例如选择 SIFT、BF-KNN、RatioTest、CrossCheck、Homography 等组件。

## 2. 整体流程

典型单次运行流程如下：

```text
main
  -> RegistrationApp
  -> Config 加载 pipeline YAML
  -> Factory 创建 feature / matcher / filter / geometry
  -> BasePipeline 执行完整流程
  -> RegistrationContext 贯穿保存中间结果
  -> outputs 输出图像、匹配图和统计信息
```

批量运行时，`RegistrationApp` 会使用 `DatasetLoader` 扫描多个样本，并对每个样本重复执行同一条 pipeline。

## 3. App 模块

作用：提供命令行入口，区分单次运行和批量运行。

核心类：

| 类 | 核心功能 |
|---|---|
| `RegistrationApp` | 解析命令行参数，加载 YAML，执行单次或批量配准，输出运行摘要和 CSV。 |

## 4. Core 模块

作用：提供项目的基础类型、配置加载、上下文、结果记录和对象工厂。

核心类与结构：

| 类/结构 | 核心功能 |
|---|---|
| `PipelineConfig` | 保存单次 pipeline YAML 解析后的配置，包括 feature、matcher、filters、geometry、输入输出路径和可视化选项。 |
| `Config` | 读取 YAML 文件，解析相对路径，生成 `PipelineConfig`。 |
| `RegistrationContext` | 整条配准流程共享的数据容器，保存图像、特征、匹配、几何、变换、评测和结果。 |
| `RegistrationResult` | 保存一次运行的统计摘要，例如关键点数、匹配数、内点数、耗时和状态信息。 |
| `Factory` | 根据 YAML 中的 `type` 创建具体算法组件。 |
| `Registration` | 顶层配准对象，持有 pipeline 并调用其执行。 |

核心枚举：

| 枚举 | 作用 |
|---|---|
| `FeatureType` | 表示特征类型：SIFT、SURF、ORB、BRISK、KAZE、AKAZE。 |
| `NormType` | 表示描述子距离类型：L1、L2、HAMMING、HAMMING2。 |
| `GeometryType` | 表示几何模型：Homography、Affine、Rigid、Similarity。 |
| `TransformType` | 表示图像变换类型：Perspective 或 Affine。 |

## 5. Data 模块

作用：定义各阶段输入输出的数据结构。

核心结构：

| 结构 | 核心功能 |
|---|---|
| `FeatureImageData` | 保存单张图像的原图、灰度图、关键点和描述子。 |
| `FeatureData` | 保存两张图像的特征结果，以及特征类型和匹配距离类型。 |
| `MatchData` | 保存原始 KNN 匹配、过滤后匹配、内点掩码和最终内点匹配。 |
| `GeometryData` | 保存估计出的几何模型、矩阵、内点数量、内点比例和状态信息。 |
| `TransformData` | 保存用于图像变换的统一矩阵和变换类型。 |
| `MetricResult` | 保存单个评测指标的名称、数值、有效状态和说明。 |
| `EvaluationData` | 保存一次运行的所有评测指标。 |

## 6. Interfaces 模块

作用：定义各类算法组件的统一接口，让 pipeline 可以通过多态调用不同实现。

核心接口：

| 接口 | 核心功能 |
|---|---|
| `IFeatureExtractor` | 特征提取器接口，负责检测关键点并计算描述子。 |
| `IMatcher` | 描述子匹配器接口，负责生成初始匹配。 |
| `IFilter` | 匹配过滤器接口，负责清理和筛选匹配点。 |
| `IGeometryEstimator` | 几何估计器接口，负责从匹配点估计变换模型。 |
| `IWarper` | 图像变换器接口，负责根据几何结果生成配准图像。 |
| `IPipeline` | 配准流水线接口，负责串联完整流程。 |
| `IRegistration` | 顶层配准任务接口，负责配置和执行。 |

## 7. Pipeline 模块

作用：组织完整配准流程。

核心类：

| 类 | 核心功能 |
|---|---|
| `BasePipeline` | 实现完整流程：读图、提特征、匹配、过滤、估计几何、warp、保存输出、显示窗口。 |
| `FeaturePipeline` | 当前默认使用的特征配准流水线，继承 `BasePipeline`。 |

`BasePipeline` 是项目主线，所有模块最终都在这里被串起来。

## 8. Feature 模块

作用：从输入图像中提取关键点和描述子。

核心类：

| 类 | 核心功能 |
|---|---|
| `SiftExtractor` | 使用 OpenCV SIFT 提取浮点描述子，通常使用 L2 距离。 |
| `SurfExtractor` | 使用 OpenCV xfeatures2d SURF 提取浮点描述子，通常使用 L2 距离。 |
| `OrbExtractor` | 使用 ORB 提取二进制描述子，通常使用 Hamming 距离。 |
| `BriskExtractor` | 使用 BRISK 提取二进制描述子，通常使用 Hamming 距离。 |
| `KazeExtractor` | 使用 KAZE 提取浮点描述子，通常使用 L2 距离。 |
| `AkazeExtractor` | 使用 AKAZE 提取描述子，默认 MLDB 二进制描述子通常使用 Hamming 距离。 |

所有提取器都会把结果写入 `RegistrationContext::feature_data`。

## 9. Matcher 模块

作用：对两张图像的描述子进行初始匹配。

核心类：

| 类 | 核心功能 |
|---|---|
| `BfMatcher` | 使用 `cv::BFMatcher` 做暴力匹配，可通过 `params.method` 选择 `match` / `knn` / `radius`。 |
| `FlannMatcher` | 使用 FLANN 做近似 KNN 匹配，浮点描述子走 KDTree，二进制描述子走 LSH。 |

选择建议：

| 场景 | 推荐 matcher |
|---|---|
| 只需要最近邻匹配 | `bf.yaml` / `BfMatcher`，并设置 `params.method: match` |
| 需要 `RatioTestFilter` | `bf.yaml` / `BfMatcher`，并设置 `params.method: knn`；或 `flann.yaml` / `FlannMatcher` |
| 大规模或近似匹配 | `flann.yaml` / `FlannMatcher` |

## 10. Filter 模块

作用：过滤低质量匹配，提升几何估计稳定性。

核心类：

| 类 | 核心功能 |
|---|---|
| `RatioTestFilter` | 对 KNN 匹配执行 Lowe ratio test，保留最近邻明显优于次近邻的匹配。 |
| `CrossCheckFilter` | 执行双向一致性检查，只保留互为最近邻的匹配。 |
| `GmsFilter` | 使用 GMS 空间一致性约束过滤匹配点。 |

过滤器按 pipeline YAML 中 `filters` 的顺序执行。

## 11. Geometry 模块

作用：根据过滤后的匹配点估计图像之间的几何关系。

核心类：

| 类 | 核心功能 |
|---|---|
| `HomographyEstimator` | 估计 3x3 单应矩阵，适合平面或透视变化。 |
| `AffineEstimator` | 估计 2x3 仿射矩阵，适合平移、旋转、缩放、剪切等变化。 |
| `RigidEstimator` | 估计刚体变换，主要包含旋转和平移。 |
| `SimilarityEstimator` | 估计相似变换，包含旋转、平移和统一缩放。 |

几何估计会生成内点掩码，并写入 `GeometryData` 和 `MatchData::inliers`。

## 12. Transform 模块

作用：根据几何估计结果生成配准后的图像。

核心类：

| 类 | 核心功能 |
|---|---|
| `IWarper` | 图像变换器接口。 |
| `PerspectiveWarper` | 使用 3x3 矩阵执行透视变换。 |
| `AffineWarper` | 使用 2x3 仿射矩阵执行仿射变换。 |

当前主流程通常由 `BasePipeline` 使用 `PerspectiveWarper` 统一处理可 warp 的几何结果。

## 13. Dataset 模块

作用：支持批量样本扫描和真值矩阵读取。

核心类：

| 类/结构 | 核心功能 |
|---|---|
| `Sample` | 表示一个样本，包含 source 图、target 图和可选真值矩阵。 |
| `DatasetLoader` | 扫描数据集目录，按命名规则寻找 source/target 图像，并尝试读取真值。 |
| `DatasetLoader::Options` | 保存数据集根目录、文件名规则、白名单和扩展名列表。 |

## 14. Evaluator 模块

作用：对配准结果计算指标，并支持批量统计。

核心类：

| 类/结构 | 核心功能 |
|---|---|
| `IMetric` | 单个评测指标接口。 |
| `Evaluator` | 加载指标配置并执行多个指标。 |
| `MetricStats` | 保存单个指标的统计结果。 |
| `Statistics` | 聚合多次运行的评测指标并计算均值、中位数、标准差等。 |
| `Benchmark` | 批量评测入口，运行多个 pipeline 和多个样本，输出 CSV 和报告。 |

指标类：

| 类 | 核心功能 |
|---|---|
| `RepeatabilityMetric` | 评估关键点重复率。 |
| `InlierRatioMetric` | 评估几何估计后的内点比例。 |
| `ReprojectionErrorMetric` | 评估重投影误差。 |
| `PsnrMetric` | 计算图像 PSNR。 |
| `RmseMetric` | 计算图像 RMSE。 |
| `SsimMetric` | 计算图像 SSIM。 |

## 15. Output 与 Visualization 模块

作用：输出结果图像、匹配可视化和批量统计。

核心类：

| 类 | 核心功能 |
|---|---|
| `DrawMatches` | 绘制两张图之间的匹配连线。 |
| `DrawInliers` | 绘制几何估计后的内点匹配，可选显示外点。 |
| `DrawOverlay` | 绘制配准结果与目标图的叠加图。 |
| `DrawDiff` | 绘制配准结果与目标图的差异图。 |
| `VisualizationManager` | 统一保存匹配图、内点图、叠加图、差异图和 warped 图。 |

常见输出目录：

| 目录 | 内容 |
|---|---|
| `outputs/matches/` | 匹配可视化图。 |
| `outputs/warped/` | 配准后的图像和 blend 图。 |
| `outputs/batch/<pipeline>/summary.csv` | 批量运行统计摘要。 |

## 16. Utils 模块

作用：提供项目通用辅助能力。

核心类和命名空间：

| 类/命名空间 | 核心功能 |
|---|---|
| `Logger` | 线程安全日志输出，支持 Debug、Info、Warn、Error。 |
| `Timer` | 单段计时器。 |
| `ScopedTimer` | 作用域计时器，自动把耗时写入结果字段。 |
| `yaml_utils` | 安全读取 YAML 中的字符串、数字、布尔值和数组。 |
| `file_utils` | 目录创建、文件读写、CSV 转义、安全文件名生成、子目录枚举。 |
| `image_utils` | 图像转灰度浮点、生成有效 mask、裁剪有效区域等。 |

## 17. 配置文件模块

作用：用 YAML 组合算法和参数。

| 目录 | 内容 |
|---|---|
| `configs/feature/` | 特征提取器配置。 |
| `configs/matcher/` | 匹配器配置，例如 `bf.yaml`、`flann.yaml`。 |
| `configs/filter/` | 匹配过滤器配置。 |
| `configs/geometry/` | 几何估计器配置。 |
| `configs/pipeline/` | 完整 pipeline 配置。 |
| `configs/evaluator/` | 评测指标和 benchmark 配置。 |

## 18. 快速阅读顺序

建议按下面顺序理解项目：

1. `configs/pipeline/sift_pipeline.yaml`
2. `apps/registration_app.cpp`
3. `include/core/config.h`
4. `include/core/context.h`
5. `include/pipeline/base_pipeline.h`
6. `src/pipeline/base_pipeline.cpp`
7. `src/core/factory.cpp`
8. `include/data/*.h`
9. `feature -> matcher -> filter -> geometry -> transform`

一句话总结：这个项目用 YAML 组装一条图像配准流水线，用 `RegistrationContext` 在模块之间传递数据，并把匹配、几何估计、warp 和结果输出统一串起来。
