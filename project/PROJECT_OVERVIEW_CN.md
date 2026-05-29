# Project 中文说明

> 说明范围
> - 包含 `project/` 主流程代码、配置和运行方式
> - 不包含 `build-mingw/`
> - `include/evaluator/`、`src/evaluator/`、`configs/evaluator/` 目前仍以预留结构为主，主线功能重点在单次运行与批量样本运行

## 1. 项目定位

这是一个基于 OpenCV 的二维图像配准实验平台。

当前主线目标是：

1. 读取两张输入图像
2. 提取局部特征点和描述子
3. 对描述子做匹配
4. 通过过滤器去掉不可靠匹配
5. 估计两图之间的几何变换
6. 将源图变换到目标图坐标系
7. 输出匹配可视化、配准结果图和运行统计

当前已经接入的几何模型有：

- `HOMOGRAPHY`
- `AFFINE`
- `RIGID`
- `SIMILARITY`

项目强调“配置驱动”：

- 算法链通过 YAML 组装
- 具体模块通过工厂创建
- 运行数据通过统一上下文传递
- 单次运行和批量运行共用同一条 pipeline 主线

## 2. 目录结构

```text
project/
├─ CMakeLists.txt
├─ main.cpp
├─ apps/
│  ├─ registration_app.h
│  └─ registration_app.cpp
├─ configs/
│  ├─ feature/
│  ├─ matcher/
│  ├─ filter/
│  ├─ geometry/
│  ├─ pipeline/
│  │  ├─ sift_pipeline.yaml
│  │  ├─ orb_pipeline.yaml
│  │  ├─ surf_pipeline.yaml
│  │  ├─ brisk_pipeline.yaml
│  │  ├─ kaze_pipeline.yaml
│  │  ├─ akaze_pipeline.yaml
│  │  └─ batch_pipeline.yaml
│  └─ evaluator/
├─ datasets/
│  ├─ test1/
│  ├─ test2/
│  └─ ...
├─ include/
│  ├─ core/
│  ├─ data/
│  ├─ dataset/
│  ├─ evaluator/
│  ├─ feature/
│  ├─ filter/
│  ├─ geometry/
│  ├─ interfaces/
│  ├─ matcher/
│  ├─ pipeline/
│  ├─ transform/
│  └─ utils/
├─ outputs/
└─ src/
   ├─ core/
   ├─ dataset/
   ├─ feature/
   ├─ filter/
   ├─ geometry/
   ├─ matcher/
   ├─ pipeline/
   ├─ transform/
   └─ utils/
```

## 3. 运行模式

目前主线支持两种运行模式。

### 3.1 单次运行

直接执行某一个 pipeline YAML：

```powershell
.\build-mingw\bin\registration_app.exe configs/pipeline/sift_pipeline.yaml
```

也可以在命令行覆盖图像路径和输出目录：

```powershell
.\build-mingw\bin\registration_app.exe configs/pipeline/sift_pipeline.yaml img1.png img2.png outputs
```

### 3.2 批量运行

执行一个“批量配置 YAML”，由它指定：

- 使用哪一个 pipeline
- 扫描哪个数据集根目录
- 是否只跑 `include` 中列出的样本
- 输出写到哪里

示例：

```powershell
.\build-mingw\bin\registration_app.exe configs/pipeline/batch_pipeline.yaml
```

## 4. 单次 pipeline 配置

以 [sift_pipeline.yaml](/abs/path/d:/Experimental-testing-platform/project/configs/pipeline/sift_pipeline.yaml) 为例：

```yaml
name: sift_pipeline

feature:  configs/feature/sift.yaml
matcher:  configs/matcher/bf.yaml

filters:
  - configs/filter/ratio_test.yaml
  - configs/filter/cross_check.yaml

geometry: configs/geometry/similarity.yaml

io:
  image1: datasets/test10/source.png
  image2: datasets/test10/target.png
  output_dir: outputs
```

它描述的是“一次运行时的算法链”：

- 用什么特征
- 用什么匹配器
- 用哪些过滤器
- 用什么几何模型
- 默认输入输出路径是什么

它不负责描述“跑哪些样本”；那部分交给批量配置。

## 5. 批量配置

当前批量配置示例是 [batch_pipeline.yaml](/abs/path/d:/Experimental-testing-platform/project/configs/pipeline/batch_pipeline.yaml)。

```yaml
name: batch_pipeline

pipeline: configs/pipeline/sift_pipeline.yaml

dataset:
  root: datasets
  pattern_source: source
  pattern_target: target
  include: []

output:
  root: ../../outputs/batch/current
  save_visuals: true
  summary_csv: true
```

### 5.1 `pipeline`

这一项指定“批量运行时复用哪一个单次 pipeline”。

例如：

- `configs/pipeline/sift_pipeline.yaml`
- `configs/pipeline/orb_pipeline.yaml`
- `configs/pipeline/surf_pipeline.yaml`

把这一行改掉，就相当于切换批量测试的方法。

### 5.2 `dataset.include`

`include` 支持两种常见用法：

1. 跑整个数据集

```yaml
include: []
```

或者直接省略 `include`

2. 只跑指定样本

```yaml
include:
  - test1
  - test2
  - test3
```

### 5.3 `output.root`

批量输出默认建议写到项目根目录下的 `outputs/` 中，例如：

```yaml
output:
  root: ../../outputs/batch/current
```

其中最后一级如果写成 `current`，程序会自动替换成当前引用的 pipeline 名：

- `sift_pipeline`
- `orb_pipeline`
- `surf_pipeline`

所以实际效果会像：

- `project/outputs/batch/sift_pipeline/...`
- `project/outputs/batch/orb_pipeline/...`

## 6. 主流程入口

### 6.1 `main.cpp`

文件：

- `main.cpp`

作用：

- 程序总入口
- 只负责调用 `RegistrationApp::run(argc, argv)`

### 6.2 `RegistrationApp`

文件：

- `apps/registration_app.h`
- `apps/registration_app.cpp`

作用：

- 解析命令行参数
- 判断当前 YAML 是单次 pipeline 还是批量配置
- 单次模式下执行一次 pipeline
- 批量模式下扫描数据集并循环执行同一个 pipeline
- 打印运行摘要
- 批量模式输出 `summary.csv`

## 7. 配置加载

### 7.1 `PipelineConfig`

文件：

- `include/core/config.h`

作用：

- 保存单次 pipeline YAML 解析后的结果

主要内容：

- `name`
- `feature_path`
- `matcher_path`
- `filter_paths`
- `geometry_path`
- `image1_path`
- `image2_path`
- `output_dir`
- 各类可视化开关

### 7.2 `Config`

文件：

- `include/core/config.h`
- `src/core/config.cpp`

作用：

- 读取 YAML
- 解析相对路径
- 生成 `PipelineConfig`

主要函数：

- `load(path)`
- `resolvePath(base_dir, relative_or_absolute)`
- `loadPipeline(path)`

## 8. 上下文与结果

### 8.1 `RegistrationContext`

文件：

- `include/core/context.h`

作用：

- 在整条 pipeline 内共享运行数据

内部主要包含：

- `feature_data`
- `match_data`
- `geometry_data`
- `transform_data`
- `evaluation`
- `result`
- `image1_path`
- `image2_path`
- `output_dir`
- `warped_image`

### 8.2 `RegistrationResult`

文件：

- `include/core/result.h`

作用：

- 记录一次运行的摘要信息

主要字段：

- 是否成功
- 错误消息
- 关键点数
- 原始匹配数
- 过滤后匹配数
- 内点数
- 内点率
- 各阶段耗时

## 9. Pipeline 结构

### 9.1 `BasePipeline`

文件：

- `include/pipeline/base_pipeline.h`
- `src/pipeline/base_pipeline.cpp`

作用：

- 把“读图 -> 提特征 -> 匹配 -> 过滤 -> 几何估计 -> 变换 -> 保存输出”这条主线串起来

主要阶段函数：

- `configure(...)`
- `loadImages(...)`
- `runExtract(...)`
- `runMatch(...)`
- `runFilters(...)`
- `runGeometry(...)`
- `runWarp(...)`
- `saveOutputs(...)`
- `showWindows(...)`
- `run(...)`

### 9.2 `FeaturePipeline`

文件：

- `include/pipeline/feature_pipeline.h`
- `src/pipeline/feature_pipeline.cpp`

作用：

- 当前默认使用的具体 pipeline 类型
- 继承 `BasePipeline`

## 10. 工厂与模块创建

### 10.1 `Factory`

文件：

- `include/core/factory.h`
- `src/core/factory.cpp`

作用：

- 根据 YAML 中的 `type` 创建具体模块

当前可创建的模块包括：

- 特征：`SIFT / SURF / ORB / BRISK / KAZE / AKAZE`
- 匹配器：`BF / FLANN`
- 过滤器：`RATIO_TEST / CROSS_CHECK / GMS`
- 几何估计：`HOMOGRAPHY / AFFINE / RIGID / SIMILARITY`

## 11. 特征、匹配、过滤、几何

### 11.1 特征提取器

目录：

- `include/feature/`
- `src/feature/`

每个提取器负责：

- 读取自身 YAML 参数
- 创建 OpenCV 提取器
- 执行 `detectAndCompute`
- 把关键点和描述子写入 `RegistrationContext`

### 11.2 匹配器

目录：

- `include/matcher/`
- `src/matcher/`

当前主线常用的是 `BFMatcher`。

它负责：

- 根据描述子类型确定 `L2` 或 `HAMMING`
- 执行 `match` 或 `knnMatch`
- 把原始匹配写入 `match_data.raw_knn`

### 11.3 过滤器

目录：

- `include/filter/`
- `src/filter/`

当前包括：

- `RatioTestFilter`
- `CrossCheckFilter`
- `GmsFilter`

说明：

- 如果配置了过滤器，按 YAML 顺序依次执行
- 如果没有过滤器，当前实现会自动把 `raw_knn` 的 top-1 提升为 `filtered`，确保后续几何阶段仍可运行

### 11.4 几何估计器

目录：

- `include/geometry/`
- `src/geometry/`

当前包括：

- `HomographyEstimator`
- `AffineEstimator`
- `RigidEstimator`
- `SimilarityEstimator`

它们负责：

- 从 `filtered matches` 提取点对
- 调用 OpenCV 几何估计接口
- 生成内点掩码
- 写回 `geometry_data`

## 12. 变换与可视化

### 12.1 图像变换

目录：

- `include/transform/`
- `src/transform/`

当前由 `PerspectiveWarper` 负责：

- 对 `HOMOGRAPHY` 直接使用 3x3 矩阵
- 对 `AFFINE / RIGID / SIMILARITY` 将 2x3 扩成 3x3 再统一 warp

### 12.2 可视化输出

目录：

- `include/utils/visualization/`
- `src/utils/visualization/`

当前会根据配置输出：

- 匹配图
- warped 图
- blend 图

另外也支持窗口显示：

- `show_source_window`
- `show_target_window`
- `show_warped_window`
- `wait_key`

批量模式下会强制关闭这些窗口，避免测试过程中卡在 `waitKey`。

## 13. 数据集扫描

### 13.1 `Sample`

文件：

- `include/dataset/sample.h`

作用：

- 表示一个样本目录中成对的 `source` / `target`

主要字段：

- `name`
- `source_path`
- `target_path`
- `H_gt`

### 13.2 `DatasetLoader`

文件：

- `include/dataset/dataset_loader.h`
- `src/dataset/dataset_loader.cpp`

作用：

- 扫描 `datasets/` 下的样本目录
- 自动寻找 `source.*` 和 `target.*`
- 支持 `include` 白名单
- 可选读取 `H_gt.txt`

## 14. 批量运行的实际规则

批量模式不是“同时运行多种方法”，而是：

- 选择一个单次 pipeline
- 对多个样本目录循环复用这一条 pipeline

例如：

```yaml
pipeline: configs/pipeline/sift_pipeline.yaml
dataset:
  include:
    - test1
    - test2
    - test3
```

表示：

- 方法固定是 `sift_pipeline`
- 样本依次跑 `test1`、`test2`、`test3`

如果改成：

```yaml
pipeline: configs/pipeline/orb_pipeline.yaml
```

则表示：

- 方法切换为 `orb_pipeline`
- 样本列表仍由 `dataset.include` 决定

## 15. 当前批量模式会继承哪些设置

批量模式会继承被引用 pipeline 里的方法配置，包括：

- `feature`
- `matcher`
- `filters`
- `geometry`
- `draw_matches`
- `draw_inliers_only`
- `max_matches_drawn`
- `warp`

批量模式会覆盖或强制处理的内容：

- `image1_path`
- `image2_path`
- `output_dir`
- `show_source_window = false`
- `show_target_window = false`
- `show_warped_window = false`

也就是说，批量运行时“算法链怎么跑”，依然以被引用的单次 pipeline YAML 为准。

## 16. 输出结果

### 16.1 单次运行

默认写到 `pipeline.yaml` 中的 `io.output_dir`。

常见内容：

- `matches/*.png`
- `warped/*.png`

### 16.2 批量运行

批量模式会在 `output.root` 下按样本名分目录输出，并写汇总 CSV。

例如：

```text
outputs/
└─ batch/
   └─ sift_pipeline/
      ├─ summary.csv
      ├─ test1/
      │  ├─ matches/
      │  └─ warped/
      ├─ test2/
      └─ test3/
```

`summary.csv` 当前会记录：

- `sample_name`
- `success`
- `message`
- `num_keypoints_first`
- `num_keypoints_second`
- `num_raw_matches`
- `num_filtered_matches`
- `num_inliers`
- `inlier_ratio`
- `t_total_ms`

## 17. 当前已移除的旧设计

`PipelineManager` 已经从主线删除。

原因是当前目标不是“管理多个方法的 pipeline 列表”，而是：

- 一个方法对应一个 pipeline
- 同一个 pipeline 可批量跑多个样本

因此现阶段由：

- `RegistrationApp`
- `DatasetLoader`
- `FeaturePipeline`

共同构成批量运行主线。

## 18. 推荐阅读顺序

如果要快速读懂当前项目，建议按下面顺序：

1. `main.cpp`
2. `apps/registration_app.h` / `apps/registration_app.cpp`
3. `include/core/config.h` / `src/core/config.cpp`
4. `include/core/context.h`
5. `include/pipeline/base_pipeline.h` / `src/pipeline/base_pipeline.cpp`
6. `include/dataset/sample.h`
7. `include/dataset/dataset_loader.h` / `src/dataset/dataset_loader.cpp`
8. `src/core/factory.cpp`
9. `feature -> matcher -> filter -> geometry -> transform`
10. `configs/pipeline/*.yaml`

## 19. 一句话总结

这个项目当前的设计核心可以概括为：

**用 YAML 组装一条单方法配准流程，用同一条流程对多个样本重复执行，并把结果统一输出到 `project/outputs/` 下。**
