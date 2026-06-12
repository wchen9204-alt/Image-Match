# 图像配准实验平台开发与使用文档

## 1. 平台定位

本平台是一个基于 C++17、OpenCV 和 YAML 配置驱动的二维图像配准实验平台。平台面向不同配准方法的统一实验、统一输出和统一评测，支持将点特征法、结构特征法、直接法和深度学习匹配法纳入同一套运行框架中。

平台的核心思想是：算法组件由 YAML 配置选择，运行流程由 Pipeline 串联，中间数据由 RegistrationContext 统一传递，实验结果由统一目录结构保存。

典型流程如下：

```text
输入图像对
  -> 读取 pipeline YAML
  -> Factory 创建算法组件
  -> Pipeline 执行配准流程
  -> RegistrationContext 保存中间数据
  -> 几何估计与图像 warping
  -> 评测指标、可视化图、摘要文件和 CSV 输出
```

## 2. 已完成的主要工作

### 2.1 平台框架搭建

已完成 C++ 主体框架，包括命令行入口、配置解析、对象工厂、运行上下文、流水线基类、结果摘要、输出管理和批处理调度。

核心模块如下：

| 模块 | 主要职责 |
|---|---|
| `apps/` | 命令行入口，负责单次运行、批量运行和对比实验调度。 |
| `core/` | 配置加载、枚举类型、工厂创建、上下文和运行结果。 |
| `data/` | 保存图像、特征、匹配、结构、直接法、几何估计和评测结果。 |
| `interfaces/` | 定义提取器、匹配器、过滤器、估计器、直接法和流水线抽象接口。 |
| `pipeline/` | 组织点特征、结构特征、直接法和深度学习四类流水线。 |
| `configs/` | 使用 YAML 管理算法参数、流水线组合、批处理和评测配置。 |
| `outputs/` | 保存单次、批量和对比实验的可视化结果与统计结果。 |

### 2.2 支持的方法类型

平台目前按照方法族组织算法，每个方法族都可以通过 YAML 切换具体实现。

| 方法族 | 已支持内容 |
|---|---|
| 点特征法 | SIFT、SURF、ORB、BRISK、KAZE、AKAZE。 |
| 结构特征法 | 边缘、直线、轮廓三类结构；支持 Chamfer、Hausdorff、ICP、相位相关、线段几何匹配、线描述子匹配和轮廓描述子匹配。 |
| 直接法 | ECC、Global LK、ESM Rigid、ZNCC Rigid、Phase Correlation、Fourier-Mellin、KLT Sparse、Farneback、DIS、TV-L1。 |
| 深度学习匹配法 | LoFTR、SuperPoint + LightGlue、SuperPoint + SuperGlue。 |

### 2.3 配置驱动机制

平台不在主程序中硬编码具体算法组合，而是通过 `configs/` 下的 YAML 文件组织实验。

以点特征 SIFT 配准为例：

```yaml
name: sift_pipeline
keypoint:  ../../keypoint/sift.yaml
matcher:  ../../matcher/bf.yaml
filters:
  - ../../filter/ratio_test.yaml
  - ../../filter/cross_check.yaml
geometry: ../../geometry/rigid.yaml
```

该配置表示：使用 SIFT 提取关键点，使用 BFMatcher 进行匹配，依次执行 Ratio Test 和 Cross Check 过滤，最后用 Rigid 几何模型估计变换。

### 2.4 统一数据流

平台使用 `RegistrationContext` 贯穿一次运行，将不同方法族的中间结果统一保存：

| 数据对象 | 保存内容 |
|---|---|
| `ImagePairData` | 源图、目标图、灰度图和输入路径。 |
| `KeypointData` | 两幅图的关键点、描述子和提取器信息。 |
| `KeypointMatchData` | 原始匹配、过滤后匹配、内点匹配和索引关系。 |
| `StructureData` | 边缘、线段、轮廓响应图与结构基元。 |
| `StructureMatchData` | 结构点对、线段匹配、平移或仿射初值。 |
| `DirectData` | 直接法矩阵、光流、点对、得分和诊断指标。 |
| `GeometryData` | 几何模型类型、矩阵、内点掩码、重投影误差。 |
| `EvaluationData` | 各类评测指标的名称、数值和有效状态。 |

这种设计使四类方法最终都能复用几何估计、warping、可视化和结果输出逻辑。

### 2.5 输出与评测

平台会为每次运行保存可视化结果和文本/JSON 摘要。典型输出包括：

| 输出内容 | 说明 |
|---|---|
| 原始图像 | 保存本次实验使用的 source 和 target。 |
| 特征或结构图 | 点特征关键点图、边缘图、线段图或轮廓图。 |
| 匹配图 | all match 和 inlier match 可视化。 |
| warped 图像 | 将 source 按估计变换配准到 target 坐标系。 |
| blend 图像 | warped source 与 target 的叠加图。 |
| false-color overlay 图像 | warped source 映射到红色、target 映射到绿色，用红边/绿边观察错位、用黄色观察重合。 |
| `run_summary.txt` | 人类可读的单次实验摘要。 |
| `run_summary.json` | 机器可读的单次实验摘要。 |
| `summary.csv` | 单次或批量实验的统计表。 |
| `comparison.csv` | 多方法横向对比实验统计表。 |

当前摘要会记录成功状态、失败原因、关键点或结构数量、匹配数、内点数、内点率、耗时、IoU、NMAD 和评测指标。

## 3. 软件环境

### 3.1 C++ 环境

平台使用 CMake 构建，核心依赖如下：

| 依赖 | 用途 |
|---|---|
| C++17 | 主体工程语言标准。 |
| OpenCV 4.x | 图像读写、特征提取、匹配、几何估计、光流、可视化等。 |
| OpenCV contrib | SURF、line_descriptor、ximgproc、optflow 等扩展模块。 |
| yaml-cpp | 解析 YAML 配置文件。 |
| CMake | 生成构建系统并管理目标。 |

当前 `CMakeLists.txt` 中本机默认路径为：

```text
OpenCV:  D:/Opcv/opencv-contrib-mingw-install
yaml-cpp: D:/yaml-cpp-install
```

如果在其他电脑上运行，需要修改 CMake 缓存参数或 `CMakeLists.txt` 中的安装路径。

### 3.2 Python 环境

深度学习匹配法通过 Python 脚本调用外部模型。主要入口为：

```text
tools/deep/learning_backend.py
```

相关 Python 依赖包括：

```text
torch
torchvision
opencv-python
kornia
certifi
```

第三方模型源码位于：

```text
third_party/LightGlue
third_party/SuperGluePretrainedNetwork
```

这些目录作为 Python 源码依赖使用，不参与 C++ CMake 编译。

## 4. 编译方法

在项目根目录执行：

```powershell
cmake --build project/build-mingw
```

编译成功后，主程序位于：

```text
project/build-mingw/bin/registration_app.exe
```

CMake 构建后会把 `configs/` 复制到可执行文件目录，便于在 build 目录中运行。

## 5. 使用方法

### 5.1 命令行格式

主程序支持两种主要运行方式。

单次实验：

```powershell
project/build-mingw/bin/registration_app.exe <pipeline.yaml> [image1] [image2] [output_dir]
```

批量实验：

```powershell
project/build-mingw/bin/registration_app.exe <batch.yaml>
```

其中，`image1`、`image2` 和 `output_dir` 是可选覆盖参数。如果不传，程序使用 pipeline YAML 中 `io` 字段指定的输入输出路径。

### 5.2 单次实验示例

点特征法：

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/keypoint/sift_pipeline.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/keypoint/orb_pipeline.yaml
```

结构特征法：

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/structure/line_pipeline.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/structure/contour_pipeline.yaml
```

直接法：

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/direct/global_direct_pipeline.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/direct/frequency_direct_pipeline.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/direct/dense_direct_pipeline.yaml
```

深度学习匹配法：

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/learning/loftr_learning_pipeline.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/learning/superpoint_lightglue_learning_pipeline.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/learning/superpoint_superglue_learning_pipeline.yaml
```

指定临时输入图像和输出目录：

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/keypoint/sift_pipeline.yaml data/source.png data/target.png project/outputs/demo
```

### 5.3 批量实验示例

批量配置位于：

```text
project/configs/pipeline/batch/
```

常用命令：

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/batch_keypoint.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/batch_structure.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/batch_direct.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/batch_learning.yaml
```

批处理会扫描 `dataset.root` 指定的数据集目录，按配置中的文件名关键词寻找源图和目标图：

```yaml
dataset:
  root: ../../../datasets
  pattern_sources: [source, moving]
  pattern_targets: [target, reference]
  include: []
```

如果 `include` 为空，则扫描全部样本；如果填写样本名，则只运行指定样本。

### 5.4 方法对比实验

平台支持对一组方法组合进行横向对比，并输出总表。

直接法对比：

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/compare_direct.yaml
```

该配置会依次测试 ECC、Global LK、ESM Rigid、ZNCC Rigid、Phase Correlation、Fourier-Mellin、KLT Sparse、DIS、Farneback 和 TV-L1。

直线结构方法对比：

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/compare_line.yaml
```

该配置会遍历 LSD、FLD、HoughLinesP 等检测器与 LBD、MSLD、LINE_SIFT 等线描述子组合。

对比结果会保存为：

```text
outputs/compare/.../comparison.csv
```

## 6. 配置文件说明

### 6.1 Pipeline 配置

单次 pipeline YAML 通常包含以下字段：

| 字段 | 含义 |
|---|---|
| `name` | 流水线名称，用于日志和输出目录命名。 |
| `method_family` | 方法族，常见值为 `keypoint`、`structure`、`direct`、`learning`。 |
| `keypoint` | 点特征提取器配置路径。 |
| `structure` | 结构特征提取与关联配置路径。 |
| `direct` | 直接法算法配置路径。 |
| `learning` | 深度学习匹配器配置路径。 |
| `matcher` | 点特征匹配器配置路径。 |
| `filters` | 匹配过滤器配置列表，按顺序执行。 |
| `geometry` | 几何估计器配置路径。 |
| `evaluator` | 评测指标配置路径。 |
| `io` | 输入图像和输出根目录。 |
| `visualization` | 可视化开关、最大绘制匹配数和窗口显示设置。 |
| `validation` | warp 后重叠率和光度一致性校验设置。 |

### 6.2 算法子配置

算法配置文件用于调整具体方法参数。例如：

| 配置目录 | 作用 |
|---|---|
| `configs/keypoint/` | 设置 SIFT、SURF、ORB、BRISK、KAZE、AKAZE 参数。 |
| `configs/matcher/` | 设置 BFMatcher 或 FlannMatcher。 |
| `configs/filter/` | 设置 Ratio Test、Cross Check、GMS 等过滤器。 |
| `configs/geometry/` | 设置 Homography、Affine、Rigid、Similarity 估计参数。 |
| `configs/structure/` | 设置边缘、直线、轮廓检测与关联参数。 |
| `configs/direct/` | 设置直接法算法参数。 |
| `configs/learning/` | 设置 Python 深度学习模型调用参数。 |

## 7. 数据集组织方式

数据集默认位于：

```text
project/datasets/
```

当前样本目录包括：

```text
Test01, Test02, ..., Test12
```

一般样本使用：

```text
source.png
target.png
```

`Test02` 使用：

```text
moving.png
reference.png
```

批处理配置通过 `pattern_sources` 和 `pattern_targets` 同时兼容这两类命名方式。

## 8. 输出目录说明

单次实验输出结构：

```text
outputs/single/<method_family>/<pipeline>/<sample>/
```

批量实验输出结构：

```text
outputs/batch/<method_family>/<pipeline>/<sample>/
```

对比实验输出结构：

```text
outputs/compare/<method_or_group>/
```

常见子目录：

| 子目录 | 内容 |
|---|---|
| `originals/` | 原始输入图像。 |
| `keypoints/` | 点特征关键点可视化。 |
| `structures/` | 边缘、直线或轮廓可视化。 |
| `matches/`、`all_match/`、`inlier_match/` | 匹配可视化。 |
| `direct/` | 直接法相关输出。 |
| `learning/` | 深度学习匹配输出。 |
| `warped/` | 配准后的图像。 |
| `blend/` | warped 图像与目标图叠加结果。 |

重点查看文件：

```text
run_summary.txt
run_summary.json
summary.csv
comparison.csv
```

## 9. 二次开发说明

二次开发分为两个层级：第一层是在平台中新增一个和点特征法、结构特征法、直接法、深度学习匹配法同级的“方法族”；第二层是在已有方法族内部新增一个具体算法，例如新增一个点特征提取器或新增一个直接法配准器。

### 9.1 添加新的同级方法族

“同级方法族”指的是和 `keypoint`、`structure`、`direct`、`learning` 并列的新实验路线。例如后续如果要加入基于互信息的配准、基于模板搜索的配准、基于语义分割辅助的配准，建议作为新的方法族接入，而不是强行塞进点特征法或直接法目录。

新增同级方法族需要同时补齐类型、数据、流水线、配置、输出和批处理入口。

| 开发位置 | 需要修改的内容 |
|---|---|
| `include/core/config.h` | 在 `MethodFamily` 中新增方法族枚举，并更新 `methodFamilyDir()` 和 `methodFamilyLabel()`。 |
| `src/core/config.cpp` | 让 `Config::loadPipeline()` 能识别新的 `method_family` 字段，并解析该方法族自己的配置路径。 |
| `include/core/context.h` | 如果新方法有专属中间结果，在 `RegistrationContext` 中增加对应数据对象。 |
| `include/data/` | 新增该方法族的数据结构；如果最终能转成点对，建议同时接入 `CorrespondenceView`。 |
| `include/interfaces/` | 如果该方法族需要新的算法抽象，添加对应接口，例如 `i_xxx_aligner.h`。 |
| `include/<new_family>/` 和 `src/<new_family>/` | 放置新方法族的具体算法声明和实现。 |
| `include/pipeline/` 和 `src/pipeline/` | 新增 `<NewFamily>Pipeline`，实现从读图到配准、几何、warp、评测和输出的完整流程。 |
| `src/core/factory.cpp` | 如果算法组件由 YAML 创建，需要在 `Factory` 中增加创建分支。 |
| `apps/registration_app.cpp` | 在 `createPipelineForConfig()` 中按新 `MethodFamily` 创建新 pipeline，并补充终端摘要、JSON/CSV 输出字段。 |
| `configs/<new_family>/` | 添加新方法族的算法参数配置。 |
| `configs/pipeline/<new_family>/` | 添加新方法族的 pipeline YAML。 |
| `configs/pipeline/batch/` | 如需批量实验，添加对应 batch YAML；如需横向对比，添加 compare YAML。 |
| `PROJECT_DIRECTORY_STRUCTURE_CN.md` | 更新目录结构说明。 |

推荐接入顺序如下：

1. 先确定新方法族的输入输出：是否输出几何矩阵、点对、光流、响应图或其他中间结果。
2. 在 `data/` 中定义数据结构，并把结果挂到 `RegistrationContext`。
3. 实现一个最小可运行的 Pipeline，只完成读图、核心配准、写入 `GeometryData` 和 `RegistrationResult`。
4. 接入 `RegistrationApp` 的方法族分发，让单次 pipeline 能跑通。
5. 补充 `configs/<new_family>/` 和 `configs/pipeline/<new_family>/`。
6. 再补充可视化、评测指标、批处理和对比实验。
7. 最后更新目录文档、使用文档和汇报说明。

新方法族的 pipeline YAML 建议保持和现有配置风格一致：

```yaml
name: example_new_family_pipeline
method_family: new_family

new_family: ../../new_family/example.yaml
geometry: ../../geometry/rigid.yaml
evaluator: ../../evaluator/metrics.yaml

io:
  image1: ../../../datasets/Test01/source.png
  image2: ../../../datasets/Test01/target.png
  output_dir: ../../../outputs

visualization:
  draw_matches: true
  warp: true
  show_source_window: false
  show_target_window: false
  show_warped_window: false
```

判断是否应该新增同级方法族，可以参考下面标准：

| 情况 | 建议 |
|---|---|
| 只是新增 SIFT、ORB 之外的另一种关键点提取器 | 放在 `keypoint` 方法族内部。 |
| 只是新增 ECC、ZNCC 之外的另一种直接法优化器 | 放在 `direct` 方法族内部。 |
| 方法流程仍然是“提点、匹配、过滤、估计几何” | 复用 `keypoint` 或 `learning` 方法族。 |
| 方法流程有独立的数据结构、核心阶段和摘要指标 | 新增同级方法族。 |
| 方法不产生常规点对，但能直接输出几何矩阵或 dense field | 可以新增同级方法族，也可以视情况归入 `direct`。 |

### 9.2 添加新的点特征方法

1. 在 `include/keypoint/` 添加提取器头文件。
2. 在 `src/keypoint/` 添加实现文件。
3. 实现 `IKeypointExtractor` 接口。
4. 在 `Factory` 中注册新的类型。
5. 在 `configs/keypoint/` 添加 YAML 参数文件。
6. 在 `configs/pipeline/keypoint/` 添加或修改 pipeline。

### 9.3 添加新的结构关联方法

1. 在 `include/matcher/structure/` 定义关联器。
2. 在 `src/matcher/structure/` 实现关联逻辑。
3. 实现 `IStructureAssociator` 接口。
4. 在 `configs/structure/*.yaml` 中添加参数。
5. 在 `Factory` 中添加创建分支。

### 9.4 添加新的直接法

1. 在 `include/direct/` 下选择 global、frequency、sparse 或 dense 子目录。
2. 实现 `IDirectAligner` 接口。
3. 输出 `DirectData`，包括矩阵、点对、得分和诊断指标。
4. 在 `configs/direct/` 添加算法配置。
5. 在 direct pipeline YAML 中引用该配置。

### 9.5 添加新的深度学习匹配器

1. 在 `tools/deep/` 添加 Python 推理入口，输出统一 matches JSON。
2. 在 `configs/learning/` 添加模型配置。
3. 在 `PythonLearningMatcher` 或学习后端中加入方法分支。
4. 复用 C++ 的几何估计和评测模块。

### 9.6 添加新的评测指标

1. 在 `include/evaluator/metrics/` 添加指标头文件。
2. 在 `src/evaluator/metrics/` 添加实现文件。
3. 在 `configs/evaluator/metrics.yaml` 中配置启用状态和参数。
4. 在 `Evaluator` 中注册并输出指标。

## 10. 阶段性成果总结

本阶段已经完成一个可运行、可配置、可扩展的图像配准实验平台。主要成果包括：

1. 搭建了统一的 C++ 图像配准实验框架。
2. 支持点特征、结构特征、直接法和深度学习匹配四类方法。
3. 使用 YAML 配置实现算法组合与参数调整。
4. 支持单次运行、批量运行和多方法对比实验。
5. 建立了统一输出目录、摘要文件、CSV 统计和可视化结果。
6. 通过 `RegistrationContext` 和 `CorrespondenceView` 统一不同方法族的数据流。
7. 为后续新增算法、评测指标和实验数据集预留了清晰扩展接口。

## 11. 当前不足与后续计划

后续可以继续改进以下方向：

1. 扩充数据集规模，加入更多具有旋转、尺度、遮挡、光照变化的样本。
2. 完善更多评测指标，例如重投影误差、SSIM、RMSE、PSNR 的批量统计展示。
3. 将 Python 深度学习依赖进一步封装，降低跨机器部署成本。
4. 增加实验报告自动生成脚本，从 CSV 自动生成汇总表和可视化图表。
5. 对各方法默认参数进行系统调参，形成推荐配置。
6. 增加失败样本自动诊断，例如匹配不足、内点不足、warp 重叠不足等分类统计。

## 12. 汇报时可强调的亮点

1. 平台不是单一算法实现，而是一个可组合的实验框架。
2. 点特征、结构、直接法和深度学习方法共用同一套输出和评测体系。
3. 通过 YAML 修改实验，不需要频繁改 C++ 代码。
4. 支持批量样本和多方法横向对比，便于做论文或课题实验统计。
5. 深度学习方法已经接入到 C++ 主流程，能够复用统一几何估计和可视化模块。

## 13. 常用命令汇总

编译：

```powershell
cmake --build project/build-mingw
```

运行 SIFT 单次实验：

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/keypoint/sift_pipeline.yaml
```

运行直线结构单次实验：

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/structure/line_pipeline.yaml
```

运行直接法对比：

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/compare_direct.yaml
```

运行批量点特征实验：

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/batch_keypoint.yaml
```

运行批量深度学习实验：

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/batch_learning.yaml
```
