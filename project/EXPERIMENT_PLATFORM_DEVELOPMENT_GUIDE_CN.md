# 实验平台开发使用说明

本文面向导师汇报和项目交接，说明本实验平台的研究目标、总体架构、运行方式、输出结果和后续扩展方法。

## 1. 平台定位

本项目是一个基于 OpenCV 与 YAML 配置驱动的二维图像配准实验平台。平台目标不是只实现某一个固定算法，而是把配准流程拆成可替换的模块，使点特征法、结构特征法以及后续可能加入的深度特征法可以在同一套运行框架下比较。

一次完整实验被拆成以下阶段：

| 阶段 | 作用 | 当前实现 |
|---|---|---|
| 图像读取 | 读取 source / target，并统一准备灰度图 | 支持普通图像与高位深 TIFF 归一化 |
| 特征或结构提取 | 从图像中提取点、线、边缘或轮廓 | SIFT、SURF、ORB、BRISK、KAZE、AKAZE；EDGE、LINE、CONTOUR |
| 匹配或结构关联 | 建立 source 与 target 的对应关系 | BF、FLANN、线描述子、相位相关、Chamfer、Hausdorff、ICP |
| 匹配过滤 | 清理误匹配，提高几何估计稳定性 | Ratio Test、Cross Check、GMS、距离阈值、最小距离、距离分布 |
| 几何估计 | 从对应关系估计空间变换 | Homography、Affine、Rigid、Similarity |
| 图像变换 | 将 source warp 到 target 坐标系 | PerspectiveWarper，兼容仿射与单应矩阵 |
| 输出与统计 | 保存图像、匹配图、摘要和 CSV | 单次与批处理输出均支持 |

平台的核心特点是“配置驱动 + 模块化接口 + 可批量评测”。研究过程中可以只修改 YAML 更换算法组合，不需要每次改 C++ 主流程。

## 2. 项目运行环境

当前工程面向 Windows / MinGW / CMake 环境开发。

主要依赖：

| 依赖 | 作用 |
|---|---|
| C++17 | 项目语言标准 |
| CMake | 构建系统 |
| OpenCV 4.12 contrib | 图像处理、特征、几何估计、xfeatures2d、ximgproc、line_descriptor |
| yaml-cpp | 读取 YAML 配置 |

当前 `CMakeLists.txt` 中使用的依赖路径：

```cmake
OPENCV_INSTALL_ROOT = D:/Opcv/opencv-contrib-mingw-install
YAML_CPP_ROOT       = D:/yaml-cpp-install
```

构建命令：

```powershell
cd D:\Experimental-testing-platform\project
cmake --build build-mingw
```

生成的可执行文件位于：

```text
project/build-mingw/bin/registration_app.exe
```

构建后会把 `project/configs` 自动复制到 `build-mingw/bin/configs`，因此在可执行文件目录中运行时可以直接使用相对配置路径。

## 3. 单次实验运行

点特征单次运行示例：

```powershell
cd D:\Experimental-testing-platform\project\build-mingw\bin
.\registration_app.exe configs\pipeline\keypoint\sift_pipeline.yaml
```

结构线方法单次运行示例：

```powershell
cd D:\Experimental-testing-platform\project\build-mingw\bin
.\registration_app.exe configs\pipeline\structure\line_pipeline.yaml
```

也可以在命令行覆盖 YAML 中的输入图像和输出目录：

```powershell
.\registration_app.exe configs\pipeline\keypoint\orb_pipeline.yaml source.png target.png outputs
```

单次运行时，程序会自动完成：

1. 加载 pipeline YAML。
2. 解析子配置路径。
3. 根据配置创建算法组件。
4. 执行完整配准流程。
5. 写出可视化和统计摘要。

## 4. 批量实验运行

点特征批处理：

```powershell
.\registration_app.exe configs\pipeline\batch\batch_keypoint.yaml
```

结构法批处理：

```powershell
.\registration_app.exe configs\pipeline\batch\batch_structure.yaml
```

批处理配置中主要包含：

| 字段 | 作用 |
|---|---|
| `pipeline` | 批量实验复用的单次 pipeline |
| `dataset.root` | 数据集根目录 |
| `pattern_source` | source 图像文件名关键词 |
| `pattern_target` | target 图像文件名关键词 |
| `include` | 只运行指定样本；为空表示全部样本 |
| `output.root` | 输出根目录 |
| `save_visuals` | 是否保存图像可视化 |
| `summary_csv` | 是否保存汇总 CSV |

当前数据集目录形如：

```text
datasets/
  test1/source.png
  test1/target.png
  test2/source.png
  test2/target.png
  ...
```

## 5. 输出目录说明

单次输出目录：

```text
outputs/single/<method_family>/<pipeline_name>/<sample_name>/
```

批量输出目录：

```text
outputs/batch/<method_family>/<pipeline_name>/<sample_name>/
```

每个样本下常见输出：

| 目录或文件 | 内容 |
|---|---|
| `originals/` | source 与 target 原图 |
| `keypoints/` | 点特征关键点图 |
| `structures/` | 结构响应图，例如边缘图、线段图、轮廓图 |
| `matches/` | 匹配可视化图 |
| `warped/` | warp 后的 source |
| `blend/` | warped source 与 target 的叠加图 |
| `run_summary.txt` | 单样本文本摘要 |
| `run_summary.json` | 单样本 JSON 摘要 |
| `summary.csv` | 单样本或批量汇总表 |

批量模式还会在 pipeline 层生成总表：

```text
outputs/batch/<method_family>/<pipeline_name>/summary.csv
```

汇总表核心字段：

| 字段 | 说明 |
|---|---|
| `success` | 是否运行成功 |
| `message` | 成功或失败原因 |
| `num_keypoints_first/second` | 点特征数量 |
| `num_structures_first/second` | 结构元素数量 |
| `num_raw_matches` | 原始匹配数量 |
| `num_filtered_matches` | 过滤后匹配数量 |
| `num_inliers` | 几何内点数量 |
| `inlier_ratio` | 内点比例或结构关联得分 |
| `warp_overlap_iou` | 可选的 warp 前景重合质量 |
| `t_*_ms` | 各阶段耗时 |

## 6. 当前已实现方法

### 6.1 点特征法

当前点特征提取器：

| 方法 | OpenCV 接口 | 描述子类型 |
|---|---|---|
| SIFT | `cv::SIFT` | 浮点 |
| SURF | `cv::xfeatures2d::SURF` | 浮点 |
| ORB | `cv::ORB` | 二进制 |
| BRISK | `cv::BRISK` | 二进制 |
| KAZE | `cv::KAZE` | 浮点 |
| AKAZE | `cv::AKAZE` | 二进制或浮点，取决于配置 |

点特征流程：

```text
KeypointExtractor -> Matcher -> Filters -> GeometryEstimator -> Warper
```

### 6.2 结构特征法

当前结构提取器：

| 类型 | 实现 |
|---|---|
| EDGE | Canny、Sobel、LoG、Laplacian |
| LINE | HoughLines、HoughLinesP、LSD、FLD |
| CONTOUR | 二值化后提取轮廓 |

线结构当前重点支持：

| 组件 | 当前实现 |
|---|---|
| 检测 | OpenCV LSD 或 line_descriptor LSDDetector |
| 描述子 | LBD |
| 匹配 | `cv::line_descriptor::BinaryDescriptorMatcher` |
| 后处理 | 方向差、长度比例、中点位移一致性、一对一去重 |
| 几何结果 | 基于匹配线段中心估计平移仿射 |
| 可视化 | 优先使用 `cv::line_descriptor::drawLineMatches` |

## 7. 配置文件使用方式

一个点特征 pipeline 由以下子配置组成：

```yaml
keypoint: ../../keypoint/sift.yaml
matcher: ../../matcher/bf.yaml
filters:
  - ../../filter/ratio_test.yaml
  - ../../filter/cross_check.yaml
geometry: ../../geometry/homography.yaml
```

一个结构 pipeline 由以下子配置组成：

```yaml
structure: ../../structure/line.yaml
geometry: ../../geometry/rigid.yaml
```

结构关联器不单独放在 pipeline YAML 中，而是在 `structure/line.yaml` 的 `association` 节点中配置。

## 8. 开发扩展流程

### 8.1 增加一个点特征提取器

1. 在 `include/keypoint/` 下新增头文件，继承 `IKeypointExtractor`。
2. 在 `src/keypoint/` 下实现 `extract(RegistrationContext&)`。
3. 在 `Factory::createKeypointExtractor` 中注册字符串。
4. 在 `configs/keypoint/` 下新增 YAML。
5. 新增或复制一个 pipeline YAML。

### 8.2 增加一个匹配过滤器

1. 继承 `IFilter`。
2. 实现 `apply(RegistrationContext&)`。
3. 在 `Factory::createFilter` 中注册。
4. 新增 `configs/filter/*.yaml`。
5. 在 pipeline 的 `filters` 列表中加入该配置。

### 8.3 增加一个几何估计器

1. 继承 `IGeometryEstimator`。
2. 从 `ctx.keypoint_data` 与 `ctx.keypoint_match_data.filtered` 读取点对。
3. 调用 OpenCV 几何估计函数或自定义估计器。
4. 写回 `ctx.geometry_data`。
5. 在 `Factory::createGeometryEstimator` 中注册。

### 8.4 增加一个线描述子

建议把“描述子计算”和“匹配器选择”分层设计：

| 描述子 | 推荐匹配方式 |
|---|---|
| LBD | OpenCV `BinaryDescriptorMatcher` |
| MSLD | 浮点描述子，推荐 BF-L2 或 FLANN |
| line-SIFT | SIFT-like 浮点描述子，推荐 BF-L2 或 FLANN |
| LLD / 学习型描述子 | 根据输出向量类型选择 Hamming、L2、cosine 或近似最近邻 |

当前 LBD 已固定使用 OpenCV 专属匹配器，因此 YAML 不再保留 `matcher: BF`，避免误导。

## 9. 汇报时可强调的设计点

1. 平台不是单算法程序，而是可扩展实验框架。
2. 点特征法和结构法共用 `BasePipeline`，但通过子类区分阶段逻辑。
3. 所有算法组件都通过接口和工厂创建，便于横向比较。
4. YAML 配置可以快速切换算法组合，适合批量实验。
5. 输出目录自动按方法族、pipeline 和样本分层，方便保存实验记录。
6. 线结构方法已经从简单端点 RANSAC 改为基于线段中心一致性的仿射估计，避免把线段端点误当作稳定点对应。
7. 后续可以继续扩展 MSLD、line-SIFT、深度线描述子和更多评测指标。

## 10. 当前局限与后续计划

当前局限：

1. 结构法目前主要输出平移仿射，旋转和尺度较大的线结构配准仍需增强。
2. 线描述子当前重点实现 LBD，MSLD、line-SIFT、学习型描述子尚未接入。
3. 评测指标已有框架，但部分指标实现和展示仍可继续完善。
4. 部分历史注释存在编码问题，建议后续统一为 UTF-8。

后续计划：

1. 扩展更多线描述子，并抽象 `AUTO` 匹配策略。
2. 增加结构法的旋转、尺度估计能力。
3. 完善批量实验的统计图表输出。
4. 增加统一 benchmark 报告，支持不同方法的横向排名。
5. 整理代码注释编码，提升可读性和交接质量。
