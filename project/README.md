# 图像配准实验平台

这是一个基于 YAML 的二维图像配准平台，核心使用 OpenCV、C++17，
并支持可选的 Python 深度学习匹配器。平台按方法族拆分为可替换的
流水线：

- 点特征法：SIFT、SURF、ORB、BRISK、KAZE、AKAZE。
- 结构法：基于边缘、直线和轮廓的流水线。
- 直接法：全局、频域、稀疏和稠密直接配准。
- 学习法：LoFTR、SuperPoint + LightGlue、SuperPoint + SuperGlue。

C++ 流水线负责图像读取、配置加载、几何估计、warp、指标计算、
可视化、批处理执行和结果摘要。学习法通过 Python 脚本读取统一的
matches JSON，然后复用同一套 C++ 几何与评测阶段。

## 目录结构

下面的目录树只使用 ASCII 字符，这样在不同终端和编码环境里都更容易读。

```text
project/
|-- CMakeLists.txt
|-- main.cpp
|-- apps/
|   |-- registration_app.h
|   `-- registration_app.cpp
|-- include/
|   |-- core/
|   |-- data/
|   |-- interfaces/
|   |-- keypoint/
|   |-- structure/
|   |-- direct/
|   |-- learning/
|   |-- matcher/
|   |-- filter/
|   |-- geometry/
|   |-- transform/
|   |-- pipeline/
|   |-- dataset/
|   |-- evaluator/
|   `-- utils/
|-- src/
|   |-- core/
|   |-- data/
|   |-- keypoint/
|   |-- structure/
|   |-- direct/
|   |-- learning/
|   |-- matcher/
|   |-- filter/
|   |-- geometry/
|   |-- transform/
|   |-- pipeline/
|   |-- dataset/
|   |-- evaluator/
|   `-- utils/
|-- configs/
|   |-- keypoint/
|   |-- structure/
|   |-- direct/
|   |-- learning/
|   |-- matcher/
|   |-- filter/
|   |-- geometry/
|   |-- evaluator/
|   `-- pipeline/
|       |-- keypoint/
|       |-- structure/
|       |-- direct/
|       |-- learning/
|       `-- batch/
|-- tools/
|   `-- deep/
|       |-- learning_backend.py
|       |-- loftr_infer.py
|       |-- superpoint_lightglue_infer.py
|       `-- superpoint_superglue_infer.py
|-- third_party/
|   |-- LightGlue/
|   `-- SuperGluePretrainedNetwork/
|-- datasets/
|-- outputs/
`-- build-mingw/
```

完整目录说明见 `PROJECT_DIRECTORY_STRUCTURE_CN.md`。

## 构建

当前本地构建目标是 MinGW：

```powershell
cmake --build project/build-mingw
```

主程序：

```text
project/build-mingw/bin/registration_app.exe
```

核心依赖：

- OpenCV 4.x，包括项目用到的 contrib 模块。
- yaml-cpp。
- 支持 C++17 的 CMake。

## 单次运行

运行一个 pipeline YAML：

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/keypoint/sift_pipeline.yaml
```

按方法族的示例：

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/keypoint/sift_pipeline.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/structure/line_pipeline.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/direct/frequency_direct_pipeline.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/learning/loftr_learning_pipeline.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/learning/superpoint_lightglue_learning_pipeline.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/learning/superpoint_superglue_learning_pipeline.yaml
```

每个 pipeline 也可以通过 `io` 块覆盖输入图像和输出目录。

## 批量运行

批处理配置位于：

```text
project/configs/pipeline/batch/
```

示例：

```powershell
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/batch_keypoint.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/batch_structure.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/batch_direct.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/batch_learning.yaml
project/build-mingw/bin/registration_app.exe project/configs/pipeline/batch/compare_direct.yaml
```

批处理会扫描配置的数据集根目录，对每个样本运行指定的单次 pipeline，
并写出汇总 CSV。

## 学习法

学习法流水线的数据流如下：

```text
LearningPipeline
  -> PythonLearningMatcher
  -> learning_backend.py in single or worker mode
  -> matches JSON
  -> C++ correspondence view with source LEARNING
  -> geometry estimation
  -> warp, metrics, visualization, summaries
```

已实现的学习方法：

- `LOFTR`: backed by Kornia LoFTR.
- `SUPERPOINT_LIGHTGLUE`: backed by SuperPoint + LightGlue.
- `SUPERPOINT_SUPERGLUE`: backed by SuperPoint + SuperGlue.

Python 主入口是：

```text
tools/deep/learning_backend.py
```

它支持两种模式：

- `single`: start Python, load one model, process one image pair, then exit.
- `worker`: keep Python alive, load the model once, and process many image pairs.

三个方法专属脚本仍可作为兼容和调试入口，但学习法 YAML 现在统一使用
后端封装。

当前学习法 YAML 指向：

```text
C:/Users/wangchenyu/AppData/Local/Python/bin/python.exe
```

这样可以避开 WindowsApps 的 Python 别名，直接启动真实解释器。

学习脚本已经依赖的 Python 包包括：

- torch
- torchvision
- opencv-python
- kornia
- certifi

仅作为源码依赖放在 `third_party/` 下、不会被 CMake 编译的目录包括：

```text
project/third_party/LightGlue
project/third_party/SuperGluePretrainedNetwork
```

Python 脚本会自动把这些目录加入 `sys.path`。

## 输出

单次运行输出位于：

```text
project/outputs/single/<method_family>/<pipeline>/
```

批量输出位于：

```text
project/outputs/batch/<method_family>/<pipeline>/
```

常见输出文件包括：

- original images
- all-match visualization
- inlier visualization
- warped image
- blend image
- false-color overlay image
- `run_summary.txt`
- `run_summary.json`
- batch `summary.csv`

## 扩展

常见扩展点：

- 通过实现 `IKeypointExtractor` 并注册到工厂中，新增点特征提取器。
- 通过结构接口新增结构提取器或关联器。
- 通过 `IDirectAligner` 和 direct pipeline 配置新增直接法配准器。
- 通过输出统一 matches JSON 的 Python 脚本新增学习法匹配器。
- 通过 geometry YAML 切换或新增几何估计器。

共享的 `RegistrationContext` 会在各个阶段之间携带图像数据、匹配、几何、
warp 结果、指标和运行摘要。
