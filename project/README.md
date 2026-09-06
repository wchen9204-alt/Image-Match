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
`-- build-msvc-opencv412/
```

完整目录说明见 `PROJECT_DIRECTORY_STRUCTURE_CN.md`。

## 构建

当前默认构建目标是 MSVC + OpenCV 4.12。构建目录使用 `NMake Makefiles`，因此必须先加载 Visual Studio 的 MSVC 开发环境。

在“x64 Native Tools Command Prompt for VS 2022”中执行：

```cmd
cmake --build project/build-msvc-opencv412
```

如果从普通 PowerShell 执行，请使用：

```powershell
cmd.exe /d /c 'call "E:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 && cmake --build project/build-msvc-opencv412'
```

如果 Visual Studio 安装在其他目录，请相应替换 `VsDevCmd.bat` 的路径。

主程序：

```text
project/build-msvc-opencv412/bin/registration_app.exe
```

如使用 MinGW 构建，也可以将上述路径中的 `build-msvc-opencv412` 替换为 `build-mingw`。

核心依赖：

- OpenCV 4.x，包括项目用到的 contrib 模块。
- yaml-cpp。
- 支持 C++17 的 CMake。

## 单次运行

运行一个 pipeline YAML：

```powershell
project/build-msvc-opencv412/bin/registration_app.exe project/configs/pipeline/keypoint/orb_pipeline.yaml
```

按方法族的示例：

```powershell
project/build-msvc-opencv412/bin/registration_app.exe project/configs/pipeline/keypoint/orb_pipeline.yaml
project/build-msvc-opencv412/bin/registration_app.exe project/configs/pipeline/structure/line_pipeline.yaml
project/build-msvc-opencv412/bin/registration_app.exe project/configs/pipeline/direct/frequency_direct_pipeline.yaml
project/build-msvc-opencv412/bin/registration_app.exe project/configs/pipeline/learning/loftr_learning_pipeline.yaml
```

每个 pipeline 也可以通过 `io` 块覆盖输入图像和输出目录。

## 批量运行

批处理配置位于：

```text
project/configs/pipeline/batch/
```

示例：

```powershell
project/build-msvc-opencv412/bin/registration_app.exe project/configs/pipeline/batch/batch_keypoint.yaml
project/build-msvc-opencv412/bin/registration_app.exe project/configs/pipeline/batch/batch_structure.yaml
project/build-msvc-opencv412/bin/registration_app.exe project/configs/pipeline/batch/batch_direct.yaml
project/build-msvc-opencv412/bin/registration_app.exe project/configs/pipeline/batch/batch_learning.yaml
project/build-msvc-opencv412/bin/registration_app.exe project/configs/pipeline/batch/compare_direct.yaml
```

批处理会扫描配置的数据集根目录，对每个样本运行指定的单次 pipeline，
并写出汇总 CSV。
