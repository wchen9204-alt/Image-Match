# 项目目录结构

本文只记录 `project/` 的源码、配置、文档及关键生成目录；构建缓存、运行输出和第三方目录中的详细文件不展开。目录和文件名以当前工作区为准。

```text
project/
├── apps/                         # 命令行调度、文本/JSON 摘要和 CSV 写出
├── build/                        # 通用 CMake 构建目录（生成物）
├── build-mingw/                  # MinGW 构建目录（生成物）
├── build-msvc-opencv412/         # MSVC + OpenCV 4.12 构建目录（生成物）
├── configs/                      # YAML 配置
│   ├── direct/                   # 直接法参数
│   ├── evaluator/                # 评测指标与基准
│   ├── filter/                   # 匹配过滤器，含刚体一致性和距离预过滤
│   ├── geometry/                 # 仿射、单应、刚体、相似变换估计
│   ├── keypoint/                 # AKAZE、BRISK、KAZE、ORB、SIFT、SURF
│   ├── learning/                 # LoFTR、SuperPoint + LightGlue/SuperGlue
│   ├── matcher/                  # BF 与 FLANN 匹配器
│   ├── pipeline/
│   │   ├── batch/                # 批处理与横向比较
│   │   ├── direct/               # 含 frequency_compare/ 子目录
│   │   ├── keypoint/
│   │   ├── learning/
│   │   └── structure/
│   ├── structure/                # 直线和轮廓结构配置
│   ├── validation/               # 质量验证 profile
│   ├── logging.yaml
│   └── preprocess.yaml
├── datasets/                     # 小写 testNN 样本目录；源/目标文件可为 source/target 或 moving/reference
├── docs/                         # 中文开发、实验和判定说明
├── include/                      # C++ 头文件
│   ├── core/ data/ dataset/ direct/ evaluator/ filter/ geometry/
│   ├── interfaces/ keypoint/ learning/ matcher/ pipeline/ structure/
│   ├── transform/
│   └── utils/
├── outputs/                      # single/、batch/、compare/ 运行结果（生成物）
├── src/                          # 与 include/ 模块对应的 C++ 实现
├── third_party/                  # LightGlue、SuperGluePretrainedNetwork 等第三方源码
├── tmp/                          # 临时生成物
├── tools/                        # 报告脚本及深度学习推理后端
├── .clang-format
├── CMakeLists.txt
├── main.cpp
└── README.md
```

## 维护约定

- 新增、删除、移动或重命名核心模块、配置目录、方法族或数据集约定时，更新本文档。
- 构建缓存、运行输出和第三方依赖的内部文件不在本文逐项维护。
