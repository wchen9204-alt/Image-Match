# 图像配准实验平台 — 项目详解

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                     RegistrationApp                      │
│              CLI → single / batch 路由                   │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│                    IPipeline (接口)                       │
│              BasePipeline (模板方法骨架)                   │
│         ┌──────────────┴──────────────┐                 │
│         │ KeypointPipeline            │ StructurePipeline │
│         │  (点特征)                    │  (结构特征)        │
│         └─────────────────────────────┘                 │
└─────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│               RegistrationContext (共享状态)              │
│  images │ keypoint_data │ keypoint_match_data            │
│  structure_data │ structure_match_data                   │
│  geometry_data │ transform_data │ evaluation_data         │
│  result (RegistrationResult)                            │
└─────────────────────────────────────────────────────────┘
```

**核心设计思想**：将"配准"拆为 7 个可替换的阶段，每个阶段由接口定义、工厂创建、YAML 配置选择具体实现。`RegistrationContext` 是唯一的共享数据载体，阶段之间只通过它交换数据。

## 2. 一次完整配准的 7 个阶段

`BasePipeline::run()` 定义了不可变的执行顺序：

```
loadImages() → runExtraction() → runAssociation()
    → runEstimation() → runWarp() → validateWarpQuality()
    → saveOutputs()
```

任何一个阶段返回 `false` 都会导致整条流水线失败。

### 阶段 1：loadImages — 图像读取

**文件**：`src/pipeline/base_pipeline.cpp` 中的 `BasePipeline::loadImages()`

**流程**：
1. 从 `ctx.image1_path` / `ctx.image2_path` 读取图像
2. 检测图像位深——高位深 TIFF（如 16-bit 医学图像）自动归一化到 8-bit
3. 生成 `ctx.images.first`（BGR）、`ctx.images.first_gray`（灰度）和对应的 second

**为什么这样设计**：后续的特征提取器接收统一的灰度图，避免每个提取器自己做一遍 `cvtColor`。BGR 保留给可视化和 warp 使用。

### 阶段 2：runExtraction — 特征/结构提取

**KeypointPipeline** → 创建 `IKeypointExtractor`，调用 `extract(ctx)`：
- 对 first_gray / second_gray 检测关键点 + 计算描述子
- 写入 `ctx.keypoint_data.first/second`
- `KeypointImageData` 包含：`vector<KeyPoint> keypoints`、`Mat descriptors`、`KeypointType type`、`NormType norm_type`

**StructurePipeline** → 创建 `IStructureExtractor`，调用 `extract(ctx)`：
- `LineExtractor`：Canny 边缘 → Hough/LSD/FLD 检测线段 → 写入 `ctx.structure_data.first.lines`（`vector<Vec4i>`）
- `EdgeExtractor`：Canny/Sobel/LoG/Laplacian → 写入 `ctx.structure_data.first.response`（二值/灰度响应图）
- `ContourExtractor`：Canny + findContours → 写入 `ctx.structure_data.first.contours`

**LineExtractor 内部细节**（`src/structure/line_extractor.cpp`）：

支持的 4 种检测方法：

| 方法 | OpenCV 接口 | 输出特点 |
|------|-----------|---------|
| HOUGH_LINES | `cv::HoughLines` + 裁剪 | 无限直线裁剪到图像边界 |
| HOUGH_LINES_P | `cv::HoughLinesP` | 概率霍夫，直接输出有限线段 |
| LSD | `cv::line_descriptor::LSDDetector`（优先）→ `cv::LineSegmentDetector`（回退） | 专用线段检测器，端点精度高 |
| FLD | `cv::ximgproc::FastLineDetector` | 快速直线检测 |

后处理流水线：
1. **去重**（`deduplicateLines`）：按长度降序排列，逐一检查角度差和线段间距，剔除重复线段
2. **限数**（`limitLines`）：保留最长 N 条线段（默认 300）
3. **渲染**（`renderLineResponse`）：在黑色画布上绘制白色线段，生成结构响应图

### 阶段 3：runAssociation — 匹配/关联

#### 3.1 点特征流水线

```
runMatch() → 描述子匹配 → raw_knn (vector<vector<DMatch>>)
    → runFilters() → IFilter 链 → filtered (vector<DMatch>)
```

- `BFMatcher`：暴力匹配，支持 L1/L2/Hamming 距离
- `FlannMatcher`：FLANN 近似最近邻，适合大数据量浮点描述子

#### 3.2 结构特征流水线

```
associator->associate(ctx) → raw_matches_knn / filtered_matches
    → runFilters() → IFilter 链 → line_matches
```

6 种结构关联器：

| 关联器 | 原理 | 适用场景 |
|--------|------|---------|
| PhaseCorrelate | 频域相位相关 | 平移为主，边缘/轮廓响应图 |
| Chamfer | 距离变换 + 平均距离最小化 | 边缘响应图 |
| Hausdorff | 分位数 Hausdorff 距离 | 含噪声的结构响应图 |
| ICP | 迭代最近点 | 结构点集 |
| LineSegment | 角度/长度/位移几何投票 | 线段几何 baseline |
| LineDescriptor | LBD 描述子 + 几何一致性 | 线段描述子匹配 |

#### 3.3 IFilter 过滤链（两种 pipeline 共享）

```
raw_knn / raw_matches_knn  →  [RatioTest] → [DistanceThreshold] → [MinDistance] → ...
                                    ↓
                              filtered_matches
```

6 种过滤器都实现了 `IFilter::apply(RegistrationContext& ctx)`。通过检测 `ctx` 中哪个数据域非空来自动判断操作的是点特征还是结构法——这就是为什么同一个 `ratio_test.yaml` 可以同时用于点特征和线特征流水线。

| 过滤器 | 原理 | 结构法支持 |
|--------|------|:---:|
| RatioTest | Lowe's ratio: d1/d2 < threshold | ✅ |
| DistanceThreshold | distance ≤ max_distance | ✅ |
| MinDistance | distance ≤ max(multiplier × minDist, minCutoff) | ✅ |
| DistanceDistribution | 均值+标准差 或 分位数 | ✅ |
| CrossCheck | 双向匹配一致性 | ❌（需描述子矩阵） |
| GMS | 网格运动统计 | ❌（需 KeyPoint 空间坐标） |

### 阶段 4：runEstimation — 几何估计

#### 4.1 点特征流水线

直接调用 `_geometry->estimate(ctx)`：
- 从 `keypoint_data` 提取点对 → RANSAC 估计 → `geometry_data.A`（2×3）或 `geometry_data.H`（3×3）

#### 4.2 结构特征流水线（两次重构后的设计）

```cpp
// structure_pipeline.cpp
// 几何模型类型由 YAML 决定（Rigid/Affine/Similarity/Homography）
// 不再被关联器内部的纯平移覆盖
if (_geometry && !ctx.structure_match_data.line_matches.empty()) {
    prepareLineEndpointMatches(ctx);  // 线段 → 端点对
    _geometry->estimate(ctx);         // RANSAC
    promoteLineInliersFromEndpointMatches(ctx);  // 端点内点 → 线内点
}
```

`prepareLineEndpointMatches` 的工作方式：
1. 遍历每条线匹配，提取两个端点坐标
2. 处理线段方向一致性（翻转反向线段，确保端点顺序一致）
3. 将端点填充到 `keypoint_data` 和 `keypoint_match_data.filtered`（复用了点特征的数据结构）
4. 几何估计器看到的就是"点对"，无需知道来源是线

`promoteLineInliersFromEndpointMatches`：检查每条线匹配的两个端点是否都在 RANSAC 内点集中 → 两个端点都是内点 → 这条线是内点。

**注意**：线端点的定位精度（2-5px）远低于 SIFT 关键点（<1px），因此 `rigid.yaml` 的 `ransacReprojThreshold` 已从 3.0 放宽到 8.0，`minInliers` 从 3 降到 2。

### 阶段 5：runWarp — 图像变换

在 `BasePipeline::runWarp()` 中统一处理：

根据 `geometry_data` 的类型创建对应的 warper：
- `HOMOGRAPHY` → `PerspectiveWarper`（3×3 透视变换 `warpPerspective`）
- `AFFINE / RIGID / SIMILARITY` → `AffineWarper`（2×3 仿射变换 `warpAffine`）

调用 `warper->warp(ctx)` → 写入 `ctx.warped_image`。

### 阶段 6：validateWarpQuality — 质量验证

**文件**：`src/pipeline/base_pipeline.cpp` 中的 `BasePipeline::validateWarpQuality()`

双重检查，YAML 中可独立开关：

**IoU 检查**（几何重合率）：
```
warpedMask ∩ targetMask / warpedMask ∪ targetMask >= min_iou (默认 0.20)
```
- 对两幅图做前景二值化（threshold > foreground_threshold）
- 计算前景 mask 的交并比
- 重合率过低 → 配准基本失败

**NMAD 检查**（重叠区光度差）：
```
mean(|warped - target|) / 255 <= max_nmad (默认 0.15)，仅重叠区域
```
- 只在 IoU 重叠区域内计算
- 平均绝对差归一化到 [0, 1]
- 光度差异过大 → 即使几何对齐了，内容也不一致

任一不达标 → 整个配准判定失败。两项都不开 → 跳过验证（兼容旧配置）。

相关配置参数在 `PipelineConfig` 中：
```cpp
bool validate_warp_overlap = false;        // IoU 检查开关
double min_warp_overlap_iou = 0.20;        // IoU 阈值
bool validate_warp_photometric = false;    // NMAD 检查开关
double max_warp_photometric_error = 0.15;  // NMAD 阈值
```

### 阶段 7：saveOutputs — 输出保存

按 `outputs/{single|batch}/{keypoint|structure}/{pipeline}/{sample}/` 结构写入：

| 目录/文件 | 内容 |
|----------|------|
| `originals/` | 原始 source.png, target.png |
| `keypoints/` 或 `structures/` | 特征点绘制图 / 结构响应图 |
| `matches/` | all_match（全部匹配连线）、inlier_match（内点连线） |
| `warped/` | 变换后的 source 图像 |
| `blend/` | warped 与 target 的半透明叠加 |
| `summary.txt` | 可读的文本摘要 |
| `summary.json` | 结构化 JSON 数据 |
| `summary.csv` | 批量汇总表（含 IoU、NMAD、各阶段耗时等） |

## 3. 数据流全貌

```
                    ┌──────────────────────┐
                    │   RegistrationContext  │
                    └──────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                      ▼
  ImagePairData        KeypointData          StructureData
  ┌──────────┐        ┌─────────────┐       ┌──────────────┐
  │first     │        │first:       │       │first:        │
  │  BGR     │        │  keypoints  │       │  response    │
  │  gray    │        │  descriptors│       │  lines       │
  │second    │        │second: ...  │       │  contours    │
  │  BGR     │        │type         │       │second: ...   │
  │  gray    │        │norm_type    │       │type          │
  └──────────┘        └─────────────┘       └──────────────┘
                              │                      │
                              ▼                      ▼
                    KeypointMatchData       StructureMatchData
                    ┌──────────────┐       ┌──────────────┐
                    │raw_knn       │       │raw_matches_knn│
                    │filtered      │       │filtered_matches│
                    │inlier_mask   │       │line_matches   │
                    │inliers       │       │inlier_line_.. │
                    └──────────────┘       │translation    │
                              │            │affine         │
                              ▼            │valid/score    │
                      GeometryData         └──────────────┘
                    ┌──────────────┐              │
                    │type          │              ▼
                    │H (3x3)       │        TransformData
                    │A (2x3)       │       ┌──────────────┐
                    │num_inliers   │       │type          │
                    │inlier_ratio  │       │M (3x3)       │
                    │valid         │       │valid         │
                    └──────────────┘       └──────────────┘
                              │
                              ▼
                       warped_image
                              │
                              ▼
                     RegistrationResult
                    ┌──────────────┐
                    │success       │
                    │message       │
                    │warp_overlap..│
                    │warp_photo... │
                    │t_*_ms        │
                    └──────────────┘
```

**关键约定**：
- 每个阶段读取的是**上一阶段写入**的字段。例如 `GeometryData` 只由 `runEstimation()` 写入，`runWarp()` 和 `validateWarpQuality()` 只读取。
- `RegistrationResult` 是跨阶段汇总字段，各阶段逐步填充。
- `resetStages()` 在每次 `run()` 开始时通过 `ctx.reset()` 清零所有运行时数据。

## 4. 工厂模式 — 如何创建算法组件

**文件**：`src/core/factory.cpp`

所有算法组件都通过 `Factory` 的静态方法创建，没有一个 `new` 出现在业务代码中：

```cpp
// YAML 中 type 字段 → 枚举 → switch-case → make_shared<具体类>
static std::shared_ptr<IKeypointExtractor> createKeypointExtractor(const YAML::Node& cfg);
static std::shared_ptr<IStructureExtractor> createStructureExtractor(const YAML::Node& cfg);
static std::shared_ptr<IStructureAssociator> createStructureAssociator(const YAML::Node& cfg);
static std::shared_ptr<IMatcher> createMatcher(const YAML::Node& cfg);
static std::shared_ptr<IFilter> createFilter(const YAML::Node& cfg);
static std::shared_ptr<IGeometryEstimator> createGeometryEstimator(const YAML::Node& cfg);
```

每种方法都用 `methodKey()` 做字符串归一化（大写 + 只保留字母数字），确保 `"line_descriptor"`、`"Line_Descriptor"`、`"LINEDESCRIPTOR"`、`"LBD"` 都能正确匹配。

**以 LineDescriptorAssociator 的创建为例**：

```cpp
// factory.cpp
if (key == "LINEDESCRIPTOR" || key == "LINEDESCRIPTORS" || key == "LBD") {
    const YAML::Node params =
        all_params && all_params["line_descriptor"]
            ? all_params["line_descriptor"]   // 读取 line_descriptor 子节点
            : assoc_cfg;                      // 回退到整个关联器配置
    return std::make_shared<LineDescriptorAssociator>(params);
}
```

## 5. 配置系统 — YAML 如何驱动一切

### 5.1 配置层次

```
line_pipeline.yaml          ← 顶层：编排子配置路径 + IO + 可视化 + 验证
  ├── structure: line.yaml  ← 结构层：提取器参数 + 关联器参数
  ├── geometry: rigid.yaml  ← 几何层：RANSAC 参数
  └── filters:              ← 过滤层：ratio_test.yaml, distance_threshold.yaml
```

### 5.2 路径解析

`Config::resolvePath(base_dir, relative_path)` 的搜索顺序：
1. `base_dir / relative_path`（相对 pipeline YAML 目录）
2. `cwd / relative_path`（相对当前工作目录）
3. 从 `base_dir` 向上遍历 4 级父目录
4. 回退到 `base_dir / relative_path`（即使文件不存在，让调用方报错）

### 5.3 配置加载流程

```
RegistrationApp::run()
  → Config::loadPipeline(pipeline_yaml)
    → 解析 io / visualization / validation / filters
    → resolvePath 得到所有子配置的绝对路径
    → 返回 PipelineConfig（纯数据 struct）

  → createPipeline() → configure(pipelineConfig)
    → configureStages():
      → Config::load(structure_path) → Factory::createStructureExtractor()
      → Config::load(geometry_path) → Factory::createGeometryEstimator()
      → for each filter_path: Config::load() → Factory::createFilter()
```

### 5.4 PipelineConfig 结构

```cpp
struct PipelineConfig {
    std::string name;
    std::filesystem::path keypoint_path;
    std::filesystem::path structure_path;
    std::filesystem::path matcher_path;
    std::filesystem::path geometry_path;
    std::vector<std::filesystem::path> filter_paths;
    std::filesystem::path image1_path, image2_path, output_dir;

    // 可视化
    bool draw_matches = true;
    int max_matches_drawn = 100;
    bool warp = true;

    // 质量验证
    bool validate_warp_overlap = false;
    double min_warp_overlap_iou = 0.20;
    bool validate_warp_photometric = false;
    double max_warp_photometric_error = 0.15;
};
```

## 6. LineDescriptorAssociator 详解

这是当前最复杂的关联器，体现了线匹配的完整链路。

### 6.1 完整流程

```
LineExtractor
  → ctx.structure_data.first.lines (vector<Vec4i>)
       │
       ▼
LineDescriptorAssociator::associate()
  │
  ├─ 1. Vec4i → KeyLine (toKeyLine)
  │     把内部线段格式转为 OpenCV line_descriptor 模块需要的 KeyLine
  │     注意：不设置 size 字段，让 OpenCV 使用默认带宽（lineLength × 0.5）
  │
  ├─ 2. computeLbd(gray, keyLines, descriptors)
  │     BinaryDescriptor::compute() → 二进制 LBD 描述子矩阵
  │     注意：compute() 可能过滤掉无效 KeyLine（边界太近等）
  │     诊断日志：srcDesc=R×C (keys=K) — R<K 说明有 KeyLine 被过滤
  │
  ├─ 3. knnMatchLbd(descriptors, keys, knn_k=8)
  │     BinaryDescriptorMatcher::knnMatch(k=8)
  │     → 描述子行索引 → class_id 重映射回原始线段索引
  │     → raw_matches_knn (vector<vector<DMatch>>)
  │
  ├─ 4. filterGeometricConsistent()  ← 几何一致性（关键质量保证）
  │     ┌─ 粗筛：方向差 ≤30° + 长度比 ≥0.30
  │     ├─ 投票：中心位移一致性（每个候选投票，选支持最多的平移向量）
  │     ├─ 筛选：只保留与最佳平移一致的候选
  │     └─ 去重：source/target 一对一（每条线只保留最佳匹配）
  │     → selected (vector<DMatch>)
  │
  └─ 5. 写入
       md.filtered_matches = selected
       md.line_matches = selected
       md.valid = line_matches.size() >= min_matches
```

### 6.2 几何一致性过滤原理

这是线匹配质量的关键。纯 LBD 描述子的 top-1 匹配假阳性很高（二进制描述子区分力有限），但加了几何约束后质量大幅提升：

1. **方向一致性**：两条匹配的线段方向差应在阈值内（<30°）。无向线段角度归一化到 [0, π)。
2. **长度比**：匹配线段的长度比不能太小（≥0.30），过滤尺度差异过大的匹配。
3. **中心位移投票**：核心创新——在正确的配准下，所有匹配线段的中心位移应一致。对位移向量投票，找到全局一致的平移，剔除不一致的匹配。

```
源图线段  ────────────────→  目标图线段
 中心(x1,y1)     位移(dx,dy)     中心(x2,y2)

正确的匹配：所有线对的(dx,dy)相近 → 投票集中 → 通过
错误的匹配：位移分散 → 投票不集中 → 被剔除
```

### 6.3 配置参数

```yaml
line_descriptor:
  descriptor: LBD                   # 描述子类型
  knn_k: 8                          # KNN 邻居数
  min_matches: 2                    # 最少匹配数
  geometric_filter: true            # 启用几何一致性筛选
  angle_threshold_deg: 30.0         # 方向差阈值（度）
  min_length_ratio: 0.30            # 最小长度比
  shift_consistency_threshold: 30.0 # 位移一致性阈值（像素）
```

### 6.4 诊断日志

运行时关注这几行日志来诊断匹配质量：

```
LineDescriptorAssociator input: srcLines=N, dstLines=M
LineDescriptorAssociator LBD: srcDesc=R x C (keys=K), dstDesc=...
LineDescriptorAssociator KNN: raw_matches_knn groups=G
```

| 对比 | 正常 | 异常 | 问题所在 |
|------|------|------|---------|
| keys=K vs srcLines=N | K ≈ N | K ≪ N | BinaryDescriptor 过滤了太多 KeyLine |
| srcDesc=R vs keys=K | R = K | R ≪ K | 描述子计算本身失败 |
| groups=G vs srcDesc=R | G ≈ R | G ≪ R | KNN 匹配或重映射有问题 |

## 7. StructurePipeline 的几何估计路径

两次重构后的设计——几何模型始终由 YAML 配置的估计器决定：

```
runEstimation()
  │
  ├─ 有 _geometry 且 line_matches 非空？
  │   YES → prepareLineEndpointMatches()
  │          - 遍历每条线匹配，提取两个端点
  │          - 处理线段方向一致性（翻转反向线段）
  │          - 填充到 keypoint_data + keypoint_match_data.filtered
  │          ↓
  │          _geometry->estimate(ctx)
  │          - RANSAC 估计（模型类型由 YAML 决定：Rigid/Affine/Similarity/Homography）
  │          - ransacReprojThreshold: 8.0px（放宽以适应线端点精度）
  │          - minInliers: 2
  │          ↓
  │          promoteLineInliersFromEndpointMatches()
  │          - 线的两个端点都在 RANSAC 内点集 → 这条线是内点
  │          - 更新 inlier_line_matches 和 score
  │
  └─ 无线匹配？（响应图关联器如 PhaseCorrelate）
      → 使用关联器自身的平移结果作为兜底
```

**重构历史**：
1. **第一版**：关联器硬编码纯平移 affine → `runEstimation()` 检测到 `md.affine` 非空就跳过几何估计器 → 用户配置 `rigid.yaml` 被忽略
2. **修正版**：移除 `affine.empty()` 条件 → 关联器不再输出 affine → 几何估计器始终生效
3. **问题**：线端点定位不准，3px RANSAC 阈值太紧 → 放宽到 8px
4. **IFilter 扩展**：关联器只保留几何一致性过滤（线特有），通用过滤（RatioTest 等）交给 IFilter 链

## 8. 批量运行流程

```
RegistrationApp::runBatch()
  │
  ├─ 1. 加载 batch YAML
  │     pipeline: ../structure/line_pipeline.yaml
  │     dataset:
  │       root: ../../../datasets
  │       pattern_source: source
  │       pattern_target: target
  │
  ├─ 2. 加载 pipeline 模板配置
  │     Config::loadPipeline(pipeline_yaml) → base_cfg
  │
  ├─ 3. 扫描数据集
  │     DatasetLoader::load()
  │       - include 为空 → 自动扫描 root 下所有子目录
  │       - 每个子目录 = 一个 Sample
  │       - resolveImage(): 按 pattern_source/target 匹配图像文件
  │       - tryLoadGroundTruth(): 可选，查找 H_gt.txt
  │
  ├─ 4. 逐样本运行
  │     for each sample:
  │       create new pipeline
  │       pipeline->configure(base_cfg)         # 重新创建所有组件
  │       override ctx.image1_path/image2_path/output_dir
  │       pipeline->run(ctx)
  │       收集 result + evaluation
  │
  └─ 5. 输出汇总
        writeSummaryCsv() → summary.csv
        buildSummaryText() → 控制台输出
        控制台: "Batch summary: N / M samples succeeded."
        退出码: 全部成功→0，有失败→1
```

## 9. 关键数据结构速查

| 结构 | 核心字段 | 写入阶段 | 读取阶段 |
|------|---------|---------|---------|
| `ImagePairData` | `first/second` (BGR + gray) | loadImages | 所有后续 |
| `KeypointImageData` | `keypoints`, `descriptors`, `type`, `norm_type` | runExtraction | match, filter |
| `KeypointMatchData` | `raw_knn`, `filtered`, `inlier_mask`, `inliers` | runMatch, runFilters | geometry |
| `StructureImageData` | `response`, `lines`, `contours` | runExtraction | associate |
| `StructureMatchData` | `raw_matches_knn`, `filtered_matches`, `line_matches`, `inlier_line_matches`, `translation`, `affine`, `valid`, `score` | associate, runFilters | estimation |
| `GeometryData` | `type`, `A` (2x3), `H` (3x3), `num_inliers`, `inlier_ratio`, `valid` | runEstimation | warp |
| `TransformData` | `type`, `M` (3x3), `valid` | runWarp | validate |
| `RegistrationResult` | `success`, `message`, `warp_overlap_iou`, `warp_photometric_error`, `num_raw_matches`, `num_filtered_matches`, `num_inliers`, `t_*_ms` | 各阶段 | summary |

### StructureMatchData 字段详解

```cpp
struct StructureMatchData {
    std::string method;                              // 关联方法名
    cv::Point2d translation{0.0, 0.0};              // 平移估计
    cv::Mat affine;                                  // 2x3 仿射矩阵
    double score = 0.0;                              // 匹配得分
    bool valid = false;                              // 结果是否有效
    std::string message;                             // 状态信息

    // 过滤链工作区（与 KeypointMatchData 对应）
    std::vector<std::vector<cv::DMatch>> raw_matches_knn;  // 原始 KNN（供 RatioTest）
    std::vector<cv::DMatch> filtered_matches;              // 过滤后的匹配

    // 最终结果
    std::vector<cv::DMatch> line_matches;           // 过滤后的线匹配
    std::vector<cv::DMatch> inlier_line_matches;    // 几何估计后的内点线匹配
};
```

## 10. 扩展指南

### 10.1 加一个新的线描述子（如 MSLD）

1. 在 `LineDescriptorAssociator` 的构造函数中支持新的 `descriptor` 选项
2. 在 `associate()` 中增加 MSLD 描述子计算分支
3. 根据描述子类型选择匹配器：
   - 浮点（MSLD、line-SIFT）→ `cv::BFMatcher` (L2) 或 `cv::FlannBasedMatcher`
   - 二进制（LBD）→ `cv::line_descriptor::BinaryDescriptorMatcher`
4. 在 `line.yaml` 的 `line_descriptor.descriptor` 中配置

### 10.2 加一个新的几何一致性过滤器

1. 新建 `include/filter/geometric_consistency_filter.h`
2. 继承 `IFilter`，实现 `apply(ctx)`——在线段空间做方向/长度/位移检查
3. 在 `Factory::createFilter` 中注册字符串映射
4. 新增 `configs/filter/geometric_consistency.yaml`
5. 在 pipeline YAML 的 `filters` 列表中添加

### 10.3 加一个直接法配准

1. 新增 `IStructureAssociator` 的子类（如 `DirectAssociator`）
2. 实现 `associate(ctx)`——直接优化像素误差
3. 在 `Factory::createStructureAssociator` 中注册
4. 新增 `configs/structure/direct.yaml`
5. 新增 `configs/pipeline/structure/direct_pipeline.yaml`
6. 无需修改 `StructurePipeline`——它通过关联器接口工作

### 10.4 通用的扩展步骤模板

对于任何新算法组件，扩展步骤都是：
1. **接口**：继承对应接口，实现核心方法
2. **工厂**：在 `Factory` 对应方法中注册字符串→类的映射
3. **配置**：新增 `configs/` 下对应 YAML
4. **流水线**：在 pipeline YAML 中引用新配置
