# 配准方法体系说明

本文从“方法”角度介绍平台已经实现的配准技术路线，包括点特征法、结构特征法、匹配方法、过滤方法、几何估计方法和评测输出。适合横向比较不同算法组合。

## 1. 总体方法分类

平台当前把二维图像配准分为两大方法族：

| 方法族 | 核心思想 | 对应 pipeline |
|---|---|---|
| 点特征法 | 检测局部关键点，计算描述子，匹配点对，再估计几何变换 | `KeypointPipeline` |
| 结构特征法 | 提取边缘、直线、轮廓等结构信息，建立结构关联，再估计变换 | `StructurePipeline` |

二者共享同一个通用执行框架：

```text
读取图像 -> 提取 -> 关联/匹配 -> 几何估计 -> warp -> 输出
```

区别在于：

1. 点特征法的关联单位是关键点。
2. 结构法的关联单位可以是响应图、点集、线段或轮廓。
3. 点特征法通常依赖描述子匹配和 RANSAC。
4. 结构法可以直接输出平移或仿射，也可以回退到几何估计器。

## 2. 点特征法

点特征法流程：

```text
KeypointExtractor
  -> Descriptor Matcher
  -> Match Filters
  -> Geometry Estimator
  -> Warper
```

### 2.1 已实现点特征

| 方法 | OpenCV 模块 | 描述子类型 | 默认距离 | 特点 |
|---|---|---|---|---|
| SIFT | features2d | 浮点 | L2 | 尺度和旋转鲁棒，稳定性好 |
| SURF | xfeatures2d | 浮点 | L2 | 速度较快，需要 contrib 和 nonfree |
| ORB | features2d | 二进制 | Hamming | 快速，适合实时场景 |
| BRISK | features2d | 二进制 | Hamming | 二进制特征，速度快 |
| KAZE | features2d | 浮点 | L2 | 非线性尺度空间，细节保持较好 |
| AKAZE | features2d | 二进制或浮点 | Hamming 或 L2 | KAZE 的快速版本 |

各方法通过 `configs/keypoint/*.yaml` 配置参数，例如关键点数量、尺度层数、阈值、描述子类型等。

### 2.2 点特征提取逻辑

共同逻辑：

1. 从 YAML 读取参数。
2. 创建对应 OpenCV 特征对象。
3. 对 source 灰度图调用 `detectAndCompute`。
4. 对 target 灰度图调用 `detectAndCompute`。
5. 保存关键点和描述子到 `ctx.keypoint_data`。
6. 设置 `KeypointType` 和 `NormType`，供匹配器自动选择距离。

## 3. 点特征匹配方法

### 3.1 BFMatcher

配置：`configs/matcher/bf.yaml`

支持模式：

| 模式 | 说明 |
|---|---|
| `MATCH` | 每个描述子只返回一个最近邻 |
| `KNN` | 每个描述子返回 K 个最近邻，常用于 ratio test |
| `RADIUS` | 返回距离半径内的所有匹配 |

距离选择：

| 描述子类型 | 距离 |
|---|---|
| SIFT / SURF / KAZE | L2 |
| ORB / BRISK / AKAZE-MLDB | Hamming |

核心逻辑：

```text
根据 ctx.keypoint_data.norm 选择 OpenCV norm
创建 cv::BFMatcher
调用 match / knnMatch / radiusMatch
写入 raw_knn
```

### 3.2 FLANN Matcher

配置：`configs/matcher/flann.yaml`

适用场景：

1. 浮点描述子数量较多时，FLANN 可以提升近似最近邻搜索速度。
2. SIFT、SURF、KAZE 这类浮点描述子更适合 FLANN。

输出同样进入过滤器链。

## 4. 点特征过滤方法

过滤器用于减少误匹配，提升几何估计稳定性。

### 4.1 Ratio Test

配置：`configs/filter/ratio_test.yaml`

思想：如果最近邻距离明显小于次近邻距离，则说明匹配更可信。

逻辑：

```text
对每个 query 的 KNN 匹配:
  best = 最近邻
  second = 次近邻
  如果 best.distance < ratio * second.distance:
      保留 best
```

适合用于 SIFT、SURF、ORB 等大多数描述子。

### 4.2 Cross Check

配置：`configs/filter/cross_check.yaml`

思想：source 到 target 的最佳匹配，反过来也必须是 target 到 source 的最佳匹配。

逻辑：

```text
正向匹配 source -> target
反向匹配 target -> source
只有互相指向的匹配才保留
```

优点是误匹配少，缺点是匹配数量可能减少。

### 4.3 GMS

配置：`configs/filter/gms.yaml`

思想：正确匹配在图像空间中具有局部运动一致性。GMS 把图像划分为网格，统计局部匹配支持度。

适合场景：

1. ORB 等二进制特征产生大量初始匹配。
2. 图像有较明显的局部一致运动。

### 4.4 Distance Threshold

配置：`configs/filter/distance_threshold.yaml`

思想：匹配距离低于固定阈值才保留。

优点：简单可控。

缺点：不同描述子距离尺度不同，需要针对方法调参。

### 4.5 Min Distance Filter

配置：`configs/filter/min_distance.yaml`

思想：根据全局最小匹配距离设置动态阈值。

逻辑：

```text
minDist = 所有匹配中的最小距离
threshold = max(minAbsolute, minDist * ratio)
保留 distance <= threshold 的匹配
```

### 4.6 Distance Distribution Filter

配置：`configs/filter/distance_distribution.yaml`

思想：根据匹配距离的均值和标准差剔除异常大距离。

适合距离分布相对稳定的实验。

## 5. 几何估计方法

几何估计输入是过滤后的点匹配，输出是空间变换模型。

### 5.1 Homography

配置：`configs/geometry/homography.yaml`

OpenCV 方法：`cv::findHomography`

模型：

```text
[x']   [h11 h12 h13] [x]
[y'] ~ [h21 h22 h23] [y]
[1 ]   [h31 h32 h33] [1]
```

适合：

1. 平面场景。
2. 视角变化较明显的图像。
3. 透视变换。

最少匹配数：4 对点。

### 5.2 Affine

配置：`configs/geometry/affine.yaml`

OpenCV 方法：`cv::estimateAffine2D`

模型：

```text
x' = a00*x + a01*y + tx
y' = a10*x + a11*y + ty
```

适合：

1. 旋转、平移、尺度、剪切。
2. 透视变化不强的场景。

最少匹配数：3 对点。

### 5.3 Rigid

配置：`configs/geometry/rigid.yaml`

实现思路：使用部分仿射估计，再约束为刚体变换。

模型包含：

1. 旋转。
2. 平移。
3. 不包含尺度变化和剪切，或尽量抑制尺度影响。

适合相机或物体只发生刚体运动的场景。

### 5.4 Similarity

配置：`configs/geometry/similarity.yaml`

模型包含：

1. 旋转。
2. 统一尺度。
3. 平移。

适合存在整体缩放但没有明显剪切的场景。

### 5.5 鲁棒估计

几何估计通常使用 RANSAC。

RANSAC 逻辑：

```text
随机采样最小点集
估计候选模型
计算所有匹配的重投影误差
误差低于阈值的点视为内点
选择内点最多或误差最小的模型
```

输出：

| 输出 | 说明 |
|---|---|
| `H` | 单应矩阵 |
| `A` | 仿射或部分仿射矩阵 |
| `inliers` | 内点匹配 |
| `inlier_ratio` | 内点比例 |
| `mean_reproj_error` | 平均重投影误差 |

## 6. 结构特征法

结构法适合纹理不明显、边缘或线条较明显的图像。

当前结构类型：

| 类型 | 说明 |
|---|---|
| EDGE | 边缘响应图 |
| LINE | 线段集合和线段响应图 |
| CONTOUR | 轮廓集合 |

结构法流程：

```text
StructureExtractor
  -> StructureAssociator
  -> optional GeometryEstimator
  -> Warper
```

## 7. 边缘结构法

边缘提取方法：

| 方法 | 核心思想 |
|---|---|
| CANNY | 多阶段边缘检测 |
| SOBEL | 梯度幅值 |
| LOG | 高斯平滑后 Laplacian |
| LAPLACIAN | 二阶导数边缘 |

边缘法通常不直接产生一对一匹配，而是把响应图交给结构关联器，例如相位相关、Chamfer、Hausdorff 或 ICP。

## 8. 轮廓结构法

轮廓提取逻辑：

1. 灰度图预处理。
2. 阈值化或边缘化。
3. 调用 `cv::findContours`。
4. 根据面积或长度过滤。
5. 输出轮廓集合和响应图。

轮廓可用于：

1. 形状匹配。
2. 结构点集配准。
3. 后续扩展轮廓描述子。

## 9. 线结构法

线结构是当前结构法重点扩展方向。

### 9.1 线段检测方法

| 方法 | OpenCV 接口 | 特点 |
|---|---|---|
| HoughLines | `cv::HoughLines` | 检测无限直线，需要裁剪到图像边界 |
| HoughLinesP | `cv::HoughLinesP` | 直接输出有限线段 |
| LSD | `cv::createLineSegmentDetector` | 局部线段检测，适合自然图像 |
| FLD | `cv::ximgproc::FastLineDetector` | 快速线段检测 |

当前 `line.yaml` 中默认使用 LSD。

### 9.2 线段后处理

线段检测后会执行：

1. 最小长度过滤。
2. 近重复线段去重。
3. 按长度保留前 `maxLines` 条。
4. 绘制结构响应图。

当前实现优先使用 OpenCV API：

| 功能 | 方法 |
|---|---|
| 线段长度 | `cv::norm` |
| 点到线段距离 | `cv::pointPolygonTest` |

### 9.3 LBD 线描述子法

配置位置：`configs/structure/line.yaml` 的 `association.params.line_descriptor`。

流程：

```text
KeyLine 检测或转换
  -> BinaryDescriptor 计算 LBD
  -> BinaryDescriptorMatcher KNN 匹配
  -> 几何一致性筛选
  -> 输出线段匹配和平移仿射
```

当前关键 OpenCV 函数：

| 功能 | OpenCV API |
|---|---|
| KeyLine 检测 | `cv::line_descriptor::LSDDetector` |
| LBD 描述子 | `cv::line_descriptor::BinaryDescriptor` |
| LBD 匹配 | `cv::line_descriptor::BinaryDescriptorMatcher` |
| 匹配可视化 | `cv::line_descriptor::drawLineMatches` |

注意：

1. LBD 已使用 line_descriptor 模块专属匹配器。
2. YAML 中不再配置 `matcher: BF`。
3. 后续 MSLD、line-SIFT 等描述子可以引入 `matching.type: AUTO`，由描述子类型决定 BF-L2、FLANN、Hamming 或专用匹配器。

### 9.4 LBD 匹配后过滤

LBD 匹配后不是直接用于 RANSAC，而是做线段几何一致性筛选：

| 过滤条件 | 作用 |
|---|---|
| 方向差阈值 | 排除方向不一致的线段 |
| 长度比例阈值 | 排除长度差异过大的线段 |
| 中点最大位移 | 排除位移明显不合理的线段 |
| 位移一致性投票 | 找到支持数最多的整体平移 |
| 一对一去重 | 保证一条 source / target 线段只匹配一次 |

最终输出：

```text
translation = 所有一致线段匹配的平均中心位移
affine = [1 0 tx; 0 1 ty]
```

这样可以避免直接把线段端点当作稳定点对应。

### 9.5 线段几何 baseline

`LineSegmentAssociator` 不计算描述子，只根据线段几何属性匹配。

流程：

1. 为每条线段计算中心、方向、长度。
2. 枚举 source 和 target 线段对。
3. 根据方向差、长度比例、中点位移筛候选。
4. 对中点位移做一致性投票。
5. 输出平均平移仿射。

它可作为线描述子方法的 baseline。

## 10. 响应图和点集结构关联方法

### 10.1 Phase Correlate

思想：通过频域相位相关估计两幅响应图的平移。

方法：

```text
结构响应图 -> float 图 -> blur -> cv::phaseCorrelate -> shift + response
```

优点：速度快。

限制：主要适合平移模型。

### 10.2 Chamfer

思想：target 结构响应图计算距离变换，source 点集移动后落在 target 结构附近则距离小。

流程：

1. target 响应图转二值 mask。
2. 计算距离变换。
3. 采样 source 结构点。
4. 在搜索窗口内枚举平移。
5. 平均距离最小的位置作为匹配结果。

### 10.3 Hausdorff

思想：用点集之间的最大或分位距离衡量结构相似性。

当前实现使用分位数距离，降低离群点影响。

### 10.4 ICP

思想：迭代最近点。

流程：

1. 初始化平移。
2. source 点集加平移。
3. 为每个点找 target 最近邻。
4. 根据对应点平均差更新平移。
5. 重复直到收敛。

适合初始位姿较接近的结构点集。

## 11. 图像变换方法

当前主要使用 `PerspectiveWarper`。

输入：

| 几何结果 | 使用方式 |
|---|---|
| Homography `H` | 透视变换 |
| Affine `A` | 仿射变换 |
| Rigid / Similarity | 以 2x3 仿射矩阵形式执行 |

输出：

1. `warped/source_target_*_warped.png`
2. `blend/source_target_*_blend.png`

blend 图用于直观判断配准是否对齐。

## 12. 评测指标

当前项目已有评测模块框架。

### 12.1 几何指标

| 指标 | 说明 |
|---|---|
| Inlier Ratio | 内点数量 / 匹配数量 |
| Reprojection Error | 内点重投影误差 |

### 12.2 图像指标

| 指标 | 说明 |
|---|---|
| PSNR | 峰值信噪比 |
| RMSE | 均方根误差 |
| SSIM | 结构相似度 |

### 12.3 特征指标

| 指标 | 说明 |
|---|---|
| Repeatability | 关键点重复率 |

### 12.4 Warp Overlap IoU

平台还支持可选的 warp 前景重合校验：

```text
warped source 前景 mask 与 target 前景 mask 的 IoU
```

适合在批处理时自动发现明显失败的配准。

## 13. 方法组合示例

### 13.1 SIFT + BF + Ratio + CrossCheck + Homography

适合纹理明显、透视变化较大的图像。

```yaml
keypoint: ../../keypoint/sift.yaml
matcher: ../../matcher/bf.yaml
filters:
  - ../../filter/ratio_test.yaml
  - ../../filter/cross_check.yaml
geometry: ../../geometry/homography.yaml
```

### 13.2 ORB + BF + GMS + Affine

适合速度要求较高、特征数量较多的实验。

### 13.3 LINE + LBD + 中心位移一致性

适合线结构明显、纹理较少的图像。

```yaml
structure: ../../structure/line.yaml
geometry: ../../geometry/rigid.yaml
```

其中 `line.yaml` 配置：

```yaml
association:
  method: line_descriptor
  params:
    line_descriptor:
      descriptor: LBD
      ratio: 1.0
      knnK: 8
      angleThresholdDeg: 30.0
      minLengthRatio: 0.30
      maxShiftDistance: 100000.0
      shiftConsistencyThreshold: 30.0
      minMatches: 2
```

## 14. 后续方法扩展建议

### 14.1 线描述子扩展

| 方法 | 描述子类型 | 推荐匹配方式 |
|---|---|---|
| MSLD | 浮点 | BF-L2 或 FLANN |
| line-SIFT | 浮点 | BF-L2 或 FLANN |
| LLD | 取决于实现 | L2、cosine、Hamming 或 ANN |
| 深度线描述子 | 浮点向量常见 | cosine 或 L2，必要时用 FAISS / ANN |

建议 YAML 未来设计：

```yaml
line_descriptor:
  descriptor: MSLD
  matching:
    type: AUTO
    ratio: 0.75
    knnK: 2
```

`AUTO` 根据描述子类型选择匹配器，而不是让用户直接写底层实现。

### 14.2 结构法几何模型增强

当前线结构主要输出平移仿射，后续可扩展：

1. 基于匹配线段方向估计旋转。
2. 基于线段长度比例估计尺度。
3. 使用线段端点、中点和方向联合估计相似变换。
4. 使用多线段约束估计仿射或单应。

### 14.3 评测增强

建议补充：

1. 按方法族输出总排名。
2. 输出每个样本的成功/失败原因统计。
3. 对 batch 结果生成图表。
4. 记录参数配置快照，便于实验复现。

## 15. 方法选择建议

| 场景 | 推荐方法 |
|---|---|
| 纹理丰富、视角变化明显 | SIFT / SURF + Homography |
| 速度优先 | ORB / BRISK + BF-Hamming |
| 非线性尺度空间细节明显 | KAZE / AKAZE |
| 纹理少但线条明显 | LINE + LBD |
| 只有边缘或轮廓结构 | EDGE / CONTOUR + Chamfer / Hausdorff / ICP |
| 主要是平移 | PhaseCorrelate 或结构中心位移 |
| 有旋转和尺度 | Similarity 或 Affine |
| 透视变化 | Homography |

总体建议：先用点特征法建立 baseline，再用结构法处理低纹理或线结构明显的样本，最后用统一批处理 CSV 横向比较。
