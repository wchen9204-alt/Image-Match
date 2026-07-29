# 配准正确性判定逻辑说明

本文说明当前 pipeline 对“配准结果是否正确”的判定逻辑。整体分成两层：

1. 方法特有判定：只检查该方法族自己特有的数据质量信号。
2. 公共最终判定：只要能产出 warp，就用同一套图像级规则做最终 success 判定。

## 验证入口与执行顺序

统一入口在 `BasePipeline::validateRegistrationQuality`。当前执行顺序为：

1. `validateMethodSpecificQuality`
2. `validateSharedFinalQuality`

其中：

- `validateMethodSpecificQuality` 内部负责方法特有判定：
  - `match_quality`：离散对应关系质量验证。
  - `direct_quality`：直接法算法自身置信度验证。
  - `structure_overlap`：结构法响应图重叠验证。
- `validateSharedFinalQuality` 当前统一落在 `warp_quality`：
  - `warp_overlap`
  - `photometric`
  - `edge_alignment`

每个验证项只有在对应 YAML 配置块显式 `enabled: true` 时才会参与判定；同时代码还会先检查方法族，避免某个方法误进入自己用不到的逻辑。

## 公共最终判定

公共最终判定由 `validateSharedFinalQuality` 调用 `validateWarpQuality` 完成。它不区分点特征法、结构法、直接法或学习法，只看最终 warp 后的图像结果是否合理。

### warp_overlap

配置位置：`validation.warp_overlap`

作用：检查 warped source 与 target 的前景区域是否具有合理的几何覆盖关系。

主要字段：

- `enabled`：是否启用。
- `min_containment`：warped source 对 target 的局部包含率下限。
- `foreground_threshold`：生成前景 mask 时使用的灰度阈值。

输出指标：

- `warp_overlap_containment`
- `warp_source_coverage`
- `warp_target_coverage`

局限：它主要回答“区域有没有覆盖上”，不能单独保证区域内部的结构和纹理也真正对齐。

### photometric

配置位置：`validation.photometric`

作用：检查 warped source 与 target 在重叠前景区域内的光度误差。当前实现只在重叠区域统计误差，避免“一张图只是另一张图局部”时被非重叠区域污染。

主要字段：

- `enabled`：是否启用。
- `max_nmad`：常规 NMAD 上限，越小越严格。

输出指标：

- `warp_photometric_error`

局限：如果阈值设得过松，仍可能放过“覆盖了但没对上”的结果；因此通常需要和 `edge_alignment` 配合使用。

### edge_alignment

配置位置：`validation.edge_alignment`

作用：在重叠区域分别提取 warped source 和 target 的边缘，计算边缘 IoU，用来补充拦截“前景覆盖充分，但内容结构没有真正对齐”的结果。

主要字段：

- `enabled`：是否启用。
- `min_iou`：边缘 IoU 下限。
- `canny_low_threshold`：Canny 低阈值。
- `canny_high_threshold`：Canny 高阈值。
- `dilate_size`：边缘 mask 膨胀核尺寸。
- `min_edge_pixels`：最低边缘像素数。

输出指标：

- `warp_edge_alignment_iou`

局限：低纹理、强模糊或边缘很少的样本，对边缘 IoU 不友好，因此它更适合作为公共最终判定里的补充项，而不是唯一依据。

## 方法特有判定

方法特有判定由 `validateMethodSpecificQuality` 负责。它只进入当前方法族真正需要的验证分支，不会让某个方法跑进自己没有数据支撑的逻辑。

### match_quality

配置位置：`validation.match_quality`

适用方法：

- 点特征法
- 学习法
- 确实输出离散点对的直接法

作用：验证离散对应关系是否足够可信。

主要字段：

- `enabled`：是否启用。
- `fail_on_violation`：不满足阈值时是否直接判失败。
- `min_inliers`：最少内点数。
- `min_inlier_ratio`：最低内点率。
- `max_reproj_error`：最大重投影误差。
- `min_inlier_spatial_coverage`：最终内点在 source / target 前景包围盒中的最低空间覆盖率。

输出指标：

- `num_inliers`
- `inlier_ratio`
- `mean_reproj_error`
- `inlier_spatial_coverage`

注意：没有真实离散点对的直接法，不能伪造或复用点特征法里的 `num_inliers` 语义。

### direct_quality

配置位置：`validation.direct_quality`

适用方法：直接法。

作用：检查直接法算法自身输出的 `confidence`、`response` 或其它得分是否达到最低要求。

主要字段：

- `enabled`：是否启用。
- `fail_on_violation`：不满足阈值时是否直接判失败。
- `min_confidence`：最低 confidence 阈值。

输出指标：

- `direct_confidence`

局限：算法自身分数只能说明优化过程或响应是否稳定，不能单独证明几何内容已经对齐，因此仍需要配合公共最终判定。

### structure_overlap

配置位置：`validation.structure_overlap`

适用方法：结构法。

作用：把 source 结构响应图 warp 到 target 坐标系后，和 target 结构响应图计算 IoU，检查结构响应是否真正对齐。

主要字段：

- `enabled`：是否启用。
- `min_iou`：结构响应图 IoU 下限。
- `foreground_threshold`：结构响应图二值化阈值。
- `dilate_size`：结构 mask 膨胀核尺寸。

输出指标：

- `structure_overlap_iou`

## 当前直接法的整体逻辑

当前直接法的成功判定也是“两层”：

1. 先做方法特有判定：
   - 稀疏/稠密直接法如果输出离散点对，可以启用 `match_quality`。
   - 全局/频域直接法如果主要输出矩阵和算法分数，可以启用 `direct_quality`。
2. 再做公共最终判定：
   - `warp_overlap`
   - `photometric`
   - `edge_alignment`

这意味着：

- 点特征法和直接法在公共最终判定上口径一致。
- 不同方法族的差异，主要落在方法特有判定上。

## 直接法点特征初始化

配置位置：`feature_initializer`

作用：在直接法优化前，先尝试一次点特征法粗估，为直接法提供更好的初始值。

当前约束：

- 只选择一种点特征方法。
- 如果配置了多种旧格式候选，也只读取第一个有效候选。
- 初值候选要先通过自己的几何接受条件和临时 warp 质量检查，才会被送入直接法。

注意：

- 点特征初始化通过，首先表示“这个初值可以交给直接法继续优化”。
- 当 `final_validation_reference = DIRECT_ONLY` 时，它不参与最终 success 选择。
- 当 `final_validation_reference = BEST_OF_DIRECT_AND_INITIALIZER` 时，
  initializer 还会作为 direct 最终结果的候选之一参与最终选择。

### final_validation_reference

配置位置：`feature_initializer.final_validation_reference`

可选值：

- `DIRECT_ONLY`
- `BEST_OF_DIRECT_AND_INITIALIZER`

语义：

#### DIRECT_ONLY

只使用 direct 最终输出的数据作为最终判定依据，不和点特征初始化结果比较。

#### BEST_OF_DIRECT_AND_INITIALIZER

direct 最终结果和已接受的 initializer 分别按各自已有规则先得到成功/失败状态，然后按下面的规则选最终结果：

1. 如果两者都失败，则最终失败。
2. 如果 initializer 成功、direct 失败，则直接使用 initializer 结果。
3. 如果 initializer 失败、direct 成功，则直接使用 direct 结果。
4. 如果两者都成功，则比较双方质量，选择更优结果作为最终结果。

当前“都成功时”只比较 `containment` 与 `photometric` 的综合分：

```text
containmentScore = clamp((containment - min_containment) / (1 - min_containment), 0, 1)
photometricScore = clamp(1 - NMAD / max_nmad, 0, 1)
finalScore = 0.35 * containmentScore + 0.65 * photometricScore
```

分数更高者作为最终结果；平分时保留 direct 结果。`edge_alignment` 仍可作为各自结果的质量门槛，但不参与两者的最终排序。

## 关于“覆盖上了但根本没对上”

像 ECC Test04 这类样本，问题通常不是前景完全没覆盖上，而是内容结构没有真正对齐。

这类样本不能只看 `warp_overlap`，因为它主要看区域覆盖；更合理的口径应该是组合判断：

1. `warp_overlap`：几何覆盖是否成立。
2. `photometric`：重叠区域灰度内容是否接近。
3. `edge_alignment`：重叠区域结构边缘是否对齐。
4. 直接法额外再看 `direct_quality`，点特征法额外再看 `match_quality`。

因此，“公共最终判定一致，方法特有判定分开”是当前更通用、也更稳定的结构。
