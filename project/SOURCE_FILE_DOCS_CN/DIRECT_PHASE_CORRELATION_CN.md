# `phase_correlation_aligner` 快速阅读笔记

本文档对应当前工程中的以下文件：

- `project/include/direct/frequency/phase_correlation_aligner.h`
- `project/src/direct/frequency/phase_correlation_aligner.cpp`
- `project/configs/direct/phase_correlation.yaml`

它的目标是帮助快速理解 `DirectPhaseCorrelationAligner` 的职责、关键函数分工、WPC 增强和亚像素峰值置信度检查。

---

## 1. 文件职责

`DirectPhaseCorrelationAligner` 是频域直接法配准器，当前只估计全局平移：

1. 从 `RegistrationContext::images.first_gray / second_gray` 读取两张灰度图
2. 预处理成 `CV_32F` 灰度图
3. 可选使用 WPC（Weighted Phase Correlation）给输入图乘空间权重
4. 可选使用 Hann 窗抑制边界突变
5. 调用 `cv::phaseCorrelate(...)` 得到亚像素平移和响应分数
6. 可选重建相位相关响应面，检查主峰/次峰比和亚像素峰值置信度
7. 将结果写入：
   - `ctx.geometry_data`
   - `ctx.direct_data`

当前输出的几何模型是 `GeometryType::AFFINE`，但矩阵内容固定为单位仿射加平移：

```text
[ 1  0  dx ]
[ 0  1  dy ]
```

---

## 2. 配置入口

配置文件是：

```text
project/configs/direct/phase_correlation.yaml
```

核心参数分三组：

### 基础 phase correlation

- `response_threshold`
  OpenCV `phaseCorrelate` 返回响应分数的最低阈值。低于该值直接拒绝。
- `blur_kernel`
  输入灰度图预平滑核大小。小于 3 表示不平滑，偶数会自动加 1。
- `use_hann_window`
  是否创建 Hann 窗，降低图像边界突变对频域峰值定位的影响。

### WPC 加权相位相关

- `weighted`
  是否启用 WPC。默认 `false`，用于保持旧版行为。
- `weight_mode`
  权重作用范围，可选：
  - `GRADIENT`：源图和目标图都使用梯度权重
  - `SOURCE_GRADIENT`：只给源图加权
  - `TARGET_GRADIENT`：只给目标图加权
  - `NONE`：不加权
- `weight_blur_kernel`
  权重图平滑核大小。
- `weight_power`
  权重幂指数，越大越强调高纹理区域。
- `weight_floor`
  权重下限，避免低纹理区域被完全压成 0。

### 峰值置信度

- `confidence_check`
  是否启用响应峰诊断。
- `peak_ratio_threshold`
  主峰与次峰比阈值。大于 0 时，低于该阈值会拒绝结果。
- `subpixel_confidence_threshold`
  亚像素置信度阈值。大于 0 时，局部峰值不够尖锐会拒绝结果。
- `peak_exclusion_radius`
  搜索次峰时，在主峰附近排除的半径。

只要 `peak_ratio_threshold` 或 `subpixel_confidence_threshold` 大于 0，构造函数会自动打开 `_confidenceCheck`。

---

## 3. 关键函数分工

### `normalizedKey(...)`

把配置字符串规整为大写字母/数字组成的 key。

用途：
- 兼容 `source_gradient`、`SOURCE_GRADIENT`、`SourceGradient` 等写法
- 让 `weight_mode` 的判断集中在一个稳定格式上

### `normalizedOddKernel(...)`

统一处理核大小：

- 小于 3：返回 0，表示关闭平滑
- 偶数：自动加 1，满足 OpenCV 高斯核对奇数尺寸的要求

### `getDoubleParam / getIntParam / getBoolParam / getStringParam`

这些函数用于读取配置参数，并兼容旧字段名。

例如：

```cpp
_weighted = getBoolParam(params, "weighted", "use_weighted_phase_correlation", false);
```

意思是优先读取 `weighted`，如果没有，再读取旧字段 `use_weighted_phase_correlation`。

### `preparePhaseGray(...)`

输入检查和基础预处理：

1. 要求图像非空且为单通道灰度图
2. 转为 `CV_32F`，范围缩放到 `[0, 1]`
3. 如果 `blur_kernel >= 3`，执行高斯预平滑

这一步的输出是后续 phase correlation 的基础输入。

### `gradientWeight(...)`

生成 WPC 使用的空间权重图：

1. 用 Sobel 计算 x/y 梯度
2. 计算梯度幅值
3. 可选对权重图高斯平滑
4. 用最大值归一化到 `[0, 1]`
5. 可选执行幂指数增强
6. 加上 `weight_floor` 保底

直觉上，它让纹理和边缘区域在相位相关中权重更高。

### `applyWeightedInputs(...)`

根据 `weight_mode` 决定给哪张图乘权重：

- `GRADIENT / BOTH`：源图和目标图都加权
- `SOURCE / SOURCE_GRADIENT`：只加权源图
- `TARGET / TARGET_GRADIENT`：只加权目标图
- `NONE`：保持原图

输出是 `phaseSrc / phaseDst`，后续 `cv::phaseCorrelate` 使用它们。

### `computePhaseResponseSurface(...)`

重新构造相位相关响应面，用于诊断峰值质量。

主要步骤：

1. 可选乘 Hann 窗
2. 分别对源图和目标图做 DFT
3. 计算互功率谱
4. 用幅值归一化，保留相位信息
5. 做逆 DFT 得到响应面

OpenCV 的 `phaseCorrelate` 直接返回平移和 response，但不暴露完整响应面；这里为了做主峰/次峰和局部尖锐度检查，需要自己重建一次。

### `responseAtWrapped(...)`

用周期边界读取响应图像素。

原因是 DFT 响应面天然是周期的，峰值可能出现在边界附近，局部邻域和次峰搜索都应该按环绕方式处理。

### `wrappedDistance(...)`

计算周期边界下的一维最短距离。

用途是在主峰附近排除一块区域，避免“次峰”其实只是主峰的邻近像素。

### `analyzePeakDiagnostics(...)`

分析响应面峰值结构，输出：

- `peakValue`
  主峰值
- `secondPeakValue`
  排除主峰邻域后的最大峰值
- `peakRatio`
  主峰 / 次峰，比值越大，峰值歧义越小
- `peakSharpness`
  主峰 / 8 邻域平均绝对值，比值越大，局部峰越尖锐
- `subpixelConfidence`
  如果 3 点二次曲面拟合满足局部最大值条件，则等于 `peakSharpness`；否则为 0

这里的亚像素置信度不是重新估计位移，而是检查 `cv::phaseCorrelate` 给出的亚像素峰附近是否像一个可靠峰。

---

## 4. `align(...)` 主流程

`align(RegistrationContext& ctx)` 是真正执行配准的入口。

流程如下：

1. 清空 `ctx.direct_data` 和 `ctx.geometry_data`
2. 读取并预处理两张灰度图
3. 检查两图尺寸必须一致
4. 如果启用 WPC，生成加权输入 `phaseSrc / phaseDst`
5. 如果启用 Hann 窗，创建窗口矩阵
6. 调用 `cv::phaseCorrelate(phaseSrc, phaseDst, window, &score)`
7. 检查平移和分数是否有限
8. 检查 `score >= response_threshold`
9. 如果启用峰值置信度检查：
   - 重建响应面
   - 分析主峰/次峰比与亚像素置信度
   - 根据阈值拒绝不可靠结果
10. 写入几何结果：
    - `gd.type = GeometryType::AFFINE`
    - `gd.A = [[1,0,dx],[0,1,dy]]`
    - `gd.inlier_ratio = score`
11. 写入直接法结果：
    - `dd.A`
    - `dd.score`
    - `dd.diagnostics`

---

## 5. 输出诊断项如何扩展

当前相位相关写入的诊断项是：

```cpp
dd.addDiagnostic("peak_ratio", "peak ratio", diag.peakRatio);
dd.addDiagnostic("peak_sharpness", "peak sharpness", diag.peakSharpness);
dd.addDiagnostic("subpixel_confidence", "subpixel conf", diag.subpixelConfidence);
```

这些诊断项会被 `registration_app.cpp` 统一输出：

- 文本摘要：遍历 `ctx.direct_data.diagnostics`
- JSON：写入 `quality.direct_diagnostics`

这个设计是为了后续 Fourier-Mellin / FMT 复用同一个出口。后续 FMT 可以追加类似：

```cpp
dd.addDiagnostic("rotation_confidence", "rotation conf", rotationConfidence);
dd.addDiagnostic("scale_confidence", "scale conf", scaleConfidence);
dd.addDiagnostic("log_polar_peak_ratio", "log-polar peak ratio", peakRatio);
```

这样应用层不需要为每个直接法方法继续添加 `if/else` 特判。

---

## 6. 当前方法的边界

- 只估计平移，不估计旋转、尺度、仿射或单应
- 要求两张灰度图尺寸一致
- WPC 是空间加权，不是频域滤波器
- 响应面诊断用于拒绝不可靠结果，不改变 `cv::phaseCorrelate` 返回的平移
- 如果 `confidence_check=false` 且阈值都为 0，则不会写入峰值诊断项

---

## 7. 最小运行方式

如果要单独跑相位相关，需要把直接法 pipeline 中的 `direct` 指向：

```yaml
direct: ../../direct/phase_correlation.yaml
```

然后运行：

```powershell
.\build-mingw\bin\registration_app.exe configs\pipeline\direct\global_direct_pipeline.yaml
```

如果只想观察 WPC 和峰值诊断，可以在 `phase_correlation.yaml` 中临时打开：

```yaml
weighted: true
confidence_check: true
peak_ratio_threshold: 0.0
subpixel_confidence_threshold: 0.0
```

阈值保持 0 时只输出诊断，不拒绝结果。
