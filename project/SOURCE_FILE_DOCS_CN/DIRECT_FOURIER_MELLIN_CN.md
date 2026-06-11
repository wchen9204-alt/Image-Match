# `fourier_mellin_aligner` 快速阅读笔记

相关文件：

- `project/include/direct/frequency/fourier_mellin_aligner.h`
- `project/src/direct/frequency/fourier_mellin_aligner.cpp`
- `project/configs/direct/fourier_mellin.yaml`
- `project/configs/pipeline/direct/frequency_direct_pipeline.yaml`

## 模块职责

`DirectFourierMellinAligner` 是频域直接法配准器，用 Fourier-Mellin 思路估计源图到目标图的相似变换：

1. 在频谱幅值图中消除平移影响。
2. 将频谱幅值转到 log-polar 空间，把旋转和尺度转换为平移。
3. 用相位相关估计 log-polar 平移，得到旋转/尺度候选。
4. 把候选变换回原图，用普通相位相关估计最终平移并评分。
5. 输出 `GeometryType::SIMILARITY` 和 2x3 仿射族矩阵 `A`。

## 主要函数

### `DirectFourierMellinAligner::DirectFourierMellinAligner`

读取 YAML 参数。关键参数包括：

- `windowed`：是否使用 Hann 窗。
- `use_pyramid` / `pyramid_levels` / `pyramid_scale`：是否在多尺度层生成旋转/尺度候选。
- `log_polar_cols` / `log_polar_rows`：log-polar 采样尺寸。
- `min_scale` / `max_scale`：尺度候选过滤范围。
- `rotation_scale_response_threshold`：旋转/尺度响应阈值。
- `translation_response_threshold`：最终平移响应阈值。

### `align`

核心入口，按顺序完成：

1. 清空 `DirectData` 和 `GeometryData`，写入方法名 `DIRECT_FOURIER_MELLIN`。
2. 将输入灰度图转为 `CV_32F`，并按配置做高斯预平滑。
3. 根据 `use_pyramid` 构建源图和目标图金字塔。
4. 每个金字塔层计算频谱幅值图，转换到 log-polar 空间。
5. 对 log-polar 图做相位相关，生成旋转/尺度候选。
6. 在原图尺度上评估每个候选：先 warp 源图，再相位相关估计平移。
7. 选择平移响应最高的候选，写回相似变换矩阵。
8. 通过统一 `dd.addDiagnostic` 写入旋转、尺度、响应和候选数量等诊断项。

## 关键辅助函数

### `computeMagnitudeSpectrum`

计算频谱幅值图：

1. 可选乘 Hann 窗，降低边界效应。
2. `cv::dft` 得到复数频谱。
3. `cv::magnitude` 取幅值，`log(1 + mag)` 压缩动态范围。
4. `fftShift` 将低频移到中心。
5. 可选抑制中心 DC 分量和频谱平滑。
6. 归一化到 `[0, 1]`，便于后续相位相关。

### `buildLogPolarSpectrum`

用 `cv::warpPolar(..., WARP_POLAR_LOG)` 将频谱幅值图映射到 log-polar 空间。该空间中：

- y 方向平移对应旋转。
- x 方向平移对应尺度变化。

### `expandShiftToCandidates`

根据 log-polar 相位相关的位移生成候选。实现中会尝试正负角度、正负尺度指数，以及可选的 180 度歧义候选，因为频谱幅值有中心对称性。

### `evaluateCandidate`

对单个旋转/尺度候选做最终评分：

1. 用 `cv::getRotationMatrix2D` 构造源图到目标图的候选相似矩阵。
2. `cv::warpAffine` 先应用旋转/尺度。
3. 对 warped 源图和目标图做相位相关，估计平移。
4. 把平移补到 2x3 矩阵中。
5. 以平移相位相关响应作为最终候选分数。

## 输出与诊断

成功时：

- `geometry_data.type = GeometryType::SIMILARITY`
- `geometry_data.A` 保存 2x3 相似变换矩阵
- `direct_data.A` 保存同一矩阵
- `direct_data.score` 保存最终平移响应

诊断项统一进入 `direct_data.diagnostics`：

- `rotation_deg`
- `scale`
- `rotation_scale_response`
- `translation_response`
- `candidate_count`
- `pyramid_levels_used`

这些指标会由 `registration_app.cpp` 的通用直接法诊断出口写入文本摘要和 JSON 的 `quality.direct_diagnostics`。

## 配置入口

单次运行频域直接法：

```powershell
build-mingw\bin\registration_app.exe configs\pipeline\direct\frequency_direct_pipeline.yaml
```

默认 `frequency_direct_pipeline.yaml` 指向：

```yaml
direct: ../../direct/fourier_mellin.yaml
```

如果要切回平移型相位相关，可改为：

```yaml
direct: ../../direct/phase_correlation.yaml
```
