# ZNCC 刚体全局直接法代码速览

本文面向“快速读懂代码”而写，聚焦这次新增的 `ZNCC_RIGID` 直接法实现，以及它在现有直接法框架里的落点。

当前状态说明：

- `ZNCC_RIGID` 已经完成框架接线、配置入口、单样本/批处理入口和输出目录接入
- 但优化器目前仍属于实验版，现阶段还不能像 `GLOBAL_LK / ESM_RIGID` 那样稳定通过现有数据集验证
- 也就是说，这份文档重点是帮助你快速理解“代码是怎么组织的”，不是宣称该方法已经达到生产可用状态

## 1. 这次新增了什么

新增文件：

- `project/include/direct/global/zncc_rigid_aligner.h`
- `project/src/direct/global/zncc_rigid_aligner.cpp`
- `project/configs/direct/zncc_rigid.yaml`
- `project/configs/pipeline/direct/global_direct_pipeline.yaml`

接线改动：

- `project/src/core/factory.cpp`
  增加 `ZNCC_RIGID / ZNCC / GLOBAL_ZNCC` 到 `ZnccRigidAligner` 的映射。
- `project/include/core/factory.h`
  更新直接法创建器的注释说明。
- `project/configs/pipeline/batch/batch_direct.yaml`
  改为引用 `global_direct_pipeline.yaml`，并在其中通过 `direct` 字段切换具体全局直接法算法。

## 2. 代码结构怎么读

### 2.1 入口类：`ZnccRigidAligner`

位置：

- `project/include/direct/global/zncc_rigid_aligner.h`
- `project/src/direct/global/zncc_rigid_aligner.cpp`

职责：

1. 从 YAML 读取 `max_iterations / epsilon / pyramid_levels / blur_kernel / gradient_threshold / sample_step`
2. 把这些参数装进 `RigidDirectOptions`
3. 调用公共流程 `rigid_direct_common::runRigidAlignment(...)`
4. 在公共流程成功后，额外回算一次最终 ZNCC 分数，并写回：
   - `ctx.direct_data.score`
   - `ctx.geometry_data.inlier_ratio`

### 2.2 公共框架：`runRigidAlignment`

位置：

- `project/include/direct/global/rigid_direct_common.h`

这是 `GLOBAL_LK / ESM_RIGID / ZNCC_RIGID` 共用的刚体直接法骨架。它负责：

1. 灰度图转 `float`
2. 可选高斯平滑
3. 同步下采样金字塔
4. 从零旋转、零平移开始做从粗到细逐层优化
5. 把最终刚体矩阵写进：
   - `ctx.geometry_data`
   - `ctx.direct_data`

真正的差异只在“每层优化器” `LevelOptimizer`。

### 2.3 每层优化器：`optimizeRigidLevelZncc`

位置：

- `project/src/direct/global/zncc_rigid_aligner.cpp`

这是本次实现的核心。当前版本的思路是：

1. 在当前参数下计算全局 ZNCC 等价目标
2. 对 `theta / tx / ty` 做数值差分，近似目标梯度和对角二阶项
3. 用带回溯的保守步长尝试下降

与 `GLOBAL_LK` 的主要区别：

- `GLOBAL_LK` 直接最小化原始灰度 SSD
- `ZNCC_RIGID` 最小化归一化后的残差平方和

代码里还加了一层简易回溯步长：

- 优先试 `1.0 * delta`
- 不下降就试 `0.5 / 0.25 / 0.1 / 0.05`
- 找到能让目标下降的步长再接受更新

## 3. 数学上它到底优化了什么

在每个有效采样点上，代码先定义：

- `s_i`：source 样本灰度
- `t_i(p)`：当前刚体参数 `p` 下，warp 后从 target 采样到的灰度

然后分别做标准化：

- `s'_i = (s_i - mean(s)) / std(s)`
- `t'_i = (t_i - mean(t)) / std(t)`

优化残差定义为：

- `r_i = t'_i - s'_i`

最终每层最小化：

- `sum(r_i^2)`

这个目标和最大化 ZNCC 是等价方向的；代码里最终还用：

- `zncc = 1 - 0.5 * objective`

把归一化残差目标重新映射回 `[-1, 1]` 附近的相关性分数。

## 4. 关键辅助函数怎么分工

都在 `project/src/direct/global/zncc_rigid_aligner.cpp`：

- `evaluateZnccObjective(...)`
  只负责在当前参数下计算 ZNCC 目标值，适合做步长回溯和最终打分。

- `optimizeRigidLevelZncc(...)`
  把目标评估和数值差分串起来，形成“评估 -> 数值梯度/曲率 -> 回溯步长 -> 更新参数”的迭代器。

## 5. 怎么运行

### 单张样本

```powershell
.\build-mingw\bin\registration_app.exe configs\pipeline\direct\global_direct_pipeline.yaml
```

运行前把 `project/configs/pipeline/direct/global_direct_pipeline.yaml` 里的 `direct` 改成：

```yaml
direct: ../../direct/zncc_rigid.yaml
```

### 批处理

```powershell
.\build-mingw\bin\registration_app.exe configs\pipeline\batch\batch_direct.yaml
```

运行前把 `project/configs/pipeline/batch/batch_direct.yaml` 里的 `pipeline` 改成：

```yaml
pipeline: ../direct/global_direct_pipeline.yaml
```

并把 `project/configs/pipeline/direct/global_direct_pipeline.yaml` 里的 `direct` 改成：

```yaml
direct: ../../direct/zncc_rigid.yaml
```

输出目录：

- 单样本：`outputs/single/direct/global_direct_pipeline/...`
- 批处理：`outputs/batch/direct/global_direct_pipeline/...`

## 6. 读代码时最值得先看哪几段

建议阅读顺序：

1. `project/include/direct/global/zncc_rigid_aligner.h`
2. `project/src/direct/global/zncc_rigid_aligner.cpp` 里的 `align(...)`
3. 同文件里的 `optimizeRigidLevelZncc(...)`
4. `project/include/direct/global/rigid_direct_common.h`

## 7. 当前实现的边界

这版实现是“先可用、再便于调试”的版本，几个已知特点：

- 只支持刚体三参数：`theta / tx / ty`
- 两张图仍要求同尺寸
- 仍沿用公共刚体框架的最终 raw photometric MSE 输出
- `ctx.direct_data.score` 和 `ctx.geometry_data.inlier_ratio` 被改成最终 ZNCC 相关分数
- 当前版本已经能编译、能被工厂创建、能进入 DirectPipeline、能生成输出目录和 summary
- 但在现有数据集上，优化器通常会停在 `no descent step found`
- 下一步若要把它做成真正可用的方法，更合适的方向是：
  1. 改成 patch-based ZNCC 而不是全局逐点强度归一化
  2. 或直接改成更系统的 LM / trust-region / coarse search 初始化
