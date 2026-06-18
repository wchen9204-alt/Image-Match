# 一、点特征法的优化策略

当前点特征法主要面向平移、旋转、静态、无明显光照变化的配准场景，并且允许一张图是另一张图的局部。根据前面的测试结果，当前失败主要集中在两个位置：

1. 进入几何阶段之前，`filtered_matches` 里仍然混有不少“局部看着像，但整体不满足 rigid”的错配。

2. 即使 `filtered_matches` 里已经包含了正确对应，单次 RANSAC / 单个最优模型也可能先落到局部最优，而不是全局更合理的 rigid 结果。

因此当前点特征主线增加了三类优化：

1. `pairwise_rigid_consistency`

2. `rigidRefineMode: SVD`

3. 基于 `filtered_matches` 的 rigid 候选生成

这三步的分工不同：

- `pairwise_rigid_consistency` 负责在进入几何估计前提高匹配纯度。
- `rigidRefineMode: SVD` 负责把 OpenCV 初始模型重新压回严格 rigid，减少统一缩放或局部漂移带来的偏差。
- 基于 `filtered_matches` 的 rigid 候选生成负责在几何估计阶段补充更多可能的 rigid 模型，避免单个 baseline 模型卡在局部最优。

## 1. pairwise_rigid_consistency

### （1）背景

1. 根据测试结果发现，`raw_matches` 的数量和 `filtered_matches` 的数量经常一样，说明已有过滤在某些配置下没有真正起作用。

2. 当前使用 `BFMatcher -> match` + 交叉验证 + 比率测试。  
   `BFMatcher::match` 的匹配效果明显优于 `BFMatcher::knnMatch`，但 `match` 输出的是一维 `vector<DMatch>`，每个查询特征只保留一个最优匹配。实验平台里 ratio test 对 `match` 的实现更多是结构适配，实际并不会像 KNN 那样产生有效的二邻近比值过滤。

3. 已实现的描述子距离分布过滤、距离阈值过滤等方法，在当前测试集上没有稳定起到正效果。

因此，需要一种不依赖 KNN、也不只看单个描述子距离的过滤方法，用来减少错误匹配。

当前实现文件：

- `project/src/filter/pairwise_rigid_consistency_filter.cpp`
- `project/include/filter/pairwise_rigid_consistency_filter.h`
- `project/configs/filter/pairwise_rigid_consistency.yaml`

### （2）原理

`pairwise_rigid_consistency` 是 `matcher` 之后、`rigid_estimator` 之前的一层点特征预过滤。

它不直接估计最终变换，而是判断当前 `filtered_matches` 内部是否满足刚体几何一致性。核心假设是：如果两张图之间主要是旋转 + 平移，那么正确匹配之间两两组成的向量应该满足：

1. source 中的 pair 长度和 target 中对应 pair 长度接近。

2. source pair 到 target pair 的相对旋转角应该集中在同一个主旋转方向附近。

因此，当前实现使用 `pairwise vote + 主旋转峰`：

- 把 matches 两两组合成 pair。
- 对每个 pair 计算 source / target 向量长度差。
- 对通过长度检查的 pair 计算相对旋转角。
- 把相对旋转角投票到旋转直方图。
- 找到主旋转峰。
- 统计每条 match 被多少个接近主旋转峰的 pair 支持。
- 只保留支持票数足够的 matches。

### （3）有效原因

1. 解决的不是“距离差”，而是“几何关系不一致”。

很多错配的问题不在于描述子距离特别大，而在于：

- 单看某一个局部 patch，两个点像是能对应。
- 但把它和其它匹配一起放进 rigid 几何里，就会发现它和别人支持的旋转方向、pair 长度关系互相冲突。

这类错配用单点描述子距离阈值并不好拦。`pairwise_rigid_consistency` 的价值，就是把“单点相似”提升到“点对几何相似”来判断。

2. 适合当前这类 rigid 场景。

当前场景假设本身就是旋转 + 平移、无缩放、静态。在这种前提下，正确匹配之间的相对旋转往往会围绕某个主方向聚集，而错配的旋转关系更容易分散。

所以：

- 正确 matches 更容易在旋转直方图里形成主峰。
- 错配更容易被投票稀释掉。
- 最后保留下来的匹配集合，往往比单纯按描述子距离排序更“像一个整体”。

3. 提升进入 RANSAC 的候选纯度。

`pairwise_rigid_consistency` 的真正价值，不只是减少匹配数量，而是提高进入 `rigid_estimator` 的候选纯度。候选纯度提高之后：

- `estimateAffinePartial2D` 的 RANSAC 更容易抽到真正一致的点。
- 初始模型更稳定。
- 后面的 rigid refine 更不容易失败。
- 基于 `filtered_matches` 的 rigid 候选生成也更容易生成有意义的候选。

### （4）核心流程

1. 从 `filtered_matches` 中取出有效匹配。这里一条 match 表示 source 图上的一个点与 target 图上的一个点之间的对应关系。

2. 将两条 match 两两组成一个 `match pair`。这里的 pair 指的是“同一张图里的两个点”：两条 match 会在 source 图上形成一个 source pair，也会在 target 图上形成一个 target pair。

3. 过滤掉 pair 距离过近的组合，避免短向量导致旋转估计不稳定。

4. 比较 source pair 与 target pair 的长度差，压掉明显不满足 rigid 假设的 pair。

5. 计算每个 pair 的相对旋转角。

6. 把这些旋转角投票到旋转直方图。

7. 取票数最高的主旋转峰。

8. 重新统计每条 match 被多少个“接近主旋转峰”的 pair 支持。

9. 仅保留支持票数达到阈值的 matches。

### （5）具体步骤

1. 读取配置参数。  
   从 `pairwise_rigid_consistency.yaml` 中读取 `min_pair_distance`、`max_distance_diff`、`max_angle_diff_deg`、`rotation_bins`、`min_votes`、`keep_top_k`、`fallback_to_input_if_empty`，并在构造函数里做最小合法化。

2. 判断当前输入是否适合做该过滤。  
   当前实现只对点特征匹配生效；如果是结构法分支，直接透传。  
   如果 `filtered_matches` 少于 3 条，也直接保留输入，因为这时 pairwise vote 没有统计意义。

3. 清洗无效 match。  
   遍历 `filtered_matches`，检查 `queryIdx/trainIdx` 是否越界。  
   只有索引合法的 match 才进入后续的 pairwise 分析，同时同步提取 source / target 点坐标。

4. 构造 match pair 并做第一轮几何筛选。  
   对有效 matches 两两配对。一个 `match pair` 由两条 match 构成：在 source 图上取这两条 match 的两个 `query` 点形成 source pair，在 target 图上取对应的两个 `train` 点形成 target pair。  
   每个 pair 先检查 source / target 两边的点间距是否都大于 `min_pair_distance`，避免短向量导致旋转不稳定；再检查两边 pair 长度差是否小于 `max_distance_diff`，压掉明显不满足 rigid 假设的组合。

5. 计算 pair 相对旋转并做直方图投票。  
   对通过筛选的 pair，分别计算 source 向量角和 target 向量角，取差值得到该 pair 支持的旋转角。  
   然后把旋转角归一化到 `(-180, 180]`，投到 `rotation_bins` 个 bin 的旋转直方图里。

6. 找出主旋转峰。  
   统计直方图中票数最高的 bin，把它当作当前这一批 matches 最有可能支持的主旋转方向。  
   这一阶段不求最终刚体，只估计“主方向”。

7. 统计每条 match 被主旋转峰支持了多少次。  
   重新遍历所有 pair，只保留那些旋转角与主旋转峰差值不超过 `max_angle_diff_deg` 的 pair。  
   每当一个 pair 被认为支持主峰，就给它关联到的两条 match 各加一票。

8. 按支持票数保留 matches。  
   票数大于等于 `min_votes` 的 match 才被保留。  
   如果配置了 `keep_top_k`，则在通过门槛的 matches 里继续按 `votes` 优先、描述子距离次之做截断，留下最稳定的一批。

9. 处理空结果回退。  
   如果最后一条都没保住，就根据 `fallback_to_input_if_empty` 决定是回退到原始输入，还是清空结果并返回失败。  
   这样可以避免过滤过严，直接把后续 `rigid_estimator` 饿死。

10. 写回新的 `filtered_matches`。  
    最终保留下来的 matches 会覆盖原来的 `ctx.keypoint_match_data.filtered_matches`，后面的 `rigid_estimator` 就只在这批更“刚体一致”的候选上继续求解。

## 2. rigidRefineMode: SVD

### （1）背景

当前 `rigid_estimator` 的 baseline 后端使用的是 `estimateAffinePartial2D`。这个后端本质上求的是 partial affine / similarity 模型，也就是：

- 允许旋转
- 允许平移
- 允许统一缩放
- 不允许剪切

但当前点特征主线面对的是严格 rigid 场景，也就是只允许旋转 + 平移，不希望最终模型里带统一缩放。

如果直接使用 OpenCV 给出的初始矩阵，就会出现两个问题：

1. 虽然 RANSAC 内点看起来成立，但最终矩阵可能仍然带有统一缩放，不是严格 rigid。

2. 在弱纹理、局部重叠或长方形前景场景里，少量内点容易把初始相似变换拉偏，导致最后的刚体结果不够稳定。

因此，当前主线把 `rigidRefineMode: SVD` 作为独立的一步保留下来，用来把初始模型重新压回严格 rigid。

当前相关配置与实现位置：

- `project/configs/geometry/rigid.yaml`
- `project/src/geometry/rigid_estimator.cpp`
- `project/include/geometry/partial_affine_utils.h`

### （2）原理

`rigidRefineMode: SVD` 的作用是：在 OpenCV RANSAC 已经筛出一批内点之后，不直接使用 OpenCV 返回的相似变换，而是重新用这些内点回归一个严格的旋转 + 平移模型。

当前 `rigidRefineMode` 现在保留两种模式：

- `NONE`：直接使用 OpenCV 返回结果
- `SVD`：用内点重新拟合严格 rigid

当前主线选择的是 `SVD`，因为它不是只在矩阵上“硬去尺度”，而是重新根据内点集求最优旋转和平移，通常比简单归一化尺度更稳定。

### （3）有效原因

1. 它把 similarity 结果压回 strict rigid。

`estimateAffinePartial2D` 给出的结果允许统一缩放，而当前场景不需要缩放。  
`SVD` refine 会把这一点纠正回来，保证最后模型只保留旋转 + 平移。

2. 它能利用已经筛出的内点重新拟合更稳定的刚体。

RANSAC 第一阶段的作用更偏向“找内点集”，而不是给出最终最优 rigid 参数。  
`SVD` refine 会在内点固定后重新回归，通常能让旋转和平移更平滑、更稳定。

3. 它本身就会提升 baseline 的质量。

即使不开 `pairwise_rigid_consistency`，也不开基于 `filtered_matches` 的 rigid 候选生成，只要 baseline 还能筛出一批基本可用的内点，`SVD` refine 也可能单独把结果拉回到更合理的 rigid。

### （4）核心流程

1. 先用 `estimateAffinePartial2D` 做 RANSAC，得到初始变换和内点 mask。

2. 从初始内点里提取 source / target 对应点。

3. 使用这些内点重新拟合严格 rigid。

4. 只保留旋转 + 平移，不允许统一缩放。

5. 根据 refine 后的新模型重新统计内点与误差。

### （5）具体步骤

1. `RigidEstimator::estimate()` 先调用 `estimateAffinePartial2D`。  
   这一步负责找到一个初始相似变换，并输出一批 RANSAC 内点。

2. 读取 `rigidRefineMode` 配置。  
   当前 `project/configs/geometry/rigid.yaml` 里主线配置为 `rigidRefineMode: SVD`。

3. 把 RANSAC 内点提取出来。  
   后续 refine 不再看全部匹配，而是只看这一批被 baseline 认可的内点。

4. 用内点重新估计严格 rigid。  
   当前实现会在内点上做 SVD 回归，重新求一组最优旋转和平移。

5. 用 refine 后的 rigid 模型重新计算误差和内点。  
   如果 refine 失败，baseline 结果就不能作为稳定 rigid 使用；如果 refine 成功，则这个结果会进入后面的候选比较或直接作为最终结果。

## 3. 基于 filtered matches 的 rigid 候选生成

### （1）背景

根据 `estimateAffinePartial2D` 的结果可以发现，RANSAC 会对 `filtered_matches` 再做一次几何筛洗。这个过程有时会让可用匹配数量大幅减少，导致很多样本虽然前面还有可用点，但最终几何估计失败。

另外，在长方形、弱纹理、局部重叠、近灰区域等场景里，单次 RANSAC / 单个最优模型可能先抓住一个局部区域形成局部最优，而不是找到全局更合理的 rigid 变换。

所以当单纯依靠 `estimateAffinePartial2D` 无法稳定得到正确结果时，需要从当前 `filtered_matches` 中额外生成一批 rigid 候选，再用统一评分选择更合理的结果。

当前相关文件：

- `project/src/geometry/rigid_estimator.cpp`
- `project/src/geometry/rigid_estimator_helpers.cpp`
- `project/include/geometry/rigid_estimator_helpers.h`
- `project/configs/geometry/rigid.yaml`

### （2）原理

基于 `filtered_matches` 的 rigid 候选生成位于 `rigid_estimator` 内部，不是过滤器，而是几何阶段的候选补救机制。

它保留 baseline RANSAC rigid 结果，但不再只相信这一个结果，而是从当前 `filtered_matches` 中额外生成一批严格 rigid 候选，再统一评分选最优。

### （3）有效原因

1. 它解决的是局部最优，不是错配本身。

基于 `filtered_matches` 的 rigid 候选生成主要不是为了解决“坏匹配太多”，而是解决：

- 正确匹配其实已经在输入里。
- 但单次 RANSAC 或单个最优模型，先抓住了一个能解释局部区域的小模型。
- 这个小模型不是全局最合理的 rigid。

2. 它对长方形、弱纹理、局部重叠尤其重要。

在下面几类场景里，多个局部模型都可能“解释一小撮点”：

- 长方形前景。
- 边界重复。
- 大面积弱纹理。
- 一张图只是另一张图的局部。
- `180` 度旋转歧义。

这种时候，baseline 模型未必错得离谱，它只是先解释了某块局部。基于 `filtered_matches` 的 rigid 候选生成可以把这些局部可行模型显式拿出来，再统一比较，不让流程过早锁死在第一眼找到的那个模型上。

3. 它把“内点数最多”升级为综合评分。

候选选择不再只看内点数和重投影误差，还会结合：

- `containment`
- `sourceCoverage`
- `targetCoverage`
- `bidirectionalCoverage`

这样候选模型不只需要解释点，还需要在前景 mask 几何上更像真实对齐。

### （4）核心流程

1. 先从 `filtered_matches` 中构建距离池。

2. 在距离池中做空间分散选点。

3. 再补一批描述子距离较好的点。

4. 再根据当前距离池内估计出的主旋转峰，补一批旋转更一致的点。

5. 从这些 seed 中两两取点，生成最小 rigid 候选。

6. 每个候选再投影回全部点，得到候选内点 mask。

7. 候选进入统一评分流程。

### （5）具体步骤

1. 先保留 baseline rigid 结果。  
   `RigidEstimator::estimate()` 先按当前配置后端跑 baseline。  
   `OPENCV_PARTIAL_AFFINE` 会先用 `estimateAffinePartial2D` 得到初始模型，再按 `rigidRefineMode` 选择 `NONE / SVD`，把模型尽量压回严格 rigid。  
   如果 baseline refine 成功，就先把它作为第一个候选放入候选池。

2. 判断是否启用 filtered-match candidates。  
   只有在以下条件都满足时，才会继续生成额外候选：  
   `enableFilteredMatchCandidates = true`、当前不是 `CUSTOM_RIGID_RANSAC`、有足够的 `filtered_matches`、`filteredDistances` 非空、`filteredMatchCandidateCount > 0`。

3. 构造候选 seed 索引集合。  
   调用 `buildMixedCandidateSeedIndices()`，用混合策略从 `filtered_matches` 中挑一批 seed：  
   先按描述子距离排序，截出一个距离池；  
   再在距离池里优先选空间上更分散的点；  
   再补一批距离最好的点；  
   最后按池内估计出的主旋转峰，补一批旋转更一致的点。  
   这一阶段的目的不是只追求“距离最小”，而是让候选种子同时兼顾质量、分散性和旋转一致性。

4. 从 seed 两两生成最小 rigid 假设。  
   对 seed 索引两两配对。  
   如果一对点在 source 或 target 上距离过近，小于 `filteredMatchCandidateMinPairDistance`，就直接跳过，避免退化。  
   通过的点对会调用 `buildRigidCandidateFromPair()`，先用 2 对点构建一个最小 rigid 假设。

5. 用全量 filtered 点回投候选并生成初始内点 mask。  
   最小 rigid 候选生成后，会把它投影回全部 `filtered` 点，用重投影误差阈值生成一个候选内点 mask。  
   这一步把“2 点假设”扩展成“对整批点的解释能力”。

6. 对每个候选再做 rigid refine。  
   候选不会直接入池，而是再次按 `rigidRefineMode` 做 refine：  
   `NONE` 直接使用当前候选；  
   `SVD` 按候选内点迭代回归严格 rigid。  
   只有 refine 成功且候选矩阵非空，才真正加入候选池。

7. 把 baseline 和额外候选统一放入同一评分池。  
   当前实现不是“baseline 一套规则，补充候选另一套规则”，而是把 baseline 和新增候选统一交给 `selectBestRigidCandidate()` 做同一套比较，避免评分口径不一致。

8. 做候选去重。  
   在 `selectBestRigidCandidate()` 里，先按旋转角差 `candidateDedupRotationDiffDeg` 和平移差 `candidateDedupTranslationDiff` 做一轮轻量去重。  
   目的是避免很多几乎一样的候选重复参与评分，刷存在感。

9. 统一计算每个候选的评分项。  
   对去重后的每个候选，统一补齐：  
   内点数、重投影误差、`containment`、`sourceCoverage`、`targetCoverage`、`bidirectionalCoverage`。  
   其中前景 mask 相关评分由 `evaluateRigidCandidateMaskScore()` 负责补算。

10. 先看前景 mask 门槛，再看传统几何统计。  
    如果启用了 `enableCandidateMaskScoring`，候选会优先看是否通过：  
    `candidateMinContainment`、`candidateMinBidirectionalCoverage`。  
    通过前景门槛的候选优先级更高。  
    如果所有候选都没通过前景门槛，就自动回退到旧规则，也就是主要按内点数和误差继续比较，避免新规则过严直接把结果清空。

11. 选出最终最优候选。  
    最终比较顺序由 `preferRigidCandidateScore()` 控制，主要优先级是：  
    是否通过前景 mask 门槛、内点数、双向 coverage、containment、重投影误差。  
    排序胜出的候选会成为最终 rigid 结果。

12. 回写最终结果到上下文。  
    选出的最优候选会写回 `geometry_data`，并通过 `promoteInliers()` 回写最终内点列表。  
    后面的 warp、coverage、photometric 验证，都是在这个最终胜出的 rigid 模型基础上继续进行。
