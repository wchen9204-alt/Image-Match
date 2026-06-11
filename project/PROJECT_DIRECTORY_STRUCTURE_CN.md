# 项目目录结构

## 一致性结论

旧版 `PROJECT_DIRECTORY_STRUCTURE_CN.md` 已经和当前项目不完全一致：顶层顺序没有按项目目录展示，`configs/` 位置偏后；同时缺少 `build/`、`SOURCE_FILE_DOCS_CN/`、`CODEX_WORKING_RULES_CN.md`、直接法/学习法相关新增目录、`zncc_rigid`、`contour_descriptor`、`string_utils`、`direct_data`、`Test11`、`Test12` 等当前文件或目录。

下面按当前 `project/` 目录的实际顺序整理。构建目录、输出目录和第三方库只列关键层级，避免把大量生成物和外部依赖文件展开到文档里。

```
project/                                     # 图像配准实验平台根目录
├── apps/                                    # 应用程序入口
│   ├── registration_app.cpp                 # CLI 解析、单次/批量运行、结果输出
│   └── registration_app.h                   # RegistrationApp 类声明
│
├── build/                                   # CMake/IDE 构建目录（生成物，非源码）
│   └── CMakeFiles/                          # CMake 中间文件
│
├── build-mingw/                             # MinGW 构建与运行目录（生成物）
│   ├── bin/                                 # 可执行文件与运行时配置副本
│   ├── CMakeFiles/                          # CMake 中间文件
│   └── ...                                  # 其他构建缓存和目标文件
│
├── configs/                                 # YAML 配置文件
│   ├── direct/                              # 直接法配置
│   │   ├── dis_flow.yaml
│   │   ├── ecc.yaml
│   │   ├── esm_rigid.yaml
│   │   ├── farneback.yaml
│   │   ├── fourier_mellin.yaml
│   │   ├── global_lk.yaml
│   │   ├── klt_sparse.yaml
│   │   ├── phase_correlation.yaml
│   │   ├── tvl1_flow.yaml
│   │   └── zncc_rigid.yaml
│   ├── evaluator/                           # 评测配置
│   │   ├── benchmark.yaml
│   │   └── metrics.yaml
│   ├── filter/                              # 匹配过滤器配置
│   │   ├── cross_check.yaml
│   │   ├── distance_distribution.yaml
│   │   ├── distance_threshold.yaml
│   │   ├── gms.yaml
│   │   ├── min_distance.yaml
│   │   └── ratio_test.yaml
│   ├── geometry/                            # 几何估计器配置
│   │   ├── affine.yaml
│   │   ├── homography.yaml
│   │   ├── rigid.yaml
│   │   └── similarity.yaml
│   ├── keypoint/                            # 点特征提取器配置
│   │   ├── akaze.yaml
│   │   ├── brisk.yaml
│   │   ├── kaze.yaml
│   │   ├── orb.yaml
│   │   ├── sift.yaml
│   │   └── surf.yaml
│   ├── learning/                            # 深度学习匹配器配置
│   │   ├── loftr.yaml
│   │   ├── superpoint_lightglue.yaml
│   │   └── superpoint_superglue.yaml
│   ├── matcher/                             # 点特征匹配器配置
│   │   ├── bf.yaml
│   │   └── flann.yaml
│   ├── pipeline/                            # 流水线编排配置
│   │   ├── batch/                           # 批量运行配置
│   │   │   ├── batch_direct.yaml
│   │   │   ├── batch_keypoint.yaml
│   │   │   ├── batch_learning.yaml
│   │   │   ├── batch_structure.yaml
│   │   │   ├── compare_direct.yaml
│   │   │   └── compare_line.yaml
│   │   ├── direct/                          # 直接法流水线
│   │   │   ├── dense_direct_pipeline.yaml
│   │   │   ├── frequency_direct_pipeline.yaml
│   │   │   ├── global_direct_pipeline.yaml
│   │   │   └── sparse_direct_pipeline.yaml
│   │   ├── keypoint/                        # 点特征流水线
│   │   │   ├── akaze_pipeline.yaml
│   │   │   ├── brisk_pipeline.yaml
│   │   │   ├── kaze_pipeline.yaml
│   │   │   ├── orb_pipeline.yaml
│   │   │   ├── sift_pipeline.yaml
│   │   │   └── surf_pipeline.yaml
│   │   ├── learning/                        # 深度学习流水线
│   │   │   ├── loftr_learning_pipeline.yaml
│   │   │   ├── superpoint_lightglue_learning_pipeline.yaml
│   │   │   └── superpoint_superglue_learning_pipeline.yaml
│   │   └── structure/                       # 结构特征流水线
│   │       ├── contour_pipeline.yaml
│   │       ├── edge_pipeline.yaml
│   │       └── line_pipeline.yaml
│   └── structure/                           # 结构特征提取与关联配置
│       ├── contour.yaml
│       ├── edge.yaml
│       └── line.yaml
│
├── datasets/                                # 测试数据集
│   ├── Test01/                              # source.png + target.png
│   ├── Test02/                              # moving.png + reference.png
│   ├── Test03/                              # source.png + target.png
│   ├── Test04/                              # source.png + target.png
│   ├── Test05/                              # source.png + target.png
│   ├── Test06/                              # source.png + target.png
│   ├── Test07/                              # source.png + target.png
│   ├── Test08/                              # source.png + target.png
│   ├── Test09/                              # source.png + target.png
│   ├── Test10/                              # source.png + target.png
│   ├── Test11/                              # source.png + target.png
│   └── Test12/                              # source.png + target.png
│
├── include/                                 # 公共头文件
│   ├── core/                                # 核心基础设施
│   │   ├── config.h                         # 配置加载与路径解析：定义 MethodFamily、PipelineConfig 和 Config::load/loadPipeline/resolvePath
│   │   ├── context.h                        # RegistrationContext 上下文：贯穿单次配准流程，集中保存图像、特征、匹配、几何、评测和输出状态
│   │   ├── factory.h                        # Factory 静态工厂：根据 YAML 枚举和参数创建提取器、匹配器、过滤器、估计器、直接法和流水线对象
│   │   ├── registration.h                   # Registration 顶层门面：实现 IRegistration，负责配置流水线并对外提供统一 run 入口
│   │   ├── result.h                         # RegistrationResult 结果摘要：记录成功状态、失败原因、匹配/内点数量、耗时、IoU、NMAD 等运行指标
│   │   └── types.h                          # 全局枚举与转换工具：KeypointType、StructureType、NormType、MatchMethod、GeometryType、TransformType、ImageIndex
│   ├── data/                                # 流水线数据结构
│   │   ├── correspondence_view.h            # CorrespondenceView 统一对应点视图：把点特征、结构、直接法、学习法结果转换成通用点对/匹配/内点只读接口
│   │   ├── direct_data.h                    # DirectData 直接法输出：保存仿射/单应矩阵、光流、采样点对、伪匹配、内点掩码、得分和诊断指标
│   │   ├── evaluation_data.h                # EvaluationData 评测结果集合：用 MetricResult 记录指标名称、数值、启用状态和说明文本
│   │   ├── geometry_data.h                  # GeometryData 几何估计结果：保存模型类型、H/A 变换矩阵、内点掩码、内点数、重投影误差和有效性
│   │   ├── image_data.h                     # ImagePairData 图像对缓存：保存 source/target 的 BGR 图、灰度图和输入路径，供各流水线共享
│   │   ├── keypoint_data.h                  # KeypointData 点特征数据：分别保存两幅图的关键点、描述子、提取器名称和特征数量
│   │   ├── keypoint_match_data.h            # KeypointMatchData 点匹配数据：保存原始匹配、过滤后匹配、内点匹配以及与几何估计对齐的索引关系
│   │   ├── structure_data.h                 # StructureData 结构特征数据：保存边缘/线段/轮廓响应图、结构基元、描述子和提取方法信息
│   │   ├── structure_match_data.h           # StructureMatchData 结构关联结果：保存结构点对、线段匹配、估计平移、仿射初值、得分和关联有效性
│   │   └── transform_data.h                 # TransformData 最终变换数据：保存可用于 warping 的矩阵、变换类型、图像尺寸、有效性和说明信息
│   ├── dataset/                             # 数据集加载
│   │   ├── dataset_loader.h
│   │   └── sample.h
│   ├── direct/                              # 直接法配准器
│   │   ├── dense/
│   │   │   ├── dense_flow_common.h
│   │   │   ├── dis_flow_aligner.h
│   │   │   ├── farneback_flow_aligner.h
│   │   │   └── tvl1_flow_aligner.h
│   │   ├── frequency/
│   │   │   ├── fourier_mellin_aligner.h
│   │   │   └── phase_correlation_aligner.h
│   │   ├── global/
│   │   │   ├── ecc_aligner.h
│   │   │   ├── esm_rigid_aligner.h
│   │   │   ├── global_lk_aligner.h
│   │   │   ├── rigid_direct_common.h
│   │   │   └── zncc_rigid_aligner.h
│   │   └── sparse/
│   │       └── klt_sparse_aligner.h
│   ├── evaluator/                           # 评测系统
│   │   ├── evaluator.h
│   │   └── metrics/
│   │       ├── feature/                     # 预留特征层指标目录
│   │       ├── geometric/
│   │       │   ├── inlier_ratio.h
│   │       │   └── reprojection_error.h
│   │       ├── image/
│   │       │   ├── psnr.h
│   │       │   ├── rmse.h
│   │       │   └── ssim.h
│   │       └── keypoint/
│   │           └── repeatability.h
│   ├── filter/                              # 匹配过滤器
│   │   ├── cross_check.h
│   │   ├── distance_distribution_filter.h
│   │   ├── distance_threshold_filter.h
│   │   ├── gms_filter.h
│   │   ├── min_distance_filter.h
│   │   └── ratio_test.h
│   ├── geometry/                            # 几何变换估计器
│   │   ├── affine_estimator.h
│   │   ├── homography_estimator.h
│   │   ├── partial_affine_utils.h
│   │   ├── rigid_estimator.h
│   │   └── similarity_estimator.h
│   ├── interfaces/                          # 抽象接口
│   │   ├── i_direct_aligner.h
│   │   ├── i_filter.h
│   │   ├── i_geometry_estimator.h
│   │   ├── i_keypoint_extractor.h
│   │   ├── i_learning_matcher.h
│   │   ├── i_matcher.h
│   │   ├── i_pipeline.h
│   │   ├── i_registration.h
│   │   ├── i_structure_associator.h
│   │   └── i_structure_extractor.h
│   ├── keypoint/                            # 点特征提取器
│   │   ├── akaze_extractor.h
│   │   ├── brisk_extractor.h
│   │   ├── kaze_extractor.h
│   │   ├── orb_extractor.h
│   │   ├── sift_extractor.h
│   │   └── surf_extractor.h
│   ├── learning/                            # 深度学习匹配桥接
│   │   └── python_learning_matcher.h
│   ├── matcher/                             # 匹配器与结构关联器
│   │   ├── feature/                         # 预留特征匹配目录
│   │   ├── keypoint/
│   │   │   ├── bf_matcher.h
│   │   │   └── flann_matcher.h
│   │   └── structure/
│   │       ├── chamfer_associator.h
│   │       ├── contour_descriptor_associator.h
│   │       ├── hausdorff_associator.h
│   │       ├── icp_associator.h
│   │       ├── line_descriptor_associator.h
│   │       ├── line_segment_associator.h
│   │       ├── phase_correlate_associator.h
│   │       └── structure_point_set.h
│   ├── pipeline/                            # 配准流水线
│   │   ├── base_pipeline.h
│   │   ├── direct_pipeline.h
│   │   ├── keypoint_pipeline.h
│   │   ├── learning_pipeline.h
│   │   └── structure_pipeline.h
│   ├── structure/                           # 结构特征提取器
│   │   ├── contour_extractor.h
│   │   ├── edge_extractor.h
│   │   └── line_extractor.h
│   ├── transform/                           # 图像 warping
│   │   ├── affine_warper.h
│   │   ├── perspective_warper.h
│   │   └── warper.h
│   └── utils/                               # 工具库
│       ├── file_utils.h
│       ├── image_utils.h
│       ├── logger.h
│       ├── string_utils.h
│       ├── timer.h
│       ├── yaml_utils.h
│       └── visualization/
│           ├── draw_diff.h
│           ├── draw_inliers.h
│           ├── draw_matches.h
│           ├── draw_overlay.h
│           └── visualization_manager.h
│
├── outputs/                                 # 运行输出目录（生成物）
│   ├── batch/                               # 批量运行结果
│   ├── compare/                             # 横向对比实验结果
│   ├── debug_ecc_fix/                       # ECC 调试输出
│   └── single/                              # 单次运行结果
│
├── SOURCE_FILE_DOCS_CN/                     # 源文件中文说明文档
│   ├── CORE_CONFIG_FUNCTIONS_CN.md
│   ├── DIRECT_FOURIER_MELLIN_CN.md
│   ├── DIRECT_PHASE_CORRELATION_CN.md
│   ├── DIRECT_ZNCC_RIGID_CN.md
│   ├── README.md
│   └── REGISTRATION_APP_FUNCTIONS_UPDATED_CN.md
│
├── src/                                     # 源文件实现
│   ├── core/
│   │   ├── config.cpp
│   │   ├── factory.cpp
│   │   ├── registration.cpp
│   │   └── types.cpp
│   ├── data/
│   │   └── correspondence_view.cpp
│   ├── dataset/
│   │   └── dataset_loader.cpp
│   ├── direct/
│   │   ├── dense/
│   │   │   ├── dense_flow_common.cpp
│   │   │   ├── dis_flow_aligner.cpp
│   │   │   ├── farneback_flow_aligner.cpp
│   │   │   └── tvl1_flow_aligner.cpp
│   │   ├── frequency/
│   │   │   ├── fourier_mellin_aligner.cpp
│   │   │   └── phase_correlation_aligner.cpp
│   │   ├── global/
│   │   │   ├── ecc_aligner.cpp
│   │   │   ├── esm_rigid_aligner.cpp
│   │   │   ├── global_lk_aligner.cpp
│   │   │   └── zncc_rigid_aligner.cpp
│   │   └── sparse/
│   │       └── klt_sparse_aligner.cpp
│   ├── evaluator/
│   │   ├── evaluator.cpp
│   │   └── metrics/
│   │       ├── feature/                     # 当前无实现文件
│   │       ├── geometric/
│   │       │   └── inlier_ratio.cpp
│   │       ├── image/                       # 当前无实现文件
│   │       └── keypoint/
│   │           └── repeatability.cpp
│   ├── filter/
│   │   ├── cross_check.cpp
│   │   ├── distance_distribution_filter.cpp
│   │   ├── distance_threshold_filter.cpp
│   │   ├── gms_filter.cpp
│   │   ├── min_distance_filter.cpp
│   │   └── ratio_test.cpp
│   ├── geometry/
│   │   ├── affine_estimator.cpp
│   │   ├── homography_estimator.cpp
│   │   ├── rigid_estimator.cpp
│   │   └── similarity_estimator.cpp
│   ├── keypoint/
│   │   ├── akaze_extractor.cpp
│   │   ├── brisk_extractor.cpp
│   │   ├── kaze_extractor.cpp
│   │   ├── orb_extractor.cpp
│   │   ├── sift_extractor.cpp
│   │   └── surf_extractor.cpp
│   ├── learning/
│   │   └── python_learning_matcher.cpp
│   ├── matcher/
│   │   ├── keypoint/
│   │   │   ├── bf_matcher.cpp
│   │   │   └── flann_matcher.cpp
│   │   └── structure/
│   │       ├── chamfer_associator.cpp
│   │       ├── contour_descriptor_associator.cpp
│   │       ├── hausdorff_associator.cpp
│   │       ├── icp_associator.cpp
│   │       ├── line_descriptor_associator.cpp
│   │       ├── line_segment_associator.cpp
│   │       ├── phase_correlate_associator.cpp
│   │       └── structure_point_set.cpp
│   ├── pipeline/
│   │   ├── base_pipeline.cpp
│   │   ├── direct_pipeline.cpp
│   │   ├── keypoint_pipeline.cpp
│   │   ├── learning_pipeline.cpp
│   │   └── structure_pipeline.cpp
│   ├── structure/
│   │   ├── contour_extractor.cpp
│   │   ├── edge_extractor.cpp
│   │   └── line_extractor.cpp
│   ├── transform/
│   │   ├── affine_warper.cpp
│   │   ├── perspective_warper.cpp
│   │   └── warper.cpp
│   └── utils/
│       ├── file_utils.cpp
│       ├── image_utils.cpp
│       ├── logger.cpp
│       ├── string_utils.cpp
│       ├── timer.cpp
│       ├── yaml_utils.cpp
│       └── visualization/
│           ├── draw_diff.cpp
│           ├── draw_inliers.cpp
│           ├── draw_matches.cpp
│           ├── draw_overlay.cpp
│           └── visualization_manager.cpp
│
├── third_party/                             # 第三方源码依赖
│   ├── LightGlue/                           # LightGlue Python 依赖
│   └── SuperGluePretrainedNetwork/          # SuperGlue Python 依赖
│
├── tools/                                   # 辅助工具脚本
│   └── deep/
│       ├── learning_backend.py
│       ├── loftr_infer.py
│       ├── superpoint_lightglue_infer.py
│       └── superpoint_superglue_infer.py
│
├── .clang-format                            # C/C++ 格式化配置
├── CMakeLists.txt                           # CMake 构建入口
├── CODEX_WORKING_RULES_CN.md                # Codex 协作/工作规则
├── main.cpp                                 # 程序入口
├── PROJECT_DIRECTORY_STRUCTURE_CN.md        # 当前目录结构说明
├── PROJECT_PIPELINE_INTERNALS_CN.md         # 平台内部流水线、数据流、Factory 与 YAML 机制说明
├── PROJECT_QUICK_INTRO_CN.md                # 中文快速入门
└── README.md                                # 项目说明
```
