# 项目目录结构

下面按当前 `project/` 目录的实际层级整理，使用树状结构展示。  
构建产物、输出目录和第三方依赖只保留关键层级，避免把大量生成文件全部展开。

```text
project/                                     # 图像配准实验平台根目录
├── apps/                                    # 应用程序入口与汇总输出
│   ├── registration_app.cpp                 # CLI 解析、单次/批量/对比运行调度
│   ├── registration_app.h                   # RegistrationApp 类声明
│   ├── registration_app_helpers.cpp         # 应用层摘要、JSON、文本输出辅助
│   ├── registration_app_helpers.h           # 应用层输出辅助声明
│   ├── summary_csv_writer.cpp               # summary.csv 表头与行写出逻辑
│   └── summary_csv_writer.h                 # summary.csv 写出接口
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
│   │   └── klt_sparse.yaml
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
│   ├── learning/                            # 学习法配置
│   │   ├── loftr.yaml
│   │   ├── superpoint_lightglue.yaml
│   │   └── superpoint_superglue.yaml
│   ├── matcher/                             # 匹配器配置
│   │   ├── bf.yaml
│   │   └── flann.yaml
│   ├── pipeline/                            # 流水线编排配置
│   │   ├── batch/                           # 批量与对比运行配置
│   │   │   ├── batch_direct.yaml
│   │   │   ├── batch_keypoint.yaml
│   │   │   ├── batch_learning.yaml
│   │   │   ├── batch_structure.yaml
│   │   │   ├── compare_direct.yaml
│   │   │   ├── compare_frequency_initializers.yaml
│   │   │   ├── compare_keypoint.yaml
│   │   │   └── compare_line.yaml
│   │   ├── direct/                          # 直接法流水线配置
│   │   │   ├── dense_direct_pipeline.yaml
│   │   │   ├── frequency_direct_pipeline.yaml
│   │   │   ├── global_direct_pipeline.yaml
│   │   │   └── sparse_direct_pipeline.yaml
│   │   ├── keypoint/                        # 点特征流水线配置
│   │   │   ├── akaze_pipeline.yaml
│   │   │   ├── brisk_pipeline.yaml
│   │   │   ├── kaze_pipeline.yaml
│   │   │   ├── orb_pipeline.yaml
│   │   │   ├── sift_pipeline.yaml
│   │   │   └── surf_pipeline.yaml
│   │   ├── learning/                        # 学习法流水线配置
│   │   │   ├── loftr_learning_pipeline.yaml
│   │   │   ├── superpoint_lightglue_learning_pipeline.yaml
│   │   │   └── superpoint_superglue_learning_pipeline.yaml
│   │   ├── structure/                       # 结构法流水线配置
│   │   │   ├── contour_pipeline.yaml
│   │   │   └── line_pipeline.yaml
│   │   └── frequency_compare/               # 频域法与 initializer 临时对比配置
│   └── structure/                           # 结构特征提取与关联配置
│       ├── contour.yaml
│       └── line.yaml
│
├── datasets/                                # 测试数据集
│   ├── Test01/
│   ├── Test02/
│   ├── Test03/
│   ├── Test04/
│   ├── Test05/
│   ├── Test06/
│   ├── Test07/
│   ├── Test08/
│   ├── Test09/
│   ├── Test10/
│   ├── Test11/
│   └── Test12/
│
├── include/                                 # 头文件目录
│   ├── core/                                # 核心配置、上下文、结果与工厂
│   │   ├── config.h
│   │   ├── context.h
│   │   ├── factory.h
│   │   ├── registration.h
│   │   ├── result.h
│   │   └── types.h
│   ├── data/                                # 流水线阶段间共享数据结构
│   │   ├── correspondence_view.h
│   │   ├── direct_data.h
│   │   ├── evaluation_data.h
│   │   ├── feature_initializer_data.h
│   │   ├── geometry_data.h
│   │   ├── image_data.h
│   │   ├── keypoint_data.h
│   │   ├── keypoint_match_data.h
│   │   ├── structure_data.h
│   │   ├── structure_match_data.h
│   │   └── transform_data.h
│   ├── dataset/                             # 数据集加载接口
│   │   ├── dataset_loader.h
│   │   └── sample.h
│   ├── direct/                              # 直接法模块
│   │   ├── common/                          # 直接法公共几何与优化辅助
│   │   │   ├── dense_flow_common.h
│   │   │   ├── direct_geometry_common.h
│   │   │   ├── ecc_common.h
│   │   │   ├── esm_rigid_common.h
│   │   │   └── rigid_direct_common.h
│   │   ├── dense/                           # 稠密光流直接法
│   │   │   ├── dis_flow_aligner.h
│   │   │   └── farneback_flow_aligner.h
│   │   ├── frequency/                       # 频域直接法
│   │   │   └── fourier_mellin_aligner.h
│   │   ├── global/                          # 全局优化类直接法
│   │   │   ├── ecc_aligner.h
│   │   │   └── esm_rigid_aligner.h
│   │   └── sparse/                          # 稀疏直接法
│   │       └── klt_sparse_aligner.h
│   ├── evaluator/                           # 评测系统
│   │   ├── evaluator.h
│   │   ├── metrics/
│   │   │   ├── geometric/
│   │   │   │   ├── inlier_ratio.h
│   │   │   │   └── reprojection_error.h
│   │   │   └── keypoint/
│   │   │       └── repeatability.h
│   │   └── quality/
│   │       └── warp_quality_evaluator.h
│   ├── filter/                              # 匹配过滤器声明
│   │   ├── cross_check.h
│   │   ├── distance_distribution_filter.h
│   │   ├── distance_threshold_filter.h
│   │   ├── gms_filter.h
│   │   ├── min_distance_filter.h
│   │   └── ratio_test.h
│   ├── geometry/                            # 几何估计器声明
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
│   ├── keypoint/                            # 点特征提取器声明
│   │   ├── akaze_extractor.h
│   │   ├── brisk_extractor.h
│   │   ├── kaze_extractor.h
│   │   ├── orb_extractor.h
│   │   ├── sift_extractor.h
│   │   └── surf_extractor.h
│   ├── learning/                            # 学习法桥接声明
│   │   └── python_learning_matcher.h
│   ├── matcher/                             # 匹配器与结构关联器声明
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
│   │       └── structure_point_set.h
│   ├── pipeline/                            # 流水线与公共 helper
│   │   ├── base_pipeline.h
│   │   ├── base_pipeline_helpers.h
│   │   ├── direct_feature_initializer.h
│   │   ├── direct_pipeline.h
│   │   ├── direct_pipeline_helpers.h
│   │   ├── keypoint_pipeline.h
│   │   ├── learning_pipeline.h
│   │   └── structure_pipeline.h
│   ├── structure/                           # 结构特征提取器声明
│   │   ├── contour_extractor.h
│   │   └── line_extractor.h
│   ├── transform/                           # 图像 warper 声明
│   │   ├── affine_warper.h
│   │   ├── perspective_warper.h
│   │   └── warper.h
│   └── utils/                               # 通用工具
│       ├── descriptor_norm_utils.h
│       ├── file_utils.h
│       ├── image_utils.h
│       ├── logger.h
│       ├── string_utils.h
│       ├── timer.h
│       ├── yaml_utils.h
│       └── visualization/                   # 可视化辅助
│           ├── direct/
│           │   └── draw_warp_difference.h
│           ├── structure/
│           │   └── draw_structure_matches.h
│           ├── draw_diff.h
│           ├── draw_inliers.h
│           ├── draw_matches.h
│           ├── draw_overlay.h
│           └── visualization_manager.h
│
├── outputs/                                 # 运行输出目录（生成物）
│   ├── batch/                               # 批量运行结果
│   ├── compare/                             # 横向对比实验结果
│   └── single/                              # 单次运行结果
│
├── reports/                                 # 实验分析与报告输出目录
│   ├── contour_descriptor/
│   ├── direct/
│   ├── keypoint/
│   └── line_descriptor/
│
├── src/                                     # 源文件实现目录
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
│   │   │   ├── common/
│   │   │   │   └── dense_flow_common.cpp
│   │   │   ├── dis_flow_aligner.cpp
│   │   │   └── farneback_flow_aligner.cpp
│   │   ├── frequency/
│   │   │   └── fourier_mellin_aligner.cpp
│   │   ├── global/
│   │   │   ├── ecc_aligner.cpp
│   │   │   └── esm_rigid_aligner.cpp
│   │   └── sparse/
│   │       └── klt_sparse_aligner.cpp
│   ├── evaluator/
│   │   ├── evaluator.cpp
│   │   ├── metrics/
│   │   │   ├── geometric/
│   │   │   │   └── inlier_ratio.cpp
│   │   │   └── keypoint/
│   │   │       └── repeatability.cpp
│   │   └── quality/
│   │       └── warp_quality_evaluator.cpp
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
│   │       └── structure_point_set.cpp
│   ├── pipeline/
│   │   ├── base_pipeline.cpp
│   │   ├── base_pipeline_helpers.cpp
│   │   ├── direct_feature_initializer.cpp
│   │   ├── direct_pipeline.cpp
│   │   ├── direct_pipeline_helpers.cpp
│   │   ├── keypoint_pipeline.cpp
│   │   ├── learning_pipeline.cpp
│   │   └── structure_pipeline.cpp
│   ├── structure/
│   │   ├── contour_extractor.cpp
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
│           ├── direct/
│           │   └── draw_warp_difference.cpp
│           ├── structure/
│           │   └── draw_structure_matches.cpp
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
├── tools/                                   # 工具脚本目录
│   ├── deep/
│   │   ├── learning_backend.py
│   │   ├── loftr_infer.py
│   │   ├── superpoint_lightglue_infer.py
│   │   └── superpoint_superglue_infer.py
│   ├── generate_contour_descriptor_report.ps1
│   ├── generate_keypoint_rigid_report.ps1
│   ├── generate_line_descriptor_report.ps1
│   └── generate_test_cases_report.ps1
│
├── .clang-format                            # C/C++ 格式化配置
├── CMakeLists.txt                           # CMake 构建入口
├── CODEX_WORKING_RULES_CN.md                # Codex 协作/工作规则
├── EXPERIMENT_PLATFORM_DEVELOPMENT_USAGE_REPORT_CN.md  # 平台开发与使用说明报告
├── LESSONS.md                               # 项目内复盘与经验记录
├── main.cpp                                 # 程序入口
├── METHOD_FAMILY_TEST_ANALYSIS_CN.md        # 方法族测试分析文档
├── PROJECT_DIRECTORY_STRUCTURE_CN.md        # 当前目录结构说明
├── PROJECT_PIPELINE_INTERNALS_CN.md         # 平台内部流水线、数据流与工厂机制说明
├── README.md                                # 项目说明
├── REGISTRATION_VALIDATION_LOGIC_CN.md      # 配准成功判定逻辑说明
└── RIGID_CONSISTENCY_AND_CANDIDATE_REPORT_CN.md  # rigid 候选与一致性分析说明
```

## 维护约定

- 新增、删除、移动或重命名重要目录、方法族目录、配置目录或核心模块时，应同步更新本文档。
- 如果只是输出目录、构建缓存或临时对比结果变化，不要求把所有生成物细节写入本文档。
