# 项目目录结构

```
project/                                     # 图像配准实验平台根目录
├── main.cpp                                 # 程序入口，委托给 RegistrationApp::run()
├── CMakeLists.txt                           # CMake 构建配置（C++17, OpenCV contrib, yaml-cpp）
│
├── README.md                                # 英文项目概述与快速入门
├── PROJECT_QUICK_INTRO_CN.md                # 中文快速入门：架构、模块、推荐阅读顺序
├── PROJECT_DIRECTORY_DETAILED_CN.md         # 中文逐文件详解：每个类/函数的用途和核心逻辑
├── METHOD_SYSTEM_OVERVIEW_CN.md             # 中文方法体系总览：点特征法 vs 结构法的完整方法矩阵
├── API_REFERENCE_CN.md                      # 中文 API 参考：所有公开类和函数的签名与说明
├── EXPERIMENT_PLATFORM_DEVELOPMENT_GUIDE_CN.md  # 中文开发指南：扩展新方法、环境、输出约定
│
├── apps/                                    # 应用程序入口
│   ├── registration_app.h                   #   RegistrationApp 类声明（单次/批量运行入口）
│   └── registration_app.cpp                 #   实现：CLI 解析、单次配准、批量配准、CSV/摘要输出
│
├── include/                                 # 公共头文件（接口、核心类、各模块）
│   ├── core/                                # 核心基础设施
│   │   ├── types.h                          #   枚举定义：KeypointType / StructureType / NormType / MatchMethod / GeometryType / ImageIndex 及转换函数
│   │   ├── config.h                         #   PipelineConfig 结构体（包含所有配置字段）+ Config 静态方法（YAML 加载）
│   │   ├── context.h                        #   RegistrationContext — 贯穿流水线的共享可变状态容器
│   │   ├── factory.h                        #   Factory — 根据 YAML 配置创建所有算法组件的静态工厂
│   │   ├── registration.h                   #   Registration — 顶层配准入口，包装 IPipeline
│   │   ├── result.h                         #   RegistrationResult — 运行结果（成功/失败、计数、耗时、IoU、NMAD）
│   │   └── transform_type.h                 #   TransformType 枚举及与 GeometryType 的映射
│   │
│   ├── interfaces/                          # 纯虚接口（策略模式基类）
│   │   ├── i_keypoint_extractor.h           #   IKeypointExtractor — 点特征提取器接口
│   │   ├── i_structure_extractor.h          #   IStructureExtractor — 结构特征提取器接口
│   │   ├── i_matcher.h                      #   IMatcher — 描述子匹配器接口（点特征）
│   │   ├── i_structure_associator.h         #   IStructureAssociator — 结构关联器接口（结构特征）
│   │   ├── i_filter.h                       #   IFilter — 匹配点对过滤器接口
│   │   ├── i_geometry_estimator.h           #   IGeometryEstimator — 几何变换估计器接口
│   │   ├── i_pipeline.h                     #   IPipeline — 流水线接口（模板方法模式基类）
│   │   ├── i_registration.h                 #   IRegistration — 顶层注册接口
│   │   └── i_warper.h                       #   IWarper — 图像 warping 接口
│   │
│   ├── data/                                # 数据结构（流水线各阶段的输入/输出载体）
│   │   ├── image_data.h                     #   ImagePairData — 原图对（BGR + 灰度）
│   │   ├── keypoint_data.h                  #   KeypointData / KeypointImageData — 关键点 + 描述子
│   │   ├── keypoint_match_data.h            #   KeypointMatchData — 原始匹配/过滤后匹配/内点
│   │   ├── structure_data.h                 #   StructureData / StructureImageData — 结构响应图 + 基元（线段/轮廓）
│   │   ├── structure_match_data.h           #   StructureMatchData — 结构关联结果（平移/仿射/线匹配）
│   │   ├── geometry_data.h                  #   GeometryData — 几何模型（H/A/内点数/重投影误差）
│   │   ├── transform_data.h                 #   TransformData — 变换矩阵
│   │   └── evaluation_data.h                #   EvaluationData — 评测指标结果集合
│   │
│   ├── keypoint/                            # 点特征提取器（6 种）
│   │   ├── sift_extractor.h                 #   SIFT 提取器
│   │   ├── surf_extractor.h                 #   SURF 提取器
│   │   ├── orb_extractor.h                  #   ORB 提取器
│   │   ├── brisk_extractor.h                #   BRISK 提取器
│   │   ├── kaze_extractor.h                 #   KAZE 提取器
│   │   └── akaze_extractor.h                #   AKAZE 提取器
│   │
│   ├── structure/                           # 结构特征提取器（3 种）
│   │   ├── edge_extractor.h                 #   EdgeExtractor — Canny / Sobel / LoG / Laplacian 边缘检测
│   │   ├── line_extractor.h                 #   LineExtractor — Hough / HoughP / LSD / FLD 直线检测
│   │   └── contour_extractor.h              #   ContourExtractor — Canny + findContours 轮廓检测
│   │
│   ├── matcher/                             # 匹配器与结构关联器
│   │   ├── keypoint/                        # 点特征匹配器（2 种）
│   │   │   ├── bf_matcher.h                 #   BfMatcher — 暴力匹配（L1/L2/Hamming）
│   │   │   └── flann_matcher.h              #   FlannMatcher — FLANN 近似最近邻匹配
│   │   └── structure/                       # 结构特征关联器（6 种）
│   │       ├── phase_correlate_associator.h  #   PhaseCorrelateAssociator — 相位相关（频域平移估计）
│   │       ├── chamfer_associator.h          #   ChamferAssociator — Chamfer 距离匹配
│   │       ├── hausdorff_associator.h        #   HausdorffAssociator — Hausdorff 距离匹配
│   │       ├── icp_associator.h              #   IcpAssociator — ICP 迭代最近点
│   │       ├── line_segment_associator.h     #   LineSegmentAssociator — 线段几何 baseline（无描述子）
│   │       ├── line_descriptor_associator.h  #   LineDescriptorAssociator — LBD 线描述子匹配 + 几何一致性
│   │       └── structure_point_set.h         #   structure_points 工具集（采样/DistanceTransform/质心）
│   │
│   ├── filter/                              # 匹配过滤器（6 种）
│   │   ├── ratio_test.h                     #   RatioTest — Lowe's ratio test
│   │   ├── cross_check.h                    #   CrossCheck — 双向一致性检查
│   │   ├── gms_filter.h                     #   GmsFilter — 基于网格的运动统计
│   │   ├── distance_threshold_filter.h       #   DistanceThresholdFilter — 描述子距离阈值过滤
│   │   ├── min_distance_filter.h            #   MinDistanceFilter — 最小描述子距离过滤
│   │   └── distance_distribution_filter.h   #   DistanceDistributionFilter — 距离分布统计过滤
│   │
│   ├── geometry/                            # 几何变换估计器（4 种）
│   │   ├── homography_estimator.h           #   HomographyEstimator — 单应矩阵（3×3），最少 4 对点
│   │   ├── affine_estimator.h               #   AffineEstimator — 仿射变换（2×3），最少 3 对点
│   │   ├── rigid_estimator.h                #   RigidEstimator — 刚性变换（旋转+平移），最少 2 对点
│   │   └── similarity_estimator.h           #   SimilarityEstimator — 相似变换（旋转+缩放+平移），最少 2 对点
│   │
│   ├── transform/                           # 图像 warping
│   │   ├── warper.h                         #   IWarper — warping 接口
│   │   ├── perspective_warper.h             #   PerspectiveWarper — 透视变换 warping
│   │   └── affine_warper.h                  #   AffineWarper — 仿射变换 warping
│   │
│   ├── pipeline/                            # 流水线（模板方法模式）
│   │   ├── base_pipeline.h                  #   BasePipeline — 公共骨架（load → extract → associate → estimate → warp → validate → save）
│   │   ├── keypoint_pipeline.h              #   KeypointPipeline — 点特征配准流水线
│   │   └── structure_pipeline.h             #   StructurePipeline — 结构特征配准流水线
│   │
│   ├── dataset/                             # 数据集管理
│   │   ├── sample.h                         #   Sample — 单样本结构（source/target 路径 + ground truth 变换）
│   │   └── dataset_loader.h                 #   DatasetLoader — 从 YAML 加载数据集样本列表
│   │
│   ├── evaluator/                           # 评测系统
│   │   ├── evaluator.h                      #   Evaluator — 指标执行调度器
│   │   ├── benchmark.h                      #   Benchmark — 批量评测入口（按指标/方法/数据集运行）
│   │   ├── statistics.h                     #   Statistics — 统计计算（均值/方差/直方图）
│   │   └── metrics/                         # 评测指标实现
│   │       ├── feature/                      #   特征层指标
│   │       ├── geometric/                   #   几何层指标
│   │       │   ├── inlier_ratio.h            #     InlierRatio — 内点率
│   │       │   └── reprojection_error.h      #     ReprojectionError — 重投影误差
│   │       ├── image/                       #   图像层指标
│   │       │   ├── psnr.h                   #     PSNR — 峰值信噪比
│   │       │   ├── rmse.h                   #     RMSE — 均方根误差
│   │       │   └── ssim.h                   #     SSIM — 结构相似性
│   │       └── keypoint/                    #   关键点指标
│   │           └── repeatability.h           #     Repeatability — 关键点可重复性
│   │
│   └── utils/                               # 工具库
│       ├── logger.h                         #   Logger — 带级别的日志输出（INFO/WARN/ERROR/DEBUG）
│       ├── timer.h                          #   Timer / ScopedTimer — 计时器
│       ├── file_utils.h                     #   FileUtils — 文件读写、CSV 转义、路径操作
│       ├── image_utils.h                    #   ImageUtils — 图像读写、深度图/高比特转换
│       ├── yaml_utils.h                     #   YAML 安全读取工具（getString/getInt/getDouble/getBool with default）
│       └── visualization/                   # 可视化工具
│           ├── visualization_manager.h      #   VisualizationManager — 可视化调度器
│           ├── draw_matches.h               #   drawMatches — 绘制匹配点连线图
│           ├── draw_inliers.h               #   drawInliers — 绘制内点连线图
│           ├── draw_overlay.h               #   drawOverlay — 绘制半透明叠加图
│           └── draw_diff.h                  #   drawDiff — 绘制差值图
│
├── src/                                     # 源文件实现（与 include/ 目录结构一一对应）
│   ├── core/
│   │   ├── config.cpp                       #   Config::load / resolvePath / loadPipeline — YAML 加载与路径解析
│   │   ├── factory.cpp                      #   Factory — 所有 create* 方法实现（switch-case 分发）
│   │   ├── registration.cpp                 #   Registration::configure / run
│   │   └── types.cpp                        #   枚举 toString / fromString / toCvNorm 实现
│   ├── keypoint/                            #   6 种点特征提取器实现
│   ├── structure/                           #   3 种结构特征提取器实现
│   │   ├── line_extractor.cpp               #   含线段去重、LSD 专用检测器、后处理流水线
│   │   ├── edge_extractor.cpp
│   │   └── contour_extractor.cpp
│   ├── matcher/
│   │   ├── keypoint/                        #   BFMatcher / FlannMatcher 实现
│   │   └── structure/                       #   6 种结构关联器实现
│   │       ├── line_descriptor_associator.cpp # LBD 描述子 + BinaryDescriptorMatcher + 几何一致性投票
│   │       ├── line_segment_associator.cpp    # 纯几何 baseline（角度/长度/中心位移投票）
│   │       ├── phase_correlate_associator.cpp
│   │       ├── chamfer_associator.cpp
│   │       ├── hausdorff_associator.cpp
│   │       ├── icp_associator.cpp
│   │       └── structure_point_set.cpp        # 结构点集工具函数
│   ├── filter/                              #   6 种过滤器实现
│   ├── geometry/                            #   4 种几何估计器实现
│   │   └── partial_affine_utils.h           #   局部仿射内部工具（仅头文件）
│   ├── transform/                           #   PerspectiveWarper / AffineWarper 实现
│   ├── pipeline/
│   │   ├── base_pipeline.cpp                #   公共骨架（loadImages / runWarp / validateWarpQuality / saveOutputs）
│   │   ├── keypoint_pipeline.cpp            #   点特征管线（extract → match → filter → estimate）
│   │   └── structure_pipeline.cpp           #   结构管线（extract → associate → estimate / 端点对回退）
│   ├── dataset/
│   │   └── dataset_loader.cpp
│   ├── evaluator/                           #   评测指标实现
│   │   └── metrics/
│   │       ├── geometric/
│   │       │   └── inlier_ratio.cpp
│   │       └── keypoint/
│   │           └── repeatability.cpp
│   └── utils/
│       ├── logger.cpp
│       ├── timer.cpp
│       ├── file_utils.cpp
│       ├── image_utils.cpp
│       ├── yaml_utils.cpp
│       └── visualization/                   #   可视化工具实现
│           ├── visualization_manager.cpp
│           ├── draw_matches.cpp
│           ├── draw_inliers.cpp
│           ├── draw_overlay.cpp
│           └── draw_diff.cpp
│
├── configs/                                 # YAML 配置文件（全平台配置驱动，无硬编码参数）
│   ├── keypoint/                            # 点特征提取器配置（6 个）
│   │   ├── sift.yaml                        #   SIFT 参数（nfeatures/nOctaveLayers/contrastThreshold 等）
│   │   ├── surf.yaml                        #   SURF 参数
│   │   ├── orb.yaml                         #   ORB 参数
│   │   ├── brisk.yaml                       #   BRISK 参数
│   │   ├── kaze.yaml                        #   KAZE 参数
│   │   └── akaze.yaml                       #   AKAZE 参数
│   ├── structure/                           # 结构特征提取器 + 关联器配置（3 个）
│   │   ├── edge.yaml                        #   边缘检测方法 + 关联方法参数
│   │   ├── line.yaml                        #   直线检测方法 + LBD/几何匹配参数
│   │   └── contour.yaml                     #   轮廓检测方法 + 关联方法参数
│   ├── matcher/                             # 匹配器配置（2 个）
│   │   ├── bf.yaml                          #   暴力匹配器（normType/crossCheck）
│   │   └── flann.yaml                       #   FLANN 匹配器（trees/checks）
│   ├── filter/                              # 过滤器配置（6 个）
│   │   ├── ratio_test.yaml
│   │   ├── cross_check.yaml
│   │   ├── gms.yaml
│   │   ├── distance_threshold.yaml
│   │   ├── min_distance.yaml
│   │   └── distance_distribution.yaml
│   ├── geometry/                            # 几何估计器配置（4 个）
│   │   ├── homography.yaml                  #   单应估计（method/threshold/confidence/maxIters）
│   │   ├── affine.yaml                      #   仿射估计
│   │   ├── rigid.yaml                       #   刚性估计
│   │   └── similarity.yaml                  #   相似估计
│   ├── pipeline/                            # 流水线编排配置
│   │   ├── keypoint/                        #   点特征流水线（6 个，各组合 keypoint+matcher+filter+geometry）
│   │   │   ├── sift_pipeline.yaml
│   │   │   ├── surf_pipeline.yaml
│   │   │   ├── orb_pipeline.yaml
│   │   │   ├── brisk_pipeline.yaml
│   │   │   ├── kaze_pipeline.yaml
│   │   │   └── akaze_pipeline.yaml
│   │   ├── structure/                       #   结构特征流水线（3 个）
│   │   │   ├── edge_pipeline.yaml           #     边缘 → ICP/Chamfer/Hausdorff/PhaseCorrelate
│   │   │   ├── line_pipeline.yaml           #     直线 → LineDescriptor/LineSegment + Rigid 回退
│   │   │   └── contour_pipeline.yaml        #     轮廓 → PhaseCorrelate/Chamfer/Hausdorff/ICP
│   │   └── batch/                           #   批量配置
│   │       ├── batch_keypoint.yaml          #     批量点特征评测（数据集+流水线组合）
│   │       └── batch_structure.yaml         #     批量结构特征评测
│   └── evaluator/                           # 评测配置
│       ├── metrics.yaml                     #   指标配置（启用哪些指标及其参数）
│       └── benchmark.yaml                   #   基准测试配置（方法列表+指标列表+数据集+重复次数）
│
├── datasets/                                # 测试数据集
│   └── test1/ .. test10/                    #   每个子文件夹包含 source.png + target.png 图像对
│
├── outputs/                                 # 运行输出目录（运行时自动创建）
│   ├── single/                              #   单次运行输出
│   │   └── {keypoint|structure}/{pipeline}/{sample}/
│   │       ├── originals/                   #     原始图像
│   │       ├── keypoints/ 或 structures/    #     特征/结构响应图
│   │       ├── matches/                     #     匹配连线图（all_match / inlier_match）
│   │       ├── warped/                      #     warped 图像
│   │       ├── blend/                       #     叠加混合图
│   │       ├── summary.txt                  #     运行摘要文本
│   │       └── summary.json                 #     运行摘要 JSON
│   └── batch/                               #   批量运行输出
│       └── {keypoint|structure}/{pipeline}/
│           ├── results.csv                  #     汇总 CSV（所有样本的结果行）
│           └── {sample}/                    #     每个样本的详细输出（结构同 single/）
│
└── build-mingw/                             # MinGW 编译输出目录（预编译产物）
    └── registration_app.exe                 #   可执行文件
```
