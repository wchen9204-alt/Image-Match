#include "geometry/rigid_estimator.h"

#include <limits>
#include <vector>

#include <opencv2/calib3d.hpp>

#include "data/correspondence_view.h"
#include "geometry/partial_affine_utils.h"
#include "geometry/rigid_estimator_helpers.h"
#include "utils/logger.h"
#include "utils/string_utils.h"
#include "utils/yaml_utils.h"

namespace ir {

// 从 YAML 配置读取 rigid 估计器参数，并初始化多候选与评分相关开关。
RigidEstimator::RigidEstimator(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    _estimatorBackend = rigid_estimator_helpers::normalizeRigidEstimatorBackend(
        yaml_utils::getString(params, "estimatorBackend", "OPENCV_PARTIAL_AFFINE"));
    const std::string method_str = yaml_utils::getString(params, "method", "RANSAC");
    const int m = robustMethodFromString(method_str);
    _method = (m < 0) ? cv::RANSAC : m;
    if (_estimatorBackend == "OPENCV_PARTIAL_AFFINE" && _method == cv::USAC_MAGSAC) {
        IR_LOG_WARN("RigidEstimator: estimateAffinePartial2D does not support USAC_MAGSAC, fallback to RANSAC.");
        _method = cv::RANSAC;
    }

    _ransacReprojThreshold = yaml_utils::getDouble(params, "ransacReprojThreshold", 3.0);
    _maxIters = yaml_utils::getInt(params, "maxIters", 2000);
    _confidence = yaml_utils::getDouble(params, "confidence", 0.99);
    _refineIters = yaml_utils::getInt(params, "refineIters", 10);
    _minInliers = yaml_utils::getInt(params, "minInliers", 3);
    _rigidRefineMode = string_utils::toUpperAscii(
        yaml_utils::getString(params, "rigidRefineMode", "SVD"));
    _enableFilteredMatchCandidates =
        yaml_utils::getBool(params, "enableFilteredMatchCandidates", false);
    _filteredMatchCandidateTopK =
        std::max(2, yaml_utils::getInt(params, "filteredMatchCandidateTopK", 24));
    _filteredMatchCandidatePoolSize =
        std::max(_filteredMatchCandidateTopK,
                 yaml_utils::getInt(params, "filteredMatchCandidatePoolSize", 64));
    _filteredMatchCandidateCount =
        std::max(0, yaml_utils::getInt(params, "filteredMatchCandidateCount", 12));
    _filteredMatchCandidateMinPairDistance =
        std::max(0.0, yaml_utils::getDouble(params, "filteredMatchCandidateMinPairDistance", 20.0));
    _enableCandidateMaskScoring =
        yaml_utils::getBool(params, "enableCandidateMaskScoring", false);
    _candidateMaskForegroundThreshold =
        std::clamp(yaml_utils::getInt(params, "candidateMaskForegroundThreshold", 10), 0, 255);
    _candidateMinContainment =
        yaml_utils::getDouble(params, "candidateMinContainment", -1.0);
    _candidateMinBidirectionalCoverage =
        yaml_utils::getDouble(params, "candidateMinBidirectionalCoverage", -1.0);
    _candidateDedupRotationDiffDeg =
        std::max(0.0, yaml_utils::getDouble(params, "candidateDedupRotationDiffDeg", 2.0));
    _candidateDedupTranslationDiff =
        std::max(0.0, yaml_utils::getDouble(params, "candidateDedupTranslationDiff", 3.0));
    if (_rigidRefineMode != "SVD" &&
        _rigidRefineMode != "NONE") {
        IR_LOG_WARN("RigidEstimator: unknown rigidRefineMode=",
                    _rigidRefineMode,
                    ", fallback to SVD.");
        _rigidRefineMode = "SVD";
    }

    IR_LOG_INFO("RigidEstimator: method=",
                method_str,
                ", estimatorBackend=",
                _estimatorBackend,
                ", thr=",
                _ransacReprojThreshold,
                ", maxIters=",
                _maxIters,
                ", confidence=",
                _confidence,
                ", refineIters=",
                _refineIters,
                ", minInliers=",
                _minInliers,
                ", rigidRefineMode=",
                _rigidRefineMode,
                ", enableFilteredMatchCandidates=",
                _enableFilteredMatchCandidates,
                ", filteredMatchCandidateTopK=",
                _filteredMatchCandidateTopK,
                ", filteredMatchCandidatePoolSize=",
                _filteredMatchCandidatePoolSize,
                ", filteredMatchCandidateCount=",
                _filteredMatchCandidateCount,
                ", filteredMatchCandidateMinPairDistance=",
                _filteredMatchCandidateMinPairDistance,
                ", enableCandidateMaskScoring=",
                _enableCandidateMaskScoring,
                ", candidateMaskForegroundThreshold=",
                _candidateMaskForegroundThreshold,
                ", candidateMinContainment=",
                _candidateMinContainment,
                ", candidateMinBidirectionalCoverage=",
                _candidateMinBidirectionalCoverage,
                ", candidateDedupRotationDiffDeg=",
                _candidateDedupRotationDiffDeg,
                ", candidateDedupTranslationDiff=",
                _candidateDedupTranslationDiff);
}

// 执行 rigid 估计主流程：baseline 求解、候选补充、候选评分与最终结果回写。
bool RigidEstimator::estimate(RegistrationContext& ctx) {
    auto& gd = ctx.geometry_data;

    // 1. 重置几何结果，并标记当前估计器输出的是 rigid 模型。
    gd.clear();
    gd.type = GeometryType::RIGID;

    // 2. 从上下文中解析当前对应点来源，并构造统一的对应关系视图。
    const CorrespondenceSource source = correspondenceSourceFromContext(ctx);
    const CorrespondenceView view =
        source == CorrespondenceSource::NONE ? buildBestCorrespondenceView(ctx)
                                             : buildCorrespondenceView(ctx, source);

    // 3. rigid 至少需要两对点才能估计旋转和平移，不满足时直接失败。
    if (view.filtered.size() < 2) {
        gd.message = "need at least 2 correspondences, got " + std::to_string(view.filtered.size());
        IR_LOG_ERROR("RigidEstimator: need at least 2 correspondences, got ", view.filtered.size());
        return false;
    }

    // 4. 将统一视图中的 filtered 匹配转成刚体估计所需的点对数组。
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts2;
    partial_affine_utils::extractPoints(view, pts1, pts2);

    // `extractPoints()` 会跳过索引非法的匹配，因此这里同步提取一份与 pts1/pts2
    // 完全对齐的距离数组，后续“按 filtered 距离挑候选”时就不会和点坐标错位。
    std::vector<float> filteredDistances;
    filteredDistances.reserve(view.filtered.size());
    for (const auto& m : view.filtered) {
        if (m.queryIdx < 0 || m.trainIdx < 0 ||
            m.queryIdx >= static_cast<int>(view.first_keypoints.size()) ||
            m.trainIdx >= static_cast<int>(view.second_keypoints.size())) {
            continue;
        }
        filteredDistances.push_back(m.distance);
    }
    if (pts1.size() < 2 || pts2.size() < 2) {
        gd.message = "need at least 2 valid correspondences after index validation";
        IR_LOG_ERROR("RigidEstimator: need at least 2 valid correspondences after index validation, got ",
                     pts1.size());
        return false;
    }

    cv::Mat A;
    std::vector<unsigned char> mask;
    bool refined = false;
    int seed_inliers = 0;

    // baseline 和额外候选都统一放进这里，最后按同一套规则比较，
    // 避免“新候选”和“原 RANSAC 结果”各用一套标准而不好判断优先级。
    std::vector<cv::Mat> candidateTransforms;
    std::vector<std::vector<unsigned char>> candidateMasks;

    // 5. 按 YAML 选择初始估计后端：
    //    - OPENCV_PARTIAL_AFFINE：先走 estimateAffinePartial2D，再按 rigidRefineMode 压回刚体。
    //    - CUSTOM_RIGID_RANSAC：直接在 s=1 约束下做自定义 RANSAC。
    if (_estimatorBackend == "CUSTOM_RIGID_RANSAC") {
        refined = partial_affine_utils::estimateRigidRansacNoScale2D(
            pts1, pts2, _ransacReprojThreshold, _maxIters, _confidence, A, mask, true);
        if (!refined) {
            gd.message = "custom rigid RANSAC returned no valid model";
            IR_LOG_ERROR("RigidEstimator: custom rigid RANSAC returned no valid model.");
            return false;
        }

        seed_inliers = partial_affine_utils::countInliers(mask);
        // 自定义 rigid RANSAC 本身已经是严格刚体模型，也作为一个候选保留下来。
        candidateTransforms.push_back(A.clone());
        candidateMasks.push_back(mask);
        IR_LOG_INFO("RigidEstimator custom rigid RANSAC inliers=",
                    seed_inliers,
                    " / ",
                    view.filtered.size(),
                    " (mask_size=",
                    mask.size(),
                    ", source=",
                    view.source_name,
                    ")");
        if (_rigidRefineMode != "SVD") {
            IR_LOG_INFO("RigidEstimator: rigidRefineMode=",
                        _rigidRefineMode,
                        " ignored for CUSTOM_RIGID_RANSAC because the model is already strict rigid.");
        }
    } else {
        cv::Mat seedA = cv::estimateAffinePartial2D(pts1,
                                                    pts2,
                                                    mask,
                                                    _method,
                                                    _ransacReprojThreshold,
                                                    static_cast<size_t>(_maxIters),
                                                    _confidence,
                                                    static_cast<size_t>(_refineIters));

        if (seedA.empty()) {
            gd.message = "estimateAffinePartial2D returned an empty matrix";
            IR_LOG_ERROR("estimateAffinePartial2D returned an empty matrix.");
            return false;
        }
        A = seedA;
        seed_inliers = partial_affine_utils::countInliers(mask);
        IR_LOG_INFO("RigidEstimator OpenCV RANSAC inliers=",
                    seed_inliers,
                    " / ",
                    view.filtered.size(),
                    " (mask_size=",
                    mask.size(),
                    ", source=",
                    view.source_name,
                    ")");

        // 6. 根据 rigidRefineMode 决定是否进一步把模型压回“纯刚体”。
        //    纯刚体是 3DoF：只允许旋转 + 平移，要求 s=1，不允许缩放。
        if (_rigidRefineMode == "NONE") {
            // NONE 表示跳过二次精修，直接沿用 OpenCV partial affine 的 A 和 mask。
            // 注意：这种结果可能包含统一缩放，不是严格刚体。
            refined = true;
            IR_LOG_INFO("RigidEstimator NONE skipped rigid refinement, inliers=",
                        partial_affine_utils::countInliers(mask),
                        " / ",
                        view.filtered.size());
        } else {
            // SVD 模式使用 OpenCV RANSAC 内点，再迭代回归严格刚体。
            refined = partial_affine_utils::refineRigidFromMask(
                pts1, pts2, _ransacReprojThreshold, mask, A, true);
            IR_LOG_INFO("RigidEstimator SVD refined=",
                        refined,
                        ", refined_inliers=",
                        refined ? partial_affine_utils::countInliers(mask) : 0,
                        " / ",
                        view.filtered.size());
        }

        if (refined && !A.empty()) {
            // 先把当前 baseline rigid 结果入池；后面若启用额外候选，就和它一起打分。
            candidateTransforms.push_back(A.clone());
            candidateMasks.push_back(mask);
        }
    }

    const bool canTryFilteredMatchCandidates =
        _enableFilteredMatchCandidates &&
        _estimatorBackend != "CUSTOM_RIGID_RANSAC" &&
        static_cast<int>(pts1.size()) >= 2 &&
        !filteredDistances.empty() &&
        _filteredMatchCandidateCount > 0;

    // baseline refine 失败时，如果开启了 filtered 多候选，就继续尝试候选模型。
    // 这正是为了处理 RANSAC 初始模型陷入局部最优、但 filtered 中仍有可用点的情况。
    if (!refined) {
        gd.message = "failed to refine rigid transform from RANSAC inliers";
        IR_LOG_WARN("RigidEstimator: baseline refine failed, mode=",
                    _estimatorBackend,
                    "/",
                    _rigidRefineMode,
                    ", seed_inliers=",
                    seed_inliers,
                    " / ",
                    view.filtered.size(),
                    canTryFilteredMatchCandidates ? "; trying filtered-match candidates."
                                                  : "; no filtered-match candidate fallback available.");
        A.release();
        mask.clear();
        if (!canTryFilteredMatchCandidates) {
            return false;
        }
    }

    if (canTryFilteredMatchCandidates) {
        // 从已经经过 filter 的匹配里继续抽样。
        // 目标不是扩大搜索范围，而是在“相对更可信”的点里补几个严格 rigid 假设，
        // 缓解 baseline RANSAC 落到局部最优的情况。
        const std::vector<int> seedIndices =
            rigid_estimator_helpers::buildMixedCandidateSeedIndices(pts1,
                                                                    pts2,
                                                                    filteredDistances,
                                                                    _filteredMatchCandidatePoolSize,
                                                                    _filteredMatchCandidateTopK,
                                                                    _filteredMatchCandidateMinPairDistance);
        const int topK = static_cast<int>(seedIndices.size());
        const double minPairDistance2 =
            _filteredMatchCandidateMinPairDistance * _filteredMatchCandidateMinPairDistance;
        int generatedCandidates = 0;
        for (int i = 0; i < topK && generatedCandidates < _filteredMatchCandidateCount; ++i) {
            for (int j = i + 1; j < topK && generatedCandidates < _filteredMatchCandidateCount; ++j) {
                const int first = seedIndices[static_cast<size_t>(i)];
                const int second = seedIndices[static_cast<size_t>(j)];

                // 两个样本点如果在 source 或 target 上离得太近，旋转方向会非常不稳定，
                // 这类 2 点 rigid 假设容易退化，直接跳过。
                if (rigid_estimator_helpers::pairDistance2(pts1[first], pts1[second]) < minPairDistance2 ||
                    rigid_estimator_helpers::pairDistance2(pts2[first], pts2[second]) < minPairDistance2) {
                    continue;
                }

                cv::Mat candidateA;
                std::vector<unsigned char> candidateMask;

                // 先用 2 对点生成一个最小刚体假设，再投影回全部 filtered 点上拿到初始 mask。
                if (!rigid_estimator_helpers::buildRigidCandidateFromPair(
                        pts1, pts2, first, second, _ransacReprojThreshold, candidateA, candidateMask)) {
                    continue;
                }

                bool candidateRefined = false;
                if (_rigidRefineMode == "NONE") {
                    candidateRefined = true;
                } else {
                    candidateRefined = partial_affine_utils::refineRigidFromMask(
                        pts1, pts2, _ransacReprojThreshold, candidateMask, candidateA, false);
                }
                if (!candidateRefined || candidateA.empty()) {
                    continue;
                }

                // 候选经过 rigid refine 后再入池，确保和 baseline 比较时都是“最终可用模型”。
                candidateTransforms.push_back(candidateA.clone());
                candidateMasks.push_back(candidateMask);
                ++generatedCandidates;
            }
        }

        cv::Mat selectedA;
        std::vector<unsigned char> selectedMask;
        int selectedInliers = 0;
        double selectedError = std::numeric_limits<double>::infinity();
        double selectedContainment = -1.0;
        double selectedBidirectionalCoverage = -1.0;

        // 最终统一按“前景几何门槛优先，再比较内点数 / coverage / containment / 误差”
        // 选择最优 rigid。若所有候选都没过前景门槛，则自动回退到旧规则，避免直接清空结果。
        if (rigid_estimator_helpers::selectBestRigidCandidate(candidateTransforms,
                                                              candidateMasks,
                                                              ctx,
                                                              pts1,
                                                              pts2,
                                                              _minInliers,
                                                              _enableCandidateMaskScoring,
                                                              _candidateMaskForegroundThreshold,
                                                              _candidateMinContainment,
                                                              _candidateMinBidirectionalCoverage,
                                                              _candidateDedupRotationDiffDeg,
                                                              _candidateDedupTranslationDiff,
                                                              selectedA,
                                                              selectedMask,
                                                              selectedInliers,
                                                              selectedError,
                                                              selectedContainment,
                                                              selectedBidirectionalCoverage)) {
            A = selectedA;
            mask = selectedMask;
            IR_LOG_INFO("RigidEstimator filtered-match candidates generated=",
                        generatedCandidates,
                        ", seed_count=",
                        topK,
                        ", total_candidates=",
                        candidateTransforms.size(),
                        ", selected_inliers=",
                        selectedInliers,
                        ", selected_error=",
                        selectedError,
                        ", selected_containment=",
                        selectedContainment,
                        ", selected_bidirectional_coverage=",
                        selectedBidirectionalCoverage);
        } else {
            IR_LOG_INFO("RigidEstimator filtered-match candidates generated=0 or no candidate survived scoring.");
        }
    }

    if (A.empty()) {
        gd.message = "rigid estimator produced an empty matrix";
        IR_LOG_ERROR("RigidEstimator: rigid estimator produced an empty matrix.");
        return false;
    }

    if (_estimatorBackend == "CUSTOM_RIGID_RANSAC") {
        IR_LOG_INFO("RigidEstimator custom backend final inliers=",
                    partial_affine_utils::countInliers(mask),
                    " / ",
                    view.filtered.size());
    }

    // 8. 将最终 mask 提升回上下文中的通用几何结果和点特征内点列表。
    partial_affine_utils::promoteInliers(ctx, view, mask);
    const int inliers = partial_affine_utils::countInliers(mask);

    // 9. 回写最终模型、内点数和内点率，并按 minInliers 判断当前 rigid 结果是否有效。
    gd.A = A;
    gd.num_inliers = inliers;
    gd.inlier_ratio = view.filtered.empty() ? 0.0 : static_cast<double>(inliers) / view.filtered.size();
    gd.valid = inliers >= _minInliers;
    if (!gd.valid) {
        gd.message = partial_affine_utils::rejectMessage("rigid transform", inliers, _minInliers);
        IR_LOG_WARN("RigidEstimator rejected model: ", gd.message);
    }

    // 10. 输出最终 rigid 内点统计，和前面的 RANSAC 初筛形成闭环。
    IR_LOG_INFO("Rigid2D inliers=",
                inliers,
                " / ",
                view.filtered.size(),
                " (ratio=",
                gd.inlier_ratio,
                ", source=",
                view.source_name,
                ")");
    return gd.valid;
}

} // namespace ir
