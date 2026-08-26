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
    const std::string candidatePairStrategyRaw =
        yaml_utils::getString(params, "filteredMatchCandidatePairStrategy", "COVERAGE_FIRST");
    _filteredMatchCandidatePairStrategy =
        rigid_estimator_helpers::normalizeCandidatePairStrategy(candidatePairStrategyRaw);
    const std::string normalizedCandidatePairStrategyRaw =
        string_utils::normalizedKey(candidatePairStrategyRaw);
    if (normalizedCandidatePairStrategyRaw != "COVERAGEFIRST" &&
        normalizedCandidatePairStrategyRaw != "LEGACYLEXICOGRAPHIC") {
        IR_LOG_WARN("RigidEstimator: unknown filteredMatchCandidatePairStrategy=",
                    candidatePairStrategyRaw,
                    ", fallback to COVERAGE_FIRST.");
    }
    _enableCandidateMaskScoring =
        yaml_utils::getBool(params, "enableCandidateMaskScoring", false);
    _candidateMaskForegroundThreshold =
        std::clamp(yaml_utils::getInt(params, "candidateMaskForegroundThreshold", 10), 0, 255);
    _candidateDedupRotationDiffDeg =
        std::max(0.0, yaml_utils::getDouble(params, "candidateDedupRotationDiffDeg", 2.0));
    _candidateDedupTranslationDiff =
        std::max(0.0, yaml_utils::getDouble(params, "candidateDedupTranslationDiff", 3.0));
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
                ", filteredMatchCandidatePairStrategy=",
                _filteredMatchCandidatePairStrategy,
                ", enableCandidateMaskScoring=",
                _enableCandidateMaskScoring,

                ", candidateMaskForegroundThreshold=",
                _candidateMaskForegroundThreshold,
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
    const CorrespondenceView view = ensureCorrespondenceView(ctx);

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
    candidateTransforms.reserve(static_cast<size_t>(1 + _filteredMatchCandidateCount));
    candidateMasks.reserve(static_cast<size_t>(1 + _filteredMatchCandidateCount));

    // 两种后端共用 baseline 统计和有效性判断。
    const auto updateBaselineDiagnostics = [&]() {
        const int baselineInliers =
            refined && !A.empty() ? partial_affine_utils::countInliers(mask) : 0;
        const double baselineMeanReprojectionError =
            baselineInliers > 0
                ? partial_affine_utils::reprojectionErrorSum(pts1, pts2, mask, A) /
                      static_cast<double>(baselineInliers)
                : -1.0;
        const bool baselineFailed =
            !refined || A.empty() || baselineInliers < _minInliers;
        gd.baseline_valid = !baselineFailed;
        gd.baseline_num_inliers = baselineInliers;
        gd.baseline_mean_reproj_error = baselineMeanReprojectionError;

        IR_LOG_INFO("RigidEstimator baseline: valid=",
                    gd.baseline_valid,
                    ", inliers=",
                    gd.baseline_num_inliers,
                    ", mean_reproj_error=",
                    gd.baseline_mean_reproj_error,
                    ", backend=",
                    _estimatorBackend);
    };

    // 5. 按 YAML 选择初始估计后端：
    //    - OPENCV_PARTIAL_AFFINE：直接使用 estimateAffinePartial2D 的结果。
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
        IR_LOG_INFO("RigidEstimator custom rigid RANSAC inliers=",
                    seed_inliers,
                    " / ",
                    view.filtered.size(),
                    " (mask_size=",
                    mask.size(),
                    ", source=",
                    view.source_name,
                    ")");
        updateBaselineDiagnostics();
    } else {
        cv::Mat seedA = cv::estimateAffinePartial2D(pts1,
                                                    pts2,
                                                    mask,
                                                    _method,
                                                    _ransacReprojThreshold,
                                                    static_cast<size_t>(_maxIters),
                                                    _confidence,
                                                    static_cast<size_t>(_refineIters));

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

        refined = !A.empty();
        updateBaselineDiagnostics();
        if (refined) {
            candidateTransforms.push_back(A.clone());
            candidateMasks.push_back(mask);
        } else {
            gd.message = "estimateAffinePartial2D returned an empty matrix";
            IR_LOG_WARN("estimateAffinePartial2D returned an empty matrix; trying filtered-match candidates.");
        }
        // 7. OpenCV 后端可基于 filtered matches 生成额外候选。
        // 候选总开关开启且前置条件满足时，始终执行，不受 baseline 成败影响。
        const bool candidatePrerequisitesMet =
            _enableFilteredMatchCandidates &&
            static_cast<int>(pts1.size()) >= 2 &&
            !filteredDistances.empty() &&
            _filteredMatchCandidateCount > 0;
        const bool canTryFilteredMatchCandidates = candidatePrerequisitesMet;
        gd.candidate_fallback_attempted = canTryFilteredMatchCandidates;
        gd.candidate_fallback_trigger_reason =
            canTryFilteredMatchCandidates ? "enabled" : "not_available";

        // 8. baseline 失败时，只有满足候选条件才允许继续由额外候选接管。
        if (!refined) {
            gd.message = "failed to refine rigid transform from RANSAC inliers";
            IR_LOG_DEBUG("RigidEstimator: baseline refine failed, mode=",
                        _estimatorBackend,
                        ", strict rigid refinement",
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

        // 9. 从已经经过 filter 的匹配里继续抽样并统一评分候选。
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
        const std::vector<rigid_estimator_helpers::CandidateSeedPair> candidateSeedPairs =
            rigid_estimator_helpers::buildCandidateSeedPairs(seedIndices,
                                                              pts1,
                                                              pts2,
                                                              filteredDistances,
                                                              _filteredMatchCandidateMinPairDistance);
        std::vector<unsigned char> attemptedPairs(candidateSeedPairs.size(), 0);
        std::vector<int> successfulSeedUsage(pts1.size(), 0);
        int generatedCandidates = 0;
        int attemptedPairCount = 0;
        while (generatedCandidates < _filteredMatchCandidateCount) {
            const int pairIndex = rigid_estimator_helpers::selectNextCandidateSeedPair(
                candidateSeedPairs,
                attemptedPairs,
                successfulSeedUsage,
                _filteredMatchCandidatePairStrategy);
            if (pairIndex < 0) {
                break;
            }
            attemptedPairs[static_cast<size_t>(pairIndex)] = 1;
            ++attemptedPairCount;
            const auto& seedPair = candidateSeedPairs[static_cast<size_t>(pairIndex)];
            const int first = seedPair.first;
            const int second = seedPair.second;

            cv::Mat candidateA;
            std::vector<unsigned char> candidateMask;

            // 先用 2 对点生成一个最小刚体假设，再投影回全部 filtered 点上拿到初始 mask。
            if (!rigid_estimator_helpers::buildRigidCandidateFromPair(
                    pts1, pts2, first, second, _ransacReprojThreshold, candidateA, candidateMask) ||
                partial_affine_utils::countInliers(candidateMask) < _minInliers) {
                continue;
            }

            const bool candidateRefined = partial_affine_utils::refineRigidFromMask(
                pts1, pts2, _ransacReprojThreshold, candidateMask, candidateA, false);
            if (!candidateRefined || candidateA.empty()) {
                continue;
            }

            // 只有成功建模并精修的候选才消耗 seed 覆盖名额；失败点对会继续尝试其他组合。
            ++successfulSeedUsage[static_cast<size_t>(first)];
            ++successfulSeedUsage[static_cast<size_t>(second)];
            candidateTransforms.push_back(candidateA.clone());
            candidateMasks.push_back(candidateMask);
            ++generatedCandidates;
        }
        cv::Mat selectedA;
        std::vector<unsigned char> selectedMask;
        int selectedInliers = 0;
        double selectedError = std::numeric_limits<double>::infinity();
        double selectedContainment = -1.0;
        std::vector<cv::Mat> validCandidateTransforms;

        // 先按内点数选出最一致的模型，再用平均重投影误差打平，
        // 与 CUSTOM_RIGID_RANSAC 保持相同的主排序标准。
        if (rigid_estimator_helpers::selectBestRigidCandidate(candidateTransforms,
                                                              candidateMasks,
                                                              ctx,
                                                              pts1,
                                                              pts2,
                                                              _minInliers,
                                                              _enableCandidateMaskScoring,
                                                              _candidateMaskForegroundThreshold,
                                                              _candidateDedupRotationDiffDeg,
                                                              _candidateDedupTranslationDiff,
                                                              selectedA,
                                                               selectedMask,
                                                               selectedInliers,
                                                               selectedError,
                                                               selectedContainment,
                                                               validCandidateTransforms)) {
            A = selectedA;
            gd.candidate_transforms = std::move(validCandidateTransforms);
            mask = selectedMask;
            IR_LOG_INFO("RigidEstimator filtered-match candidates generated=",
                        generatedCandidates,
                        ", seed_count=",
                        topK,
                        ", legal_pair_count=",
                        candidateSeedPairs.size(),
                        ", attempted_pair_count=",
                        attemptedPairCount,
                        ", pair_strategy=",
                        _filteredMatchCandidatePairStrategy,
                        ", candidate_pool_size_including_baseline=",
                        candidateTransforms.size(),
                        ", selected_inliers=",
                        selectedInliers,
                        ", selected_error=",
                        selectedError,
                        ", selected_containment=",
                        selectedContainment);
        } else {
            IR_LOG_INFO("RigidEstimator filtered-match candidates generated=0 or no candidate survived scoring, legal_pair_count=",
                        candidateSeedPairs.size(),
                        ", attempted_pair_count=",
                        attemptedPairCount,
                        ", pair_strategy=",
                        _filteredMatchCandidatePairStrategy);
        }
        }
    }

    // 10. 候选选择后必须得到非空刚体矩阵。
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

    // 11. 将最终 mask 提升回上下文中的通用几何结果和点特征内点列表。
    partial_affine_utils::promoteInliers(ctx, view, mask);
    const int inliers = partial_affine_utils::countInliers(mask);

    // 12. 回写最终模型、内点数和内点率，并按 minInliers 判断当前 rigid 结果是否有效。
    gd.A = A;
    gd.num_inliers = inliers;
    gd.inlier_ratio = view.filtered.empty() ? 0.0 : static_cast<double>(inliers) / view.filtered.size();
    gd.valid = inliers >= _minInliers;
    if (!gd.valid) {
        gd.message = partial_affine_utils::rejectMessage("rigid transform", inliers, _minInliers);
        IR_LOG_WARN("RigidEstimator rejected model: ", gd.message);
    }

    // 13. 输出最终 rigid 内点统计，和前面的 RANSAC 初筛形成闭环。
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
