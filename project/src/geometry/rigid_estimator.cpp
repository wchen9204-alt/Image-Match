#include "geometry/rigid_estimator.h"

#include <opencv2/calib3d.hpp>

#include <string>
#include <vector>

#include "data/correspondence_view.h"
#include "geometry/partial_affine_utils.h"
#include "utils/logger.h"
#include "utils/string_utils.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

std::string normalizeRigidEstimatorBackend(const std::string& raw) {
    const std::string key = string_utils::normalizedKey(raw);
    if (key == "CUSTOMRIGIDRANSAC" || key == "CUSTOMRANSAC" || key == "RIGIDRANSAC") {
        return "CUSTOM_RIGID_RANSAC";
    }
    return "OPENCV_PARTIAL_AFFINE";
}

} // namespace

RigidEstimator::RigidEstimator(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    _estimatorBackend = normalizeRigidEstimatorBackend(
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
    if (_rigidRefineMode == "NORMALIZE_SCAL") {
        _rigidRefineMode = "NORMALIZE_SCALE";
    }
    if (_rigidRefineMode != "SVD" &&
        _rigidRefineMode != "NORMALIZE_SCALE" &&
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
                _rigidRefineMode);
}

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

    cv::Mat A;
    std::vector<unsigned char> mask;
    bool refined = false;
    int seed_inliers = 0;

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
        } else if (_rigidRefineMode == "NORMALIZE_SCALE") {
            // NORMALIZE_SCALE 从 OpenCV partial affine 的 sR 中去掉统一缩放，
            // 再按当前内点重算平移。
            double removed_scale = 1.0;
            refined = partial_affine_utils::refineRigidByNormalizeScaleFromMask(
                pts1, pts2, _ransacReprojThreshold, mask, A, removed_scale);
            IR_LOG_INFO("RigidEstimator NORMALIZE_SCALE removed_scale=",
                        removed_scale,
                        ", refined=",
                        refined,
                        ", refined_inliers=",
                        refined ? partial_affine_utils::countInliers(mask) : 0,
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
    }

    // 7. refine 失败通常说明当前 RANSAC 内点无法支撑 rigid 约束。
    if (!refined) {
        gd.message = "failed to refine rigid transform from RANSAC inliers";
        IR_LOG_ERROR("RigidEstimator: failed to refine rigid transform from RANSAC inliers, mode=",
                     _estimatorBackend,
                     "/",
                     _rigidRefineMode,
                     ", seed_inliers=",
                     seed_inliers,
                     " / ",
                     view.filtered.size(),
                     ".");
        return false;
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
