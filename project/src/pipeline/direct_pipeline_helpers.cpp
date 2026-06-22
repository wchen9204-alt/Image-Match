#include "pipeline/direct_pipeline_helpers.h"

#include <filesystem>
#include <string>

#include <opencv2/imgproc.hpp>

#include "pipeline/base_pipeline_helpers.h"
#include "utils/logger.h"

namespace fs = std::filesystem;

namespace ir::direct_pipeline_helpers {

namespace {

/// 从点特征初始值结果里提取当前可用于 warp 的矩阵。
/// rigid / similarity / affine 统一走 2x3，homography 走 3x3。
bool matrixFromFeatureInitializer(const FeatureInitializerData& init, cv::Mat& matrix) {
    matrix.release();
    if ((init.type == GeometryType::RIGID || init.type == GeometryType::SIMILARITY ||
         init.type == GeometryType::AFFINE) &&
        !init.A.empty()) {
        init.A.convertTo(matrix, CV_64F);
        return matrix.rows >= 2 && matrix.cols >= 3;
    }
    if (init.type == GeometryType::HOMOGRAPHY && !init.H.empty()) {
        init.H.convertTo(matrix, CV_64F);
        return matrix.rows >= 3 && matrix.cols >= 3;
    }
    return false;
}

/// 从直接法当前几何结果里提取残差矩阵。
/// 这里优先遵循 geometry.type，再兼容少量“矩阵已写入但 type 未细分”的情况。
bool matrixFromGeometry(const GeometryData& geometry, cv::Mat& matrix) {
    matrix.release();
    if (geometry.type == GeometryType::HOMOGRAPHY && !geometry.H.empty()) {
        geometry.H.convertTo(matrix, CV_64F);
        return matrix.rows >= 3 && matrix.cols >= 3;
    }
    if (!geometry.A.empty()) {
        geometry.A.convertTo(matrix, CV_64F);
        return matrix.rows >= 2 && matrix.cols >= 3;
    }
    if (!geometry.H.empty()) {
        geometry.H.convertTo(matrix, CV_64F);
        return matrix.rows >= 3 && matrix.cols >= 3;
    }
    return false;
}

/// 把 2x3 仿射族矩阵补成 3x3 齐次矩阵，便于后续统一做矩阵复合。
cv::Mat toHomogeneousMatrix(const cv::Mat& matrix) {
    if (matrix.rows == 3 && matrix.cols == 3) {
        return matrix.clone();
    }
    if (matrix.rows >= 2 && matrix.cols >= 3) {
        cv::Mat h = cv::Mat::eye(3, 3, CV_64F);
        matrix(cv::Rect(0, 0, 3, 2)).copyTo(h(cv::Rect(0, 0, 3, 2)));
        return h;
    }
    return {};
}

/// 给几种几何模型一个复杂度顺序，便于合并点特征初值和直接法残差后的最终类型判定。
int geometryComplexityRank(GeometryType type) {
    switch (type) {
    case GeometryType::RIGID: return 1;
    case GeometryType::SIMILARITY: return 2;
    case GeometryType::AFFINE: return 3;
    case GeometryType::HOMOGRAPHY: return 4;
    default: return 0;
    }
}

/// 两段几何结果合并时，保留表达能力更强的那个模型类型。
GeometryType mergedGeometryType(GeometryType first, GeometryType second) {
    return geometryComplexityRank(first) >= geometryComplexityRank(second) ? first : second;
}

/// 用 source -> target 矩阵把原始 source 图和灰度图一起 warp 到 target 画布上。
/// 这样后续直接法就可以在“已经粗对齐”的图上只估计剩余残差。
bool warpSourceByMatrix(const RegistrationContext& ctx,
                        const cv::Mat& sourceToTarget,
                        cv::Mat& warpedColor,
                        cv::Mat& warpedGray) {
    warpedColor.release();
    warpedGray.release();
    if (ctx.images.first.empty() || ctx.images.first_gray.empty() || ctx.images.second.empty()) {
        return false;
    }

    if (sourceToTarget.rows == 3 && sourceToTarget.cols == 3) {
        cv::warpPerspective(ctx.images.first,
                            warpedColor,
                            sourceToTarget,
                            ctx.images.second.size(),
                            cv::INTER_LINEAR,
                            cv::BORDER_CONSTANT,
                            cv::Scalar(0));
        cv::warpPerspective(ctx.images.first_gray,
                            warpedGray,
                            sourceToTarget,
                            ctx.images.second.size(),
                            cv::INTER_LINEAR,
                            cv::BORDER_CONSTANT,
                            cv::Scalar(0));
        return !warpedColor.empty() && !warpedGray.empty();
    }

    cv::warpAffine(ctx.images.first,
                   warpedColor,
                   sourceToTarget,
                   ctx.images.second.size(),
                   cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT,
                   cv::Scalar(0));
    cv::warpAffine(ctx.images.first_gray,
                   warpedGray,
                   sourceToTarget,
                   ctx.images.second.size(),
                   cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT,
                   cv::Scalar(0));
    return !warpedColor.empty() && !warpedGray.empty();
}

/// 把“初始值 source -> target”和“残差 source' -> target”复合成最终 source -> target。
/// 这里 residualH * initializerH 的顺序不能反，否则语义会错。
bool composeSourceToTargetTransforms(const cv::Mat& residualSourceToTarget,
                                     const cv::Mat& initializerSourceToTarget,
                                     cv::Mat& composed) {
    composed.release();
    const cv::Mat residualH = toHomogeneousMatrix(residualSourceToTarget);
    const cv::Mat initializerH = toHomogeneousMatrix(initializerSourceToTarget);
    if (residualH.empty() || initializerH.empty()) {
        return false;
    }

    composed = residualH * initializerH;
    return composed.rows == 3 && composed.cols == 3 && cv::checkRange(composed, true);
}

/// 通用 prewarp 会把直接法内部的 source 点坐标带到预 warp 后的坐标系里。
/// 输出最终结果前，需要把这些点映射回原始 source 坐标系，避免后续评估和可视化读错点位。
bool remapPrewarpedSourcePointsToOriginal(const cv::Mat& initializerSourceToTarget,
                                          std::vector<cv::Point2f>& sourcePoints) {
    if (sourcePoints.empty()) {
        return true;
    }

    cv::Mat inverseMatrix;
    if (!base_pipeline_helpers::invertTransformMatrix(initializerSourceToTarget, inverseMatrix)) {
        return false;
    }

    if (inverseMatrix.rows == 3 && inverseMatrix.cols == 3) {
        cv::perspectiveTransform(sourcePoints, sourcePoints, inverseMatrix);
        return true;
    }

    cv::transform(sourcePoints, sourcePoints, inverseMatrix);
    return true;
}

} // namespace

/// 这两种直接法会在算法内部自行读取并使用点特征初始值，pipeline 不再额外做通用 prewarp。
bool alignerConsumesFeatureInitializerInternally(const std::string& alignerName) {
    return alignerName == "ECC" || alignerName == "ESM_RIGID";
}

/// 每次输出前先清掉上一轮残留图，避免本轮未生成时误读旧结果。
void removeStaleDirectVisualization(const fs::path& path) {
    std::error_code ec;
    if (fs::exists(path, ec)) {
        fs::remove(path, ec);
        if (ec) {
            IR_LOG_WARN("Failed to remove stale direct visualization: ", path.string());
        } else {
            IR_LOG_INFO("Removed stale direct visualization: ", path.string());
        }
    }
}

/// 把点特征初始值阶段的诊断字段写回最终 result，供摘要、CSV 和 JSON 统一复用。
void syncFeatureInitializerDiagnostics(RegistrationContext& ctx) {
    const auto& init = ctx.feature_initializer_data;
    auto& result = ctx.result;

    result.feature_initializer_attempted = init.attempted;
    result.feature_initializer_method = init.method;
    result.feature_initializer_inliers = init.num_inliers;
    result.feature_initializer_inlier_ratio = init.inlier_ratio;
    result.feature_initializer_spatial_coverage = init.inlier_spatial_coverage;
    result.feature_initializer_warp_photometric_error = init.warp_photometric_error;
    result.feature_initializer_warp_edge_alignment_iou = init.warp_edge_alignment_iou;
}

/// 对尚未内建“初值消费”逻辑的直接法，通用地完成一次 source 预 warp。
/// 若初始值未被接受，或者该直接法会自己消费初始值，则这里什么都不做并返回 true。
bool applyFeatureInitializerPrewarp(RegistrationContext& ctx,
                                    const std::string& alignerName,
                                    cv::Mat& initializerMatrix,
                                    cv::Mat& originalColor,
                                    cv::Mat& originalGray,
                                    bool& applied) {
    applied = false;
    initializerMatrix.release();
    originalColor.release();
    originalGray.release();

    if (!ctx.feature_initializer_data.accepted ||
        alignerConsumesFeatureInitializerInternally(alignerName)) {
        return true;
    }

    if (!matrixFromFeatureInitializer(ctx.feature_initializer_data, initializerMatrix)) {
        return false;
    }

    cv::Mat warpedColor;
    cv::Mat warpedGray;
    if (!warpSourceByMatrix(ctx, initializerMatrix, warpedColor, warpedGray)) {
        return false;
    }

    originalColor = ctx.images.first.clone();
    originalGray = ctx.images.first_gray.clone();
    ctx.images.first = std::move(warpedColor);
    ctx.images.first_gray = std::move(warpedGray);
    applied = true;
    return true;
}

/// 将直接法在预 warp 图上估计出的残差，与点特征法给出的粗配准结果合成为最终结果。
/// 合成后同时回写 geometry_data 和 direct_data，保持后续验证、输出读取一致。
bool mergeFeatureInitializerAndDirectResult(RegistrationContext& ctx,
                                            const cv::Mat& initializerMatrix) {
    cv::Mat residualMatrix;
    if (!matrixFromGeometry(ctx.geometry_data, residualMatrix)) {
        return false;
    }

    cv::Mat composed;
    if (!composeSourceToTargetTransforms(residualMatrix, initializerMatrix, composed)) {
        return false;
    }

    const GeometryType mergedType =
        mergedGeometryType(ctx.geometry_data.type, ctx.feature_initializer_data.type);
    ctx.geometry_data.type = mergedType;
    ctx.geometry_data.valid = true;
    ctx.direct_data.valid = true;

    if (mergedType == GeometryType::HOMOGRAPHY) {
        ctx.geometry_data.H = composed.clone();
        ctx.geometry_data.A.release();
        ctx.direct_data.H = ctx.geometry_data.H.clone();
        ctx.direct_data.A.release();
    } else {
        ctx.geometry_data.A = composed(cv::Rect(0, 0, 3, 2)).clone();
        ctx.geometry_data.H.release();
        ctx.direct_data.A = ctx.geometry_data.A.clone();
        ctx.direct_data.H.release();
    }

    if (!remapPrewarpedSourcePointsToOriginal(initializerMatrix, ctx.direct_data.points1)) {
        return false;
    }

    ctx.direct_data.addDiagnostic("feature_initializer_prewarp_applied",
                                  "feature initializer prewarp applied",
                                  1.0);
    return true;
}

/// 单独根据点特征初始值生成一张 warp 后的 source 图，用来输出 initializer false-color overlay。
bool buildInitializerWarpedSource(const RegistrationContext& ctx, cv::Mat& warped) {
    warped.release();
    const auto& init = ctx.feature_initializer_data;
    if (!init.accepted || ctx.images.first.empty() || ctx.images.second.empty()) {
        return false;
    }

    if ((init.type == GeometryType::RIGID || init.type == GeometryType::SIMILARITY ||
         init.type == GeometryType::AFFINE) &&
        !init.A.empty()) {
        cv::warpAffine(ctx.images.first,
                       warped,
                       init.A,
                       ctx.images.second.size(),
                       cv::INTER_LINEAR,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(0));
        return !warped.empty();
    }

    if (init.type == GeometryType::HOMOGRAPHY && !init.H.empty()) {
        cv::warpPerspective(ctx.images.first,
                            warped,
                            init.H,
                            ctx.images.second.size(),
                            cv::INTER_LINEAR,
                            cv::BORDER_CONSTANT,
                            cv::Scalar(0));
        return !warped.empty();
    }

    return false;
}

/// 按最终判定来源在 initializer overlay 和 direct overlay 之间二选一，生成最终展示图。
bool buildFinalSelectedFalseColorOverlay(const RegistrationContext& ctx,
                                         int foregroundThreshold,
                                         cv::Mat& overlay) {
    overlay.release();
    cv::Mat selectedWarped;

    // 1. 若最终结果采用 initializer，则单独重建 initializer 的 warped source。
    if (ctx.result.final_validation_source == "INITIALIZER") {
        if (!buildInitializerWarpedSource(ctx, selectedWarped)) {
            return false;
        }
    } else {
        // 2. 其它情况默认沿用 direct 最终 warp 结果。
        if (ctx.warped_image.empty()) {
            return false;
        }
        selectedWarped = ctx.warped_image;
    }

    // 3. 使用统一的 false-color overlay 逻辑，保证 direct / initializer 的显示口径一致。
    return base_pipeline_helpers::buildFalseColorOverlay(selectedWarped,
                                                         ctx.images.second,
                                                         foregroundThreshold,
                                                         overlay);
}

} // namespace ir::direct_pipeline_helpers
