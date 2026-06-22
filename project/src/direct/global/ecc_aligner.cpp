#include "direct/global/ecc_aligner.h"

#include <string>

#include "direct/common/ecc_common.h"
#include "core/types.h"
#include "utils/image_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

EccAligner::EccAligner(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    // 兼容算法配置文件中直接写参数或写在 params 节点下两种形式。
    _motionModel = yaml_utils::getString(params, "motion_model", "RIGID");
    _maxIterations = yaml_utils::getInt(params, "max_iterations", 100);
    _epsilon = yaml_utils::getDouble(params, "epsilon", 1e-6);
    _gaussianFilterSize = yaml_utils::getInt(params, "gaussian_filter_size", 5);
}

bool EccAligner::align(RegistrationContext& ctx) {
    auto& dd = ctx.direct_data;
    auto& gd = ctx.geometry_data;
    dd.clear();
    gd.clear();
    dd.method = name();

    // 1. 读取并归一化灰度图；ECC 当前要求源图和目标图尺寸一致。
    cv::Mat src;
    cv::Mat dst;
    if (!image_utils::convertGrayToFloat01(ctx.images.first_gray, src) ||
        !image_utils::convertGrayToFloat01(ctx.images.second_gray, dst)) {
        dd.message = "ECC requires non-empty grayscale images";
        gd.message = dd.message;
        return false;
    }
    if (src.size() != dst.size()) {
        dd.message = "ECC requires images with the same size";
        gd.message = dd.message;
        return false;
    }

    // 2. 根据运动模型初始化 OpenCV ECC 所需的 warp 矩阵和终止条件。
    const int motionType = ecc_common::motionTypeFromString(_motionModel);
    // OpenCV ECC 的 HOMOGRAPHY 使用 3x3 初始化，其它模型使用 2x3 初始化。
    cv::Mat warp =
        motionType == cv::MOTION_HOMOGRAPHY ? cv::Mat::eye(3, 3, CV_32F)
                                            : cv::Mat::eye(2, 3, CV_32F);
    if (ecc_common::initialWarpFromFeatureInitializer(ctx, motionType, warp)) {
        dd.addDiagnostic("feature_initializer_ecc_seeded",
                         "ECC seeded by feature init",
                         1.0);
        IR_LOG_INFO("ECC initialized from feature initializer: ",
                    ctx.feature_initializer_data.method);
    }
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                    std::max(1, _maxIterations),
                                    _epsilon);

    // 3. 调用 findTransformECC 优化 ECC 矩阵。
    double score = 0.0;
    try {
        // OpenCV ECC 的 warp 矩阵按 WARP_INVERSE_MAP 语义使用：template 像素到 input 像素。
        // 平台统一约定 gd.A/gd.H 是 source -> target，因此写回前需要取逆。
        score = cv::findTransformECC(dst,
                                     src,
                                     warp,
                                     motionType,
                                     criteria,
                                     cv::noArray(),
                                     _gaussianFilterSize);
    } catch (const cv::Exception& e) {
        dd.message = std::string("findTransformECC failed: ") + e.what();
        gd.message = dd.message;
        IR_LOG_WARN("ECC rejected: ", dd.message);
        return false;
    }

    if (!std::isfinite(score)) {
        dd.message = "findTransformECC returned non-finite score";
        gd.message = dd.message;
        return false;
    }

    // 4. 将 ECC 优化得到的矩阵写回统一几何输出，供后续 warp/评估复用。
    gd.type = ecc_common::geometryTypeFromMotionType(motionType);
    gd.valid = true;
    gd.num_inliers = 1;
    gd.inlier_ratio = score;
    dd.valid = true;
    dd.score = score;

    // ECC 不产生点对；把 OpenCV 返回矩阵转成 source -> target 后写入 geometry_data。
    if (gd.type == GeometryType::HOMOGRAPHY) {
        cv::Mat warp64;
        warp.convertTo(warp64, CV_64F);
        if (!cv::invert(warp64, gd.H)) {
            dd.message = "ECC homography matrix is not invertible";
            gd.message = dd.message;
            gd.valid = false;
            dd.valid = false;
            return false;
        }
        dd.H = gd.H.clone();
    } else {
        cv::Mat warp64;
        warp.convertTo(warp64, CV_64F);
        cv::invertAffineTransform(warp64, gd.A);
        dd.A = gd.A.clone();
    }

    IR_LOG_INFO("ECC estimated ", toString(gd.type), " score=", score);
    return true;
}

} // namespace ir
