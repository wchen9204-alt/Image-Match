#include "feature/sift_extractor.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

SiftExtractor::SiftExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];

    _nfeatures = yaml_utils::getInt(params, "nfeatures", 0);
    _nOctaveLayers = yaml_utils::getInt(params, "nOctaveLayers", 3);
    _contrastThreshold = yaml_utils::getDouble(params, "contrastThreshold", 0.04);
    _edgeThreshold = yaml_utils::getDouble(params, "edgeThreshold", 10.0);
    _sigma = yaml_utils::getDouble(params, "sigma", 1.6);

    _impl =
        cv::SIFT::create(_nfeatures, _nOctaveLayers, _contrastThreshold, _edgeThreshold, _sigma);

    IR_LOG_INFO("SIFT created: nfeatures=",
                _nfeatures,
                ", nOctaveLayers=",
                _nOctaveLayers,
                ", contrast=",
                _contrastThreshold,
                ", edge=",
                _edgeThreshold,
                ", sigma=",
                _sigma);
}

bool SiftExtractor::extract(RegistrationContext& ctx) {
    if (!_impl) {
        IR_LOG_ERROR("SIFT extractor not constructed.");
        return false;
    }

    // SIFT 生成浮点描述子，因此在特征阶段直接写入 L2 距离约定。
    auto& fd = ctx.feature_data;
    fd.type = FeatureType::SIFT;
    fd.norm_type = NormType::L2;

    if (fd.first.image.empty() || fd.second.image.empty()) {
        IR_LOG_ERROR("SIFT::extract - source images are empty.");
        return false;
    }

    // 灰度图优先复用上游预处理结果，避免重复颜色空间转换。
    if (fd.first.gray.empty()) {
        cv::cvtColor(fd.first.image, fd.first.gray, cv::COLOR_BGR2GRAY);
    }
    if (fd.second.gray.empty()) {
        cv::cvtColor(fd.second.image, fd.second.gray, cv::COLOR_BGR2GRAY);
    }

    // 检测与描述子计算合并执行，保证关键点与描述子参数完全一致。
    _impl->detectAndCompute(fd.first.gray, cv::noArray(), fd.first.keypoints, fd.first.descriptors);
    _impl->detectAndCompute(
        fd.second.gray, cv::noArray(), fd.second.keypoints, fd.second.descriptors);

    IR_LOG_INFO("SIFT extracted ",
                fd.first.keypoints.size(),
                " / ",
                fd.second.keypoints.size(),
                " keypoints");
    return !fd.empty();
}

} // namespace ir
