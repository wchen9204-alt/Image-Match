#include "structure/edge_extractor.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

int normalizeAperture(int value) {
    if (value != 3 && value != 5 && value != 7) {
        return 3;
    }
    return value;
}

void runCanny(const cv::Mat& gray,
              cv::Mat& edges,
              double threshold1,
              double threshold2,
              int apertureSize,
              bool l2Gradient,
              int blurKernel,
              int dilateIterations) {
    cv::Mat input = gray;
    cv::Mat blurred;
    if (blurKernel >= 3) {
        if (blurKernel % 2 == 0) {
            ++blurKernel;
        }
        cv::GaussianBlur(gray, blurred, cv::Size(blurKernel, blurKernel), 0.0);
        input = blurred;
    }

    cv::Canny(input, edges, threshold1, threshold2, apertureSize, l2Gradient);
    if (dilateIterations > 0) {
        cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), dilateIterations);
    }
}

} // namespace

EdgeExtractor::EdgeExtractor(const YAML::Node& cfg) {
    const auto params = cfg["params"];
    _threshold1 = yaml_utils::getDouble(params, "threshold1", 50.0);
    _threshold2 = yaml_utils::getDouble(params, "threshold2", 150.0);
    _apertureSize = normalizeAperture(yaml_utils::getInt(params, "apertureSize", 3));
    _l2Gradient = yaml_utils::getBool(params, "l2Gradient", false);
    _blurKernel = yaml_utils::getInt(params, "blurKernel", 0);
    _dilateIterations = yaml_utils::getInt(params, "dilateIterations", 0);

    IR_LOG_INFO("EdgeExtractor: threshold1=",
                _threshold1,
                ", threshold2=",
                _threshold2,
                ", apertureSize=",
                _apertureSize,
                ", blurKernel=",
                _blurKernel,
                ", dilateIterations=",
                _dilateIterations);
}

bool EdgeExtractor::extract(RegistrationContext& ctx) {
    auto& sd = ctx.structure_data;
    const auto& fd = ctx.feature_data;

    if (fd.first.gray.empty() || fd.second.gray.empty()) {
        IR_LOG_ERROR("EdgeExtractor: input grayscale images are empty.");
        return false;
    }

    sd.clear();
    sd.type = StructureType::EDGE;
    runCanny(fd.first.gray,
             sd.first.mask,
             _threshold1,
             _threshold2,
             _apertureSize,
             _l2Gradient,
             _blurKernel,
             _dilateIterations);
    runCanny(fd.second.gray,
             sd.second.mask,
             _threshold1,
             _threshold2,
             _apertureSize,
             _l2Gradient,
             _blurKernel,
             _dilateIterations);

    const int n1 = sd.first.primitiveCount(StructureType::EDGE);
    const int n2 = sd.second.primitiveCount(StructureType::EDGE);
    IR_LOG_INFO("EdgeExtractor extracted edge pixels: ", n1, " / ", n2);
    return n1 > 0 && n2 > 0;
}

} // namespace ir
