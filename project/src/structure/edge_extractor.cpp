#include "structure/edge_extractor.h"

#include <algorithm>
#include <string>

#include <opencv2/imgproc.hpp>

#include "utils/image_utils.h"
#include "utils/logger.h"
#include "utils/string_utils.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

EdgeOperatorType edgeOperatorFromString(const std::string& raw) {
    const std::string method = string_utils::toUpperAscii(raw);
    if (method == "SOBEL") {
        return EdgeOperatorType::SOBEL;
    }
    if (method == "LOG" || method == "LAPLACIAN_OF_GAUSSIAN") {
        return EdgeOperatorType::LOG;
    }
    if (method == "LAPLACIAN") {
        return EdgeOperatorType::LAPLACIAN;
    }
    return EdgeOperatorType::CANNY;
}

const char* toString(EdgeOperatorType type) {
    switch (type) {
    case EdgeOperatorType::SOBEL:
        return "SOBEL";
    case EdgeOperatorType::LOG:
        return "LOG";
    case EdgeOperatorType::LAPLACIAN:
        return "LAPLACIAN";
    case EdgeOperatorType::CANNY:
    default:
        return "CANNY";
    }
}

// 对输入图像应用可选的高斯模糊以减少噪声对边缘检测的影响。
void applyOptionalBlur(const cv::Mat& gray, cv::Mat& out, int blurKernel) {
    image_utils::applyOptionalGaussianBlur(gray, out, blurKernel);
}

// 对边缘强度图像应用阈值。
void thresholdMagnitude(const cv::Mat& magnitude, cv::Mat& response, double threshold) {
    cv::Mat normalized;
    cv::normalize(magnitude, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    if (threshold > 0.0) {
        cv::threshold(normalized, response, threshold, 255, cv::THRESH_BINARY);
    } else {
        response = normalized;
    }
}

void runCanny(const cv::Mat& gray,
              cv::Mat& response,
              double threshold1,
              double threshold2,
              int apertureSize,
              bool l2Gradient,
              int blurKernel) {
    cv::Mat input;
    applyOptionalBlur(gray, input, blurKernel);
    cv::Canny(input, response, threshold1, threshold2, apertureSize, l2Gradient);
}

void runSobel(const cv::Mat& gray,
              cv::Mat& response,
              int kernelSize,
              double scale,
              double delta,
              double threshold,
              int blurKernel) {
    cv::Mat input;
    applyOptionalBlur(gray, input, blurKernel);

    cv::Mat gradX;
    cv::Mat gradY;
    cv::Sobel(input, gradX, CV_32F, 1, 0, kernelSize, scale, delta);
    cv::Sobel(input, gradY, CV_32F, 0, 1, kernelSize, scale, delta);

    cv::Mat magnitude;
    cv::magnitude(gradX, gradY, magnitude);
    thresholdMagnitude(magnitude, response, threshold);
}

void runLaplacian(const cv::Mat& gray,
                  cv::Mat& response,
                  int kernelSize,
                  double scale,
                  double delta,
                  double threshold,
                  int blurKernel) {
    cv::Mat input;
    applyOptionalBlur(gray, input, blurKernel);

    cv::Mat laplacian;
    cv::Laplacian(input, laplacian, CV_32F, kernelSize, scale, delta);
    cv::Mat magnitude = cv::abs(laplacian);
    thresholdMagnitude(magnitude, response, threshold);
}

void applyDilation(cv::Mat& response, int dilateIterations) {
    if (dilateIterations > 0) {
        cv::dilate(response, response, cv::Mat(), cv::Point(-1, -1), dilateIterations);
    }
}

} // namespace

EdgeExtractor::EdgeExtractor(const YAML::Node& cfg) {
    const YAML::Node extractor = cfg["extractor"];
    const YAML::Node params = extractor && extractor["params"] ? extractor["params"] : cfg["params"];
    const std::string method =
        extractor ? yaml_utils::getString(extractor, "method", "CANNY") : "CANNY";

    _operatorType = edgeOperatorFromString(method);
    _threshold1 = yaml_utils::getDouble(params, "threshold1", 50.0);
    _threshold2 = yaml_utils::getDouble(params, "threshold2", 150.0);
    _apertureSize =
        image_utils::normalizedCannyAperture(yaml_utils::getInt(params, "apertureSize", 3));
    _l2Gradient = yaml_utils::getBool(params, "l2Gradient", false);
    _blurKernel = yaml_utils::getInt(params, "blurKernel", 0);
    _dilateIterations = yaml_utils::getInt(params, "dilateIterations", 0);
    _kernelSize =
        image_utils::normalizedOddKernel(yaml_utils::getInt(params, "kernelSize", 3), 3, 1);
    _scale = yaml_utils::getDouble(params, "scale", 1.0);
    _delta = yaml_utils::getDouble(params, "delta", 0.0);
    _responseThreshold = yaml_utils::getDouble(params, "responseThreshold", 50.0);

    IR_LOG_INFO("EdgeExtractor: operator=",
                toString(_operatorType),
                ", threshold1=",
                _threshold1,
                ", threshold2=",
                _threshold2,
                ", apertureSize=",
                _apertureSize,
                ", kernelSize=",
                _kernelSize,
                ", responseThreshold=",
                _responseThreshold,
                ", blurKernel=",
                _blurKernel,
                ", dilateIterations=",
                _dilateIterations);
}

std::string EdgeExtractor::outputLabel() const {
    return std::string("EDGE_") + toString(_operatorType);
}

bool EdgeExtractor::extract(RegistrationContext& ctx) {
    auto& sd = ctx.structure_data;
    const auto& images = ctx.images;

    if (images.first_gray.empty() || images.second_gray.empty()) {
        IR_LOG_ERROR("EdgeExtractor: input grayscale images are empty.");
        return false;
    }

    sd.clear();
    sd.type = StructureType::EDGE;

    auto extractOne = [&](const cv::Mat& gray, cv::Mat& response) {
        switch (_operatorType) {
        case EdgeOperatorType::SOBEL:
            runSobel(
                gray, response, _kernelSize, _scale, _delta, _responseThreshold, _blurKernel);
            break;
        case EdgeOperatorType::LOG:
            runLaplacian(
                gray, response, _kernelSize, _scale, _delta, _responseThreshold, _blurKernel);
            break;
        case EdgeOperatorType::LAPLACIAN:
            runLaplacian(gray, response, _kernelSize, _scale, _delta, _responseThreshold, 0);
            break;
        case EdgeOperatorType::CANNY:
        default:
            runCanny(gray, response, _threshold1, _threshold2, _apertureSize, _l2Gradient, _blurKernel);
            break;
        }
        applyDilation(response, _dilateIterations);
    };

    extractOne(images.first_gray, sd.first.response);
    extractOne(images.second_gray, sd.second.response);

    const int n1 = sd.first.primitiveCount(StructureType::EDGE);
    const int n2 = sd.second.primitiveCount(StructureType::EDGE);
    IR_LOG_INFO("EdgeExtractor extracted edge pixels: ", n1, " / ", n2);
    return n1 > 0 && n2 > 0;
}

} // namespace ir
