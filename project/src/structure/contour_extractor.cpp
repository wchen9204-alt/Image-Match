#include "structure/contour_extractor.h"

#include <string>

#include "structure/contour_extractor_helpers.h"
#include "utils/image_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

ContourExtractor::ContourExtractor(const YAML::Node& cfg) {
    const YAML::Node extractor = cfg["extractor"];
    const YAML::Node params = extractor && extractor["params"] ? extractor["params"] : cfg["params"];

    // 同时兼容 structure.yaml 的 extractor.params 和旧版直接平铺参数写法。
    _blurKernel = contour_extractor_helpers::normalizedBlurKernel(
        yaml_utils::getInt(params, "blur_kernel", 0));
    _autoCanny = yaml_utils::getBool(params, "auto_canny", false);
    _cannyThreshold1 = yaml_utils::getDouble(params, "cannyThreshold1", 50.0);
    _cannyThreshold2 = yaml_utils::getDouble(params, "cannyThreshold2", 150.0);
    _apertureSize =
        image_utils::normalizedCannyAperture(yaml_utils::getInt(params, "apertureSize", 3));
    _retrievalMode = yaml_utils::getString(params, "retrievalMode", "EXTERNAL");
    _chainApprox = yaml_utils::getString(params, "chainApprox", "SIMPLE");
    _minArea = yaml_utils::getDouble(params, "minArea", 20.0);
    _minPerimeter = yaml_utils::getDouble(params, "minPerimeter", 0.0);
    _minPoints = yaml_utils::getInt(params, "minPoints", 3);
    _minBboxWidth = yaml_utils::getInt(params, "min_bbox_width", 0);
    _minBboxHeight = yaml_utils::getInt(params, "min_bbox_height", 0);
    _minExtent = yaml_utils::getDouble(params, "min_extent", 0.0);
    _maxAspectRatio = yaml_utils::getDouble(params, "max_aspect_ratio", 0.0);
    _maxContours = yaml_utils::getInt(params, "maxContours", 1000);
    _contourThickness = yaml_utils::getInt(params, "contourThickness", 1);

    IR_LOG_INFO("ContourExtractor: blurKernel=",
                _blurKernel,
                ", autoCanny=",
                _autoCanny,
                ", minArea=",
                _minArea,
                ", minPerimeter=",
                _minPerimeter,
                ", minPoints=",
                _minPoints,
                ", minBboxWidth=",
                _minBboxWidth,
                ", minBboxHeight=",
                _minBboxHeight,
                ", minExtent=",
                _minExtent,
                ", maxAspectRatio=",
                _maxAspectRatio,
                ", maxContours=",
                _maxContours,
                ", retrievalMode=",
                _retrievalMode,
                ", chainApprox=",
                _chainApprox,
                ", contourThickness=",
                _contourThickness);
}

bool ContourExtractor::extract(RegistrationContext& ctx) {
    auto& sd = ctx.structure_data;
    const auto& images = ctx.images;

    if (images.first_gray.empty() || images.second_gray.empty()) {
        IR_LOG_ERROR("ContourExtractor: input grayscale images are empty.");
        return false;
    }

    sd.clear();
    sd.type = StructureType::CONTOUR;

    // 两张图共享同一套轮廓提取参数，保证结构统计和匹配前提一致。
    const int retrievalMode =
        contour_extractor_helpers::contourRetrievalModeFromString(_retrievalMode);
    const int approxMode =
        contour_extractor_helpers::contourApproxModeFromString(_chainApprox);
    const bool ok1 = contour_extractor_helpers::extractContoursForImage(images.first_gray,
                                                                        sd.first.response,
                                                                        sd.first.contours,
                                                                        _blurKernel,
                                                                        _autoCanny,
                                                                        _cannyThreshold1,
                                                                        _cannyThreshold2,
                                                                        _apertureSize,
                                                                        retrievalMode,
                                                                        approxMode,
                                                                        _minArea,
                                                                        _minPerimeter,
                                                                        _minPoints,
                                                                        _minBboxWidth,
                                                                        _minBboxHeight,
                                                                        _minExtent,
                                                                        _maxAspectRatio,
                                                                        _maxContours,
                                                                        _contourThickness);
    const bool ok2 = contour_extractor_helpers::extractContoursForImage(images.second_gray,
                                                                        sd.second.response,
                                                                        sd.second.contours,
                                                                        _blurKernel,
                                                                        _autoCanny,
                                                                        _cannyThreshold1,
                                                                        _cannyThreshold2,
                                                                        _apertureSize,
                                                                        retrievalMode,
                                                                        approxMode,
                                                                        _minArea,
                                                                        _minPerimeter,
                                                                        _minPoints,
                                                                        _minBboxWidth,
                                                                        _minBboxHeight,
                                                                        _minExtent,
                                                                        _maxAspectRatio,
                                                                        _maxContours,
                                                                        _contourThickness);

    IR_LOG_INFO("ContourExtractor extracted contours: ",
                sd.first.contours.size(),
                " / ",
                sd.second.contours.size());
    return ok1 && ok2;
}

} // namespace ir

