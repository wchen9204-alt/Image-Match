#include "matcher/structure/phase_correlate_associator.h"

#include <opencv2/imgproc.hpp>

#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

bool preparePhaseImage(const cv::Mat& response, cv::Mat& out, int blurKernel) {
    if (response.empty()) {
        return false;
    }

    cv::Mat src;
    if (response.channels() == 1) {
        src = response;
    } else {
        cv::cvtColor(response, src, cv::COLOR_BGR2GRAY);
    }

    if (cv::countNonZero(src) == 0) {
        return false;
    }

    src.convertTo(out, CV_32F, 1.0 / 255.0);
    if (blurKernel >= 3) {
        if (blurKernel % 2 == 0) {
            ++blurKernel;
        }
        cv::GaussianBlur(out, out, cv::Size(blurKernel, blurKernel), 0.0);
    }
    return true;
}

} // namespace

PhaseCorrelateAssociator::PhaseCorrelateAssociator(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    _scoreThreshold = yaml_utils::getDouble(params, "responseThreshold", 0.01);
    _blurKernel = yaml_utils::getInt(params, "phaseBlurKernel", 5);
}

bool PhaseCorrelateAssociator::associate(RegistrationContext& ctx) {
    auto& md = ctx.structure_match_data;
    md.clear();
    md.method = name();

    cv::Mat src;
    cv::Mat dst;
    if (!preparePhaseImage(ctx.structure_data.first.response, src, _blurKernel) ||
        !preparePhaseImage(ctx.structure_data.second.response, dst, _blurKernel)) {
        md.message = "structure response images are empty";
        return false;
    }

    if (src.size() != dst.size()) {
        md.message = "structure response images have different sizes";
        return false;
    }

    double score = 0.0;
    const cv::Point2d shift = cv::phaseCorrelate(src, dst, cv::noArray(), &score);

    md.translation = shift;
    md.score = score;
    md.valid = score >= _scoreThreshold;
    if (!md.valid) {
        md.message = "phase correlation response below threshold: " + std::to_string(score);
        IR_LOG_WARN("PhaseCorrelateAssociator rejected match: ", md.message);
    }

    IR_LOG_INFO("PhaseCorrelateAssociator estimated translation dx=",
                shift.x,
                ", dy=",
                shift.y,
                ", score=",
                score);
    return md.valid;
}

} // namespace ir
