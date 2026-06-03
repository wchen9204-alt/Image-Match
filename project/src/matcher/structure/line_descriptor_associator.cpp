#include "matcher/structure/line_descriptor_associator.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>

#include <opencv2/features2d.hpp>

#ifdef IR_HAS_OPENCV_LINE_DESCRIPTOR
#include <opencv2/line_descriptor.hpp>
#endif

#include "core/types.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

std::string upperAscii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

#ifdef IR_HAS_OPENCV_LINE_DESCRIPTOR

cv::line_descriptor::KeyLine toKeyLine(const cv::Vec4i& line, int classId) {
    const float x1 = static_cast<float>(line[0]);
    const float y1 = static_cast<float>(line[1]);
    const float x2 = static_cast<float>(line[2]);
    const float y2 = static_cast<float>(line[3]);
    const float dx = x2 - x1;
    const float dy = y2 - y1;

    cv::line_descriptor::KeyLine keyLine;
    keyLine.startPointX = x1;
    keyLine.startPointY = y1;
    keyLine.endPointX = x2;
    keyLine.endPointY = y2;
    keyLine.sPointInOctaveX = x1;
    keyLine.sPointInOctaveY = y1;
    keyLine.ePointInOctaveX = x2;
    keyLine.ePointInOctaveY = y2;
    keyLine.lineLength = std::sqrt(dx * dx + dy * dy);
    keyLine.angle = std::atan2(dy, dx);
    keyLine.class_id = classId;
    keyLine.octave = 0;
    keyLine.numOfPixels = static_cast<int>(std::max(1.0f, keyLine.lineLength));
    keyLine.response = keyLine.lineLength;
    keyLine.pt = cv::Point2f((x1 + x2) * 0.5f, (y1 + y2) * 0.5f);
    return keyLine;
}

std::vector<cv::line_descriptor::KeyLine> toKeyLines(const std::vector<cv::Vec4i>& lines) {
    std::vector<cv::line_descriptor::KeyLine> out;
    out.reserve(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) {
        out.push_back(toKeyLine(lines[i], static_cast<int>(i)));
    }
    return out;
}

bool computeLbd(const cv::Mat& gray,
                std::vector<cv::line_descriptor::KeyLine>& keyLines,
                cv::Mat& descriptors,
                std::string& message) {
    descriptors.release();
    if (gray.empty()) {
        message = "input gray image is empty";
        return false;
    }
    if (keyLines.empty()) {
        message = "line set is empty";
        return false;
    }

    cv::Ptr<cv::line_descriptor::BinaryDescriptor> descriptor =
        cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
    descriptor->compute(gray, keyLines, descriptors);
    if (descriptors.empty()) {
        message = "LBD descriptor matrix is empty";
        return false;
    }
    return true;
}

std::vector<cv::DMatch> ratioMatch(const cv::Mat& srcDescriptors,
                                   const cv::Mat& dstDescriptors,
                                   const std::vector<cv::line_descriptor::KeyLine>& srcKeys,
                                   const std::vector<cv::line_descriptor::KeyLine>& dstKeys,
                                   double ratio) {
    cv::BFMatcher matcher(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(srcDescriptors, dstDescriptors, knn, 2);

    std::vector<cv::DMatch> accepted;
    for (const auto& neighbours : knn) {
        if (neighbours.empty()) {
            continue;
        }
        const cv::DMatch& best = neighbours[0];
        if (neighbours.size() >= 2 &&
            best.distance > static_cast<float>(ratio) * neighbours[1].distance) {
            continue;
        }
        if (best.queryIdx < 0 || best.trainIdx < 0 ||
            best.queryIdx >= static_cast<int>(srcKeys.size()) ||
            best.trainIdx >= static_cast<int>(dstKeys.size())) {
            continue;
        }

        const int queryLine = srcKeys[static_cast<size_t>(best.queryIdx)].class_id;
        const int trainLine = dstKeys[static_cast<size_t>(best.trainIdx)].class_id;
        if (queryLine < 0 || trainLine < 0) {
            continue;
        }
        accepted.emplace_back(queryLine, trainLine, best.distance);
    }

    std::stable_sort(accepted.begin(), accepted.end(), [](const cv::DMatch& a,
                                                          const cv::DMatch& b) {
        return a.distance < b.distance;
    });

    std::vector<cv::DMatch> unique;
    std::vector<unsigned char> usedQuery(srcKeys.size(), 0);
    std::vector<unsigned char> usedTrain(dstKeys.size(), 0);
    for (const auto& m : accepted) {
        const size_t qi = static_cast<size_t>(m.queryIdx);
        const size_t ti = static_cast<size_t>(m.trainIdx);
        if (qi >= usedQuery.size() || ti >= usedTrain.size() || usedQuery[qi] || usedTrain[ti]) {
            continue;
        }
        usedQuery[qi] = 1;
        usedTrain[ti] = 1;
        unique.push_back(m);
    }
    return unique;
}

#endif

} // namespace

LineDescriptorAssociator::LineDescriptorAssociator(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    _descriptor = yaml_utils::getString(params, "descriptor", "LBD");
    _matcher = yaml_utils::getString(params, "matcher", "BF");
    _ratio = yaml_utils::getDouble(params, "ratio", 0.75);
    _minMatches = yaml_utils::getInt(params, "minMatches", 4);

    IR_LOG_INFO("LineDescriptorAssociator: descriptor=",
                _descriptor,
                ", matcher=",
                _matcher,
                ", ratio=",
                _ratio,
                ", minMatches=",
                _minMatches);
}

bool LineDescriptorAssociator::associate(RegistrationContext& ctx) {
    auto& md = ctx.structure_match_data;
    md.clear();
    md.method = name();

#ifndef IR_HAS_OPENCV_LINE_DESCRIPTOR
    md.message =
        "OpenCV line_descriptor module is not available; rebuild OpenCV contrib with "
        "line_descriptor to enable LBD";
    IR_LOG_ERROR("LineDescriptorAssociator: ", md.message);
    return false;
#else
    if (ctx.structure_data.type != StructureType::LINE) {
        md.message = "LineDescriptorAssociator requires LINE structure data";
        return false;
    }

    const std::string descriptor = upperAscii(_descriptor);
    const std::string matcher = upperAscii(_matcher);
    if (descriptor != "LBD") {
        md.message = "unsupported line descriptor: " + _descriptor;
        return false;
    }
    if (matcher != "BF" && matcher != "BFMATCHER" && matcher != "BRUTEFORCE") {
        md.message = "unsupported line descriptor matcher: " + _matcher;
        return false;
    }

    std::vector<cv::line_descriptor::KeyLine> srcKeys =
        toKeyLines(ctx.structure_data.first.lines);
    std::vector<cv::line_descriptor::KeyLine> dstKeys =
        toKeyLines(ctx.structure_data.second.lines);

    cv::Mat srcDescriptors;
    cv::Mat dstDescriptors;
    if (!computeLbd(ctx.images.first_gray, srcKeys, srcDescriptors, md.message) ||
        !computeLbd(ctx.images.second_gray, dstKeys, dstDescriptors, md.message)) {
        IR_LOG_WARN("LineDescriptorAssociator rejected match: ", md.message);
        return false;
    }

    md.line_matches = ratioMatch(srcDescriptors, dstDescriptors, srcKeys, dstKeys, _ratio);
    md.valid = static_cast<int>(md.line_matches.size()) >= _minMatches;
    md.score = ctx.structure_data.first.lines.empty()
                   ? 0.0
                   : static_cast<double>(md.line_matches.size()) /
                         static_cast<double>(ctx.structure_data.first.lines.size());
    if (!md.valid) {
        md.message = "not enough LBD line matches: " + std::to_string(md.line_matches.size());
        IR_LOG_WARN("LineDescriptorAssociator rejected match: ", md.message);
        return false;
    }

    IR_LOG_INFO("LineDescriptorAssociator produced LBD matches: ",
                md.line_matches.size(),
                ", srcDescriptors=",
                srcDescriptors.rows,
                ", dstDescriptors=",
                dstDescriptors.rows);
    return true;
#endif
}

} // namespace ir
