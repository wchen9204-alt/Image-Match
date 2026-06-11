#include "pipeline/structure_pipeline.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#ifdef IR_HAS_OPENCV_LINE_DESCRIPTOR
#include <opencv2/line_descriptor.hpp>
#endif

#include "core/config.h"
#include "core/factory.h"
#include "data/correspondence_view.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/yaml_utils.h"

namespace fs = std::filesystem;

namespace ir {

namespace {

// 将任意输入图像转为 BGR，便于在统一画布上叠加彩色结构连线。
cv::Mat toBgr(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }
    if (image.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }
    if (image.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }
    return image.clone();
}

// 从结构响应图中收集非零点，作为结构匹配连线可视化的候选点。
std::vector<cv::Point> collectResponsePoints(const cv::Mat& response) {
    std::vector<cv::Point> points;
    if (response.empty()) {
        return points;
    }

    cv::Mat gray;
    if (response.channels() == 1) {
        gray = response;
    } else {
        cv::cvtColor(response, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat binary;
    cv::threshold(gray, binary, 0.0, 255.0, cv::THRESH_BINARY);
    if (binary.depth() != CV_8U) {
        binary.convertTo(binary, CV_8U);
    }
    cv::findNonZero(binary, points);
    return points;
}

// 在 target 响应图的局部窗口内寻找距离投影点最近的结构点。
bool nearestResponsePoint(const cv::Mat& response,
                          const cv::Point& center,
                          int radius,
                          cv::Point& nearest) {
    if (response.empty()) {
        return false;
    }

    const int x0 = std::max(0, center.x - radius);
    const int y0 = std::max(0, center.y - radius);
    const int x1 = std::min(response.cols - 1, center.x + radius);
    const int y1 = std::min(response.rows - 1, center.y + radius);
    if (x0 > x1 || y0 > y1) {
        return false;
    }

    // 局部邻域线性扫描足够直观，也避免为可视化阶段引入额外索引结构。
    double bestDist2 = std::numeric_limits<double>::infinity();
    bool found = false;
    for (int y = y0; y <= y1; ++y) {
        const uchar* row = response.ptr<uchar>(y);
        for (int x = x0; x <= x1; ++x) {
            if (row[x] == 0) {
                continue;
            }
            const double dx = static_cast<double>(x - center.x);
            const double dy = static_cast<double>(y - center.y);
            const double d2 = dx * dx + dy * dy;
            if (d2 < bestDist2) {
                bestDist2 = d2;
                nearest = cv::Point(x, y);
                found = true;
            }
        }
    }
    return found;
}

// 渲染结构匹配连线图：
// 1. 左右拼接 source / target 原图；
// 2. 从 source 结构响应点中均匀抽样；
// 3. 根据结构匹配得到的平移量投影到 target；
// 4. 在投影点附近寻找最近 target 响应点并绘制连线。
cv::Point lineMidpoint(const cv::Vec4i& line) {
    return cv::Point((line[0] + line[2]) / 2, (line[1] + line[3]) / 2);
}

cv::Point2f contourCentroidPoint(const std::vector<cv::Point>& contour) {
    const cv::Moments m = cv::moments(contour);
    if (std::abs(m.m00) < 1e-9) {
        return {-1.0f, -1.0f};
    }
    return {static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00)};
}

cv::Point2d applyAffinePoint(const cv::Mat& A, const cv::Point2d& p) {
    return {A.at<double>(0, 0) * p.x + A.at<double>(0, 1) * p.y + A.at<double>(0, 2),
            A.at<double>(1, 0) * p.x + A.at<double>(1, 1) * p.y + A.at<double>(1, 2)};
}

cv::Point2f linePoint1(const cv::Vec4i& line) {
    return {static_cast<float>(line[0]), static_cast<float>(line[1])};
}

cv::Point2f linePoint2(const cv::Vec4i& line) {
    return {static_cast<float>(line[2]), static_cast<float>(line[3])};
}

#ifdef IR_HAS_OPENCV_LINE_DESCRIPTOR
cv::line_descriptor::KeyLine toDrawableKeyLine(const cv::Vec4i& line, int classId) {
    const cv::Point2f p1 = linePoint1(line);
    const cv::Point2f p2 = linePoint2(line);
    const cv::Point2f delta = p2 - p1;

    cv::line_descriptor::KeyLine keyLine;
    keyLine.startPointX = p1.x;
    keyLine.startPointY = p1.y;
    keyLine.endPointX = p2.x;
    keyLine.endPointY = p2.y;
    keyLine.sPointInOctaveX = p1.x;
    keyLine.sPointInOctaveY = p1.y;
    keyLine.ePointInOctaveX = p2.x;
    keyLine.ePointInOctaveY = p2.y;
    keyLine.lineLength = cv::norm(delta);
    keyLine.angle = std::atan2(delta.y, delta.x);
    keyLine.class_id = classId;
    keyLine.octave = 0;
    keyLine.numOfPixels = static_cast<int>(std::max(1.0f, keyLine.lineLength));
    keyLine.response = keyLine.lineLength;
    keyLine.size = keyLine.lineLength;
    keyLine.pt = (p1 + p2) * 0.5f;
    return keyLine;
}

std::vector<cv::line_descriptor::KeyLine> toDrawableKeyLines(
    const std::vector<cv::Vec4i>& lines) {
    std::vector<cv::line_descriptor::KeyLine> out;
    out.reserve(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) {
        out.push_back(toDrawableKeyLine(lines[i], static_cast<int>(i)));
    }
    return out;
}
#endif

void promoteStructureInliersFromGeometryMask(RegistrationContext& ctx) {
    auto& md = ctx.structure_match_data;
    const auto& gd = ctx.geometry_data;
    const CorrespondenceView view = buildStructureCorrespondenceView(ctx);

    std::vector<int> pointInlierCounts(md.line_matches.size(), 0);
    for (size_t i = 0; i < view.filtered.size() && i < gd.inlier_mask.size(); ++i) {
        if (!gd.inlier_mask[i]) {
            continue;
        }
        const int structureMatchIndex = view.filtered[i].imgIdx;
        if (structureMatchIndex >= 0 &&
            structureMatchIndex < static_cast<int>(pointInlierCounts.size())) {
            ++pointInlierCounts[static_cast<size_t>(structureMatchIndex)];
        }
    }

    md.inlier_line_matches.clear();
    const int requiredPoints = ctx.structure_data.type == StructureType::LINE ? 2 : 1;
    for (size_t i = 0; i < pointInlierCounts.size(); ++i) {
        if (pointInlierCounts[i] >= requiredPoints) {
            md.inlier_line_matches.push_back(md.line_matches[i]);
        }
    }
    md.score = md.line_matches.empty()
                   ? 0.0
                   : static_cast<double>(md.inlier_line_matches.size()) /
                         static_cast<double>(md.line_matches.size());
}

cv::Mat renderLineSegmentMatches(const RegistrationContext& ctx,
                                 const std::vector<cv::DMatch>& matches,
                                 int maxMatches) {
    if (matches.empty() || ctx.images.first.empty() || ctx.images.second.empty() ||
        ctx.structure_data.first.lines.empty() || ctx.structure_data.second.lines.empty()) {
        return {};
    }

    cv::Mat src = toBgr(ctx.images.first);
    cv::Mat dst = toBgr(ctx.images.second);
    if (src.empty() || dst.empty()) {
        return {};
    }

    std::vector<cv::DMatch> draw = matches;
    if (maxMatches > 0 && static_cast<int>(draw.size()) > maxMatches) {
        std::partial_sort(draw.begin(),
                          draw.begin() + maxMatches,
                          draw.end(),
                          [](const cv::DMatch& a, const cv::DMatch& b) {
                              return a.distance < b.distance;
                          });
        draw.resize(maxMatches);
    }

#ifdef IR_HAS_OPENCV_LINE_DESCRIPTOR
    const std::vector<cv::line_descriptor::KeyLine> srcKeyLines =
        toDrawableKeyLines(ctx.structure_data.first.lines);
    const std::vector<cv::line_descriptor::KeyLine> dstKeyLines =
        toDrawableKeyLines(ctx.structure_data.second.lines);

    try {
        cv::Mat lineMatches;
        const std::vector<char> matchesMask(draw.size(), 1);
        cv::line_descriptor::drawLineMatches(src,
                                             srcKeyLines,
                                             dst,
                                             dstKeyLines,
                                             draw,
                                             lineMatches,
                                             cv::Scalar(255, 180, 0),
                                             cv::Scalar(0, 220, 255),
                                             matchesMask);
        if (!lineMatches.empty()) {
            return lineMatches;
        }
    } catch (const cv::Exception& e) {
        IR_LOG_WARN("drawLineMatches failed; falling back to manual line rendering: ", e.what());
    }
#endif

    const int canvasRows = std::max(src.rows, dst.rows);
    const int canvasCols = src.cols + dst.cols;
    cv::Mat canvas(canvasRows, canvasCols, src.type(), cv::Scalar::all(0));
    src.copyTo(canvas(cv::Rect(0, 0, src.cols, src.rows)));
    dst.copyTo(canvas(cv::Rect(src.cols, 0, dst.cols, dst.rows)));

    int drawn = 0;
    for (const auto& m : draw) {
        if (m.queryIdx < 0 || m.trainIdx < 0 ||
            m.queryIdx >= static_cast<int>(ctx.structure_data.first.lines.size()) ||
            m.trainIdx >= static_cast<int>(ctx.structure_data.second.lines.size())) {
            continue;
        }

        const cv::Vec4i srcLine = ctx.structure_data.first.lines[static_cast<size_t>(m.queryIdx)];
        const cv::Vec4i dstLine = ctx.structure_data.second.lines[static_cast<size_t>(m.trainIdx)];
        const cv::Point dstMidRaw = lineMidpoint(dstLine);

        const cv::Point srcP1(srcLine[0], srcLine[1]);
        const cv::Point srcP2(srcLine[2], srcLine[3]);
        const cv::Point dstP1(dstLine[0] + src.cols, dstLine[1]);
        const cv::Point dstP2(dstLine[2] + src.cols, dstLine[3]);
        const cv::Point srcMid = lineMidpoint(srcLine);
        const cv::Point dstMid(dstMidRaw.x + src.cols, dstMidRaw.y);

        const cv::Scalar srcColor(0, 180, 255);
        const cv::Scalar dstColor(0, 255, 0);
        const cv::Scalar matchColor(255, 180, 0);
        cv::line(canvas, srcP1, srcP2, srcColor, 2, cv::LINE_AA);
        cv::line(canvas, dstP1, dstP2, dstColor, 2, cv::LINE_AA);
        cv::line(canvas, srcMid, dstMid, matchColor, 1, cv::LINE_AA);
        cv::circle(canvas, srcMid, 2, srcColor, cv::FILLED, cv::LINE_AA);
        cv::circle(canvas, dstMid, 2, dstColor, cv::FILLED, cv::LINE_AA);
        ++drawn;
    }

    return drawn > 0 ? canvas : cv::Mat{};
}

// 轮廓匹配可视化：
// 1. 左右拼接 source / target 原图。
// 2. 在两侧分别描边显示匹配到的 source / target contour。
// 3. 使用轮廓质心作为连线锚点，直观看每一对轮廓对应关系。
cv::Mat renderContourMatches(const RegistrationContext& ctx,
                             const std::vector<cv::DMatch>& matches,
                             int maxMatches) {
    if (matches.empty() || ctx.images.first.empty() || ctx.images.second.empty() ||
        ctx.structure_data.first.contours.empty() || ctx.structure_data.second.contours.empty()) {
        return {};
    }

    cv::Mat src = toBgr(ctx.images.first);
    cv::Mat dst = toBgr(ctx.images.second);
    if (src.empty() || dst.empty()) {
        return {};
    }

    std::vector<cv::DMatch> draw = matches;
    if (maxMatches > 0 && static_cast<int>(draw.size()) > maxMatches) {
        std::partial_sort(draw.begin(),
                          draw.begin() + maxMatches,
                          draw.end(),
                          [](const cv::DMatch& a, const cv::DMatch& b) {
                              return a.distance < b.distance;
                          });
        draw.resize(maxMatches);
    }

    const int canvasRows = std::max(src.rows, dst.rows);
    const int canvasCols = src.cols + dst.cols;
    cv::Mat canvas(canvasRows, canvasCols, src.type(), cv::Scalar::all(0));
    src.copyTo(canvas(cv::Rect(0, 0, src.cols, src.rows)));
    dst.copyTo(canvas(cv::Rect(src.cols, 0, dst.cols, dst.rows)));

    int drawn = 0;
    for (const auto& m : draw) {
        if (m.queryIdx < 0 || m.trainIdx < 0 ||
            m.queryIdx >= static_cast<int>(ctx.structure_data.first.contours.size()) ||
            m.trainIdx >= static_cast<int>(ctx.structure_data.second.contours.size())) {
            continue;
        }

        const auto& srcContour =
            ctx.structure_data.first.contours[static_cast<size_t>(m.queryIdx)];
        const auto& dstContour =
            ctx.structure_data.second.contours[static_cast<size_t>(m.trainIdx)];
        if (srcContour.empty() || dstContour.empty()) {
            continue;
        }

        const cv::Point2f srcCentroid = contourCentroidPoint(srcContour);
        const cv::Point2f dstCentroid = contourCentroidPoint(dstContour);
        if (srcCentroid.x < 0.0f || srcCentroid.y < 0.0f ||
            dstCentroid.x < 0.0f || dstCentroid.y < 0.0f) {
            continue;
        }

        std::vector<std::vector<cv::Point>> srcDrawContours{srcContour};
        std::vector<cv::Point> shiftedDstContour;
        shiftedDstContour.reserve(dstContour.size());
        for (const cv::Point& p : dstContour) {
            shiftedDstContour.emplace_back(p.x + src.cols, p.y);
        }
        std::vector<std::vector<cv::Point>> dstDrawContours{shiftedDstContour};

        const cv::Scalar srcColor(0, 180, 255);
        const cv::Scalar dstColor(0, 255, 0);
        const cv::Scalar matchColor(255, 180, 0);
        cv::drawContours(canvas, srcDrawContours, -1, srcColor, 2, cv::LINE_AA);
        cv::drawContours(canvas, dstDrawContours, -1, dstColor, 2, cv::LINE_AA);

        const cv::Point srcCenter(cvRound(srcCentroid.x), cvRound(srcCentroid.y));
        const cv::Point dstCenter(cvRound(dstCentroid.x) + src.cols, cvRound(dstCentroid.y));
        cv::line(canvas, srcCenter, dstCenter, matchColor, 1, cv::LINE_AA);
        cv::circle(canvas, srcCenter, 3, srcColor, cv::FILLED, cv::LINE_AA);
        cv::circle(canvas, dstCenter, 3, dstColor, cv::FILLED, cv::LINE_AA);
        ++drawn;
    }

    return drawn > 0 ? canvas : cv::Mat{};
}

cv::Mat renderStructureMatches(const RegistrationContext& ctx, int maxMatches) {
    if (!ctx.structure_match_data.valid || ctx.images.first.empty() || ctx.images.second.empty() ||
        ctx.structure_data.first.response.empty() || ctx.structure_data.second.response.empty()) {
        return {};
    }

    cv::Mat src = toBgr(ctx.images.first);
    cv::Mat dst = toBgr(ctx.images.second);
    if (src.empty() || dst.empty()) {
        return {};
    }

    cv::Mat dstResponseGray;
    if (ctx.structure_data.second.response.channels() == 1) {
        dstResponseGray = ctx.structure_data.second.response;
    } else {
        cv::cvtColor(ctx.structure_data.second.response, dstResponseGray, cv::COLOR_BGR2GRAY);
    }
    cv::Mat dstResponse;
    cv::threshold(dstResponseGray, dstResponse, 0.0, 255.0, cv::THRESH_BINARY);
    if (dstResponse.depth() != CV_8U) {
        dstResponse.convertTo(dstResponse, CV_8U);
    }

    const std::vector<cv::Point> srcPoints =
        collectResponsePoints(ctx.structure_data.first.response);
    if (srcPoints.empty() || cv::countNonZero(dstResponse) == 0) {
        return {};
    }

    // 左右拼接原图，保持和 keypoint drawMatches 类似的视觉布局。
    const int canvasRows = std::max(src.rows, dst.rows);
    const int canvasCols = src.cols + dst.cols;
    cv::Mat canvas(canvasRows, canvasCols, src.type(), cv::Scalar::all(0));
    src.copyTo(canvas(cv::Rect(0, 0, src.cols, src.rows)));
    dst.copyTo(canvas(cv::Rect(src.cols, 0, dst.cols, dst.rows)));

    const int limit = maxMatches > 0 ? maxMatches : 100;
    const int candidateCount = std::max(limit * 20, limit);
    const int stride =
        std::max(1, static_cast<int>(srcPoints.size()) / std::max(1, candidateCount));
    const int searchRadius = 10;
    const cv::Point2d shift = ctx.structure_match_data.translation;
    const cv::Mat& affine = ctx.structure_match_data.affine;
    const bool hasAffine = !affine.empty() && affine.rows == 2 && affine.cols == 3 &&
                           affine.type() == CV_64F;

    // 均匀抽样 source 响应点，避免边缘点过密导致连线图不可读。
    int drawn = 0;
    for (size_t i = 0; i < srcPoints.size() && drawn < limit; i += static_cast<size_t>(stride)) {
        const cv::Point& p = srcPoints[i];
        const cv::Point2d projectedFloat =
            hasAffine ? applyAffinePoint(affine, cv::Point2d(p)) : cv::Point2d(p) + shift;
        const cv::Point projected(cvRound(projectedFloat.x), cvRound(projectedFloat.y));
        if (projected.x < 0 || projected.y < 0 || projected.x >= dst.cols ||
            projected.y >= dst.rows) {
            continue;
        }

        cv::Point q;
        if (!nearestResponsePoint(dstResponse, projected, searchRadius, q)) {
            continue;
        }

        const cv::Point pCanvas = p;
        const cv::Point qCanvas(q.x + src.cols, q.y);
        const cv::Scalar color(0, 255, 255);
        cv::line(canvas, pCanvas, qCanvas, color, 1, cv::LINE_AA);
        cv::circle(canvas, pCanvas, 2, cv::Scalar(0, 180, 255), cv::FILLED, cv::LINE_AA);
        cv::circle(canvas, qCanvas, 2, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
        ++drawn;
    }

    return drawn > 0 ? canvas : cv::Mat{};
}

} // namespace

void StructurePipeline::resetStages() {
    _extractor.reset();
    _associator.reset();
    _geometry.reset();
    _filters.clear();
}

bool StructurePipeline::configureStages(const PipelineConfig& cfg) {
    // 1. 检查 structure YAML 路径，结构流水线依赖同一份配置创建提取器和关联器。
    if (cfg.structure_path.empty()) {
        IR_LOG_ERROR("StructurePipeline: missing structure config path.");
        return false;
    }
    // 2. 读取结构配置，并创建结构提取器、结构匹配器与通用几何估计器。
    const YAML::Node structureCfg = Config::load(cfg.structure_path);
    _extractor = Factory::createStructureExtractor(structureCfg);
    _associator = Factory::createStructureAssociator(structureCfg);
    if (!cfg.geometry_path.empty()) {
        const YAML::Node geometryCfg = Config::load(cfg.geometry_path);
        _geometry = Factory::createGeometryEstimator(geometryCfg);
    } else {
        IR_LOG_WARN("StructurePipeline: missing geometry config path; falling back to "
                    "association transform when available.");
    }

    // 3. 从 pipeline YAML 的 filters: 列表加载过滤链（与 KeypointPipeline 一致）。
    _filters.clear();
    for (const auto& fp : cfg.filter_paths) {
        _filters.push_back(Factory::createFilter(Config::load(fp)));
    }

    // 4. 输出当前结构方法组合，方便批量实验时核对配置是否生效。
    IR_LOG_INFO("StructurePipeline stages configured: extractor=",
                _extractor->name(),
                ", associator=",
                _associator->name(),
                ", geometry=",
                (_geometry ? _geometry->name() : std::string("NONE")),
                ", filters=",
                _filters.size());
    return true;
}

bool StructurePipeline::runExtraction(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_extract_ms);

    // 1. 检查结构提取器是否已经由 configureStages 创建。
    if (!_extractor) {
        IR_LOG_ERROR("StructurePipeline::runExtraction: no structure extractor configured.");
        return false;
    }

    // 2. 执行结构响应提取，结果写入 ctx.structure_data。
    const bool ok = _extractor->extract(ctx);

    // 3. 按当前结构类型统计 source / target 的结构数量。
    ctx.result.num_structures_first =
        ctx.structure_data.first.primitiveCount(ctx.structure_data.type);
    ctx.result.num_structures_second =
        ctx.structure_data.second.primitiveCount(ctx.structure_data.type);
    return ok;
}

bool StructurePipeline::runAssociation(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_match_ms);

    // 1. 检查结构匹配器和结构响应数据是否可用。
    if (!_associator) {
        IR_LOG_ERROR("StructurePipeline::runAssociation: no structure associator configured.");
        return false;
    }
    if (ctx.structure_data.empty()) {
        IR_LOG_ERROR("StructurePipeline::runAssociation: structure data is empty.");
        return false;
    }

    // 2. 执行结构匹配，关联器负责写入 raw_matches_knn / filtered_matches 等。
    const bool ok = _associator->associate(ctx);
    if (!ok) {
        return false;
    }

    // 3. 对有线匹配数据的关联器，执行过滤链。
    if (!_filters.empty() && !ctx.structure_match_data.raw_matches_knn.empty()) {
        if (!runFilters(ctx)) {
            IR_LOG_WARN("StructurePipeline::runAssociation: filter chain rejected all matches.");
            return false;
        }
    }

    // 4. 同步匹配计数到运行摘要。
    if (!ctx.structure_match_data.line_matches.empty() ||
        !ctx.structure_match_data.inlier_line_matches.empty()) {
        ctx.result.num_raw_matches =
            static_cast<int>(ctx.structure_match_data.raw_matches_knn.size());
        ctx.result.num_filtered_matches =
            static_cast<int>(ctx.structure_match_data.line_matches.size());
    } else {
        ctx.result.num_raw_matches = ok ? 1 : 0;
        ctx.result.num_filtered_matches = ok ? 1 : 0;
    }
    return ok;
}

bool StructurePipeline::runFilters(RegistrationContext& ctx) {
    auto& md = ctx.structure_match_data;

    // 1. 关联器已种子 filtered_matches（top-1），直接交给过滤链逐级处理。
    //    RatioTest 等需要 KNN 的过滤器会从 raw_matches_knn 重新计算。
    for (const auto& f : _filters) {
        if (!f) {
            continue;
        }
        if (!f->apply(ctx)) {
            IR_LOG_WARN("StructurePipeline filter ", f->name(), " returned false");
        }
    }

    // 2. 过滤完成后同步 filtered_matches → line_matches。
    md.line_matches = md.filtered_matches;
    md.inlier_line_matches = md.line_matches;
    md.valid = !md.line_matches.empty();

    IR_LOG_INFO("StructurePipeline filters done: ",
                md.line_matches.size(),
                " / ",
                md.raw_matches_knn.size(),
                " line matches after filtering");
    return md.valid;
}

bool StructurePipeline::runEstimation(RegistrationContext& ctx) {
    ScopedTimer st(ctx.result.t_geometry_ms);

    // 1. 初始化几何结果。
    auto& gd = ctx.geometry_data;
    gd.clear();
    ctx.correspondence_source = "STRUCTURE";

    // 2. 检查结构匹配是否给出了有效结果，无效时保留失败原因。
    if (!ctx.structure_match_data.valid) {
        gd.message = ctx.structure_match_data.message.empty()
                         ? "structure match result is invalid"
                         : ctx.structure_match_data.message;
        IR_LOG_WARN("StructurePipeline::runEstimation: ", gd.message);
        return false;
    }

    // 3. 有结构匹配时，几何估计器直接读取 CorrespondenceView，不再临时伪装成 keypoint match。
    if (_geometry && !ctx.structure_match_data.line_matches.empty()) {
        if (buildStructureCorrespondenceView(ctx).empty()) {
            gd.message = "no valid structure correspondences for geometry estimation";
            IR_LOG_WARN("StructurePipeline::runEstimation: ", gd.message);
            return false;
        }

        const cv::Point2d assocTranslation = ctx.structure_match_data.translation;
        const cv::Mat assocAffine = ctx.structure_match_data.affine.clone();

        const bool ok = _geometry->estimate(ctx);
        promoteStructureInliersFromGeometryMask(ctx);

        if (ok) {
            ctx.structure_match_data.affine =
                (ctx.geometry_data.A.empty() ? cv::Mat{} : ctx.geometry_data.A.clone());
            if (!ctx.geometry_data.A.empty()) {
                ctx.structure_match_data.translation =
                    {ctx.geometry_data.A.at<double>(0, 2), ctx.geometry_data.A.at<double>(1, 2)};
            } else if (!ctx.geometry_data.H.empty()) {
                ctx.structure_match_data.translation =
                    {ctx.geometry_data.H.at<double>(0, 2) / ctx.geometry_data.H.at<double>(2, 2),
                     ctx.geometry_data.H.at<double>(1, 2) / ctx.geometry_data.H.at<double>(2, 2)};
            }
        } else {
            ctx.structure_match_data.affine = assocAffine;
            ctx.structure_match_data.translation = assocTranslation;
        }

        // 保留关联器/预筛后的匹配数量，不在几何阶段用内点数覆盖，
        // 这样 summary 里可以区分“筛后匹配数”和“最终几何内点数”。
        ctx.result.num_inliers =
            static_cast<int>(ctx.structure_match_data.inlier_line_matches.size());
        ctx.result.inlier_ratio = ctx.structure_match_data.score;
        if (!ok) {
            ctx.structure_match_data.message = ctx.geometry_data.message;
            return false;
        }

        IR_LOG_INFO("StructurePipeline geometry estimated by ",
                    _geometry->name(),
                    ", correspondence_inliers=",
                    ctx.geometry_data.num_inliers,
                    ", line_inliers=",
                    ctx.structure_match_data.inlier_line_matches.size(),
                    ", score=",
                    ctx.structure_match_data.score);
        return true;
    }

    // 4. 无几何估计器或无线段匹配时的回退路径：
    //    响应图关联器（PhaseCorrelate / Chamfer / Hausdorff / ICP）直接给出平移，
    //    线段关联器在无 geometry 配置时也以平均平移作为兜底。
    gd.type = GeometryType::AFFINE;
    const cv::Point2d shift = ctx.structure_match_data.translation;
    const cv::Mat& affine = ctx.structure_match_data.affine;
    if (!affine.empty() && affine.rows == 2 && affine.cols == 3) {
        affine.convertTo(gd.A, CV_64F);
    } else {
        gd.A = (cv::Mat_<double>(2, 3) << 1.0, 0.0, shift.x, 0.0, 1.0, shift.y);
    }
    gd.valid = true;

    // 5. 同步内点数和得分到通用运行摘要。
    gd.num_inliers = ctx.structure_match_data.inlier_line_matches.empty()
                         ? ctx.result.num_structures_first
                         : static_cast<int>(ctx.structure_match_data.inlier_line_matches.size());
    gd.inlier_ratio = ctx.structure_match_data.score;
    ctx.result.num_inliers = gd.num_inliers;
    ctx.result.inlier_ratio = gd.inlier_ratio;

    IR_LOG_INFO("StructurePipeline fallback affine dx=",
                gd.A.at<double>(0, 2),
                ", dy=",
                gd.A.at<double>(1, 2),
                ", score=",
                ctx.structure_match_data.score);
    return true;
}

std::string StructurePipeline::buildOutputStem(const RegistrationContext& ctx) const {
    const std::string sampleStem = ctx.image1_path.stem().string() + "_" +
                                   ctx.image2_path.stem().string();
    return sampleStem + "_" + (_extractor ? _extractor->name() : std::string("STRUCTURE")) +
           "_" + (_associator ? _associator->name() : std::string("MATCH"));
}

bool StructurePipeline::saveOutputs(RegistrationContext& ctx) {
    if (_config.output_dir.empty()) {
        return true;
    }

    // 1. 创建结构响应图和结构匹配图的输出目录。
    const fs::path structureDir = _config.output_dir / "structures";
    const fs::path matchesDir = _config.output_dir / "matches";
    std::error_code ec;
    fs::create_directories(structureDir, ec);
    fs::create_directories(matchesDir, ec);

    const std::string stem = buildOutputStem(ctx);
    const std::string sampleStem = ctx.image1_path.stem().string() + "_" +
                                   ctx.image2_path.stem().string();
    const std::string structureStem =
        sampleStem + "_" + (_extractor ? _extractor->outputLabel() : std::string("STRUCTURE"));

    // 2. 保存结构提取器生成的 source / target 响应图。
    if (!ctx.structure_data.first.response.empty()) {
        const fs::path out = structureDir / (structureStem + "_source_structure.png");
        cv::imwrite(out.string(), ctx.structure_data.first.response);
        IR_LOG_INFO("Wrote source structure visualization: ", out.string());
    }
    if (!ctx.structure_data.second.response.empty()) {
        const fs::path out = structureDir / (structureStem + "_target_structure.png");
        cv::imwrite(out.string(), ctx.structure_data.second.response);
        IR_LOG_INFO("Wrote target structure visualization: ", out.string());
    }

    // 3. 按配置保存结构匹配连线图，便于和点特征 matches 输出对照。
    if (_config.draw_matches) {
        cv::Mat vis;
        const std::vector<cv::DMatch>& preferredMatches =
            !ctx.structure_match_data.inlier_line_matches.empty()
                ? ctx.structure_match_data.inlier_line_matches
                : ctx.structure_match_data.line_matches;

        if (ctx.structure_data.type == StructureType::LINE && !preferredMatches.empty()) {
            vis = renderLineSegmentMatches(ctx, preferredMatches, _config.max_matches_drawn);
        } else if (ctx.structure_data.type == StructureType::CONTOUR &&
                   !preferredMatches.empty()) {
            vis = renderContourMatches(ctx, preferredMatches, _config.max_matches_drawn);
        }
        if (vis.empty()) {
            vis = renderStructureMatches(ctx, _config.max_matches_drawn);
        }
        if (!vis.empty()) {
            const fs::path out = matchesDir / (stem + "_structure_matches.png");
            if (cv::imwrite(out.string(), vis)) {
                IR_LOG_INFO("Wrote structure matches visualization: ", out.string());
            } else {
                IR_LOG_WARN("Failed to write structure matches visualization: ", out.string());
            }
        } else {
            IR_LOG_WARN("Structure matches visualization skipped: no drawable correspondences.");
        }
    }

    // 4. 委托基类保存 originals / warped / blend 等通用输出。
    return BasePipeline::saveOutputs(ctx);
}

} // namespace ir
