#include "direct/sparse/klt_sparse_aligner.h"

#include <algorithm>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "core/types.h"
#include "direct/common/direct_geometry_common.h"
#include "geometry/partial_affine_utils.h"
#include "utils/logger.h"
#include "utils/yaml_utils.h"

namespace ir {

namespace {

using direct_geometry_common::RobustFitOptions;

// 功能：在指定 ROI 内提取角点，并把局部坐标换回整图坐标。
// 作用：供网格化角点检测复用，兼顾局部纹理覆盖和后续统一的整图点坐标表示。
void appendCornersFromRegion(const cv::Mat& gray,
                             const cv::Rect& roi,
                             int maxCorners,
                             double qualityLevel,
                             double minDistance,
                             int blockSize,
                             std::vector<cv::Point2f>& corners) {
    if (roi.width <= 0 || roi.height <= 0 || maxCorners <= 0) {
        return;
    }

    std::vector<cv::Point2f> localCorners;
    cv::goodFeaturesToTrack(gray(roi),
                            localCorners,
                            std::max(1, maxCorners),
                            qualityLevel,
                            minDistance,
                            cv::Mat(),
                            std::max(3, blockSize));
    for (const cv::Point2f& local : localCorners) {
        corners.emplace_back(local.x + static_cast<float>(roi.x), local.y + static_cast<float>(roi.y));
    }
}

// 功能：按整图或网格方式提取稀疏角点，尽量兼顾纹理强度与空间覆盖。
static void detectCorners(const cv::Mat& gray,
                          int maxCorners,
                          double qualityLevel,
                          double minDistance,
                          int blockSize,
                          bool gridEnabled,
                          int gridRows,
                          int gridCols,
                          std::vector<cv::Point2f>& corners) {
    corners.clear();
    if (gray.empty()) {
        return;
    }

    if (!gridEnabled) {
        cv::goodFeaturesToTrack(gray,
                                corners,
                                std::max(1, maxCorners),
                                qualityLevel,
                                minDistance,
                                cv::Mat(),
                                std::max(3, blockSize));
        return;
    }

    const int rows = std::max(1, gridRows);
    const int cols = std::max(1, gridCols);
    const int perCell = std::max(1, (std::max(1, maxCorners) + rows * cols - 1) / (rows * cols));
    std::vector<std::vector<cv::Point2f>> cellCorners;
    cellCorners.reserve(static_cast<size_t>(rows * cols));

    for (int row = 0; row < rows; ++row) {
        const int y0 = row * gray.rows / rows;
        const int y1 = (row + 1) * gray.rows / rows;
        for (int col = 0; col < cols; ++col) {
            const int x0 = col * gray.cols / cols;
            const int x1 = (col + 1) * gray.cols / cols;
            std::vector<cv::Point2f> local;
            local.reserve(static_cast<size_t>(perCell));
            appendCornersFromRegion(gray,
                                    cv::Rect(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0)),
                                    perCell,
                                    qualityLevel,
                                    minDistance,
                                    blockSize,
                                    local);
            cellCorners.push_back(std::move(local));
        }
    }

    corners.reserve(static_cast<size_t>(std::max(1, maxCorners)));
    for (size_t depth = 0; corners.size() < static_cast<size_t>(std::max(1, maxCorners)); ++depth) {
        bool appended = false;
        for (const auto& local : cellCorners) {
            if (depth >= local.size()) {
                continue;
            }
            corners.push_back(local[depth]);
            appended = true;
            if (corners.size() >= static_cast<size_t>(std::max(1, maxCorners))) {
                break;
            }
        }
        if (!appended) {
            break;
        }
    }
}

// 功能：对已检测角点做亚像素精修。
// 作用：降低整数像素角点位置的量化误差，让后续金字塔 LK 跟踪初值更稳定。
void refineCornersSubpix(const cv::Mat& gray,
                         int windowSize,
                         int maxIterations,
                         double epsilon,
                         std::vector<cv::Point2f>& corners) {
    if (gray.empty() || corners.empty()) {
        return;
    }

    const int win = std::max(1, windowSize | 1);
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                    std::max(1, maxIterations),
                                    std::max(0.0, epsilon));
    cv::cornerSubPix(gray,
                     corners,
                     cv::Size(win, win),
                     cv::Size(-1, -1),
                     criteria);
}

} // namespace

KltSparseAligner::KltSparseAligner(const YAML::Node& cfg) {
    const YAML::Node params = cfg["params"] ? cfg["params"] : cfg;
    // 兼容算法配置文件中直接写参数或写在 params 节点下两种形式。
    _maxCorners = yaml_utils::getInt(params, "max_corners", 500);
    _qualityLevel = yaml_utils::getDouble(params, "quality_level", 0.01);
    _minDistance = yaml_utils::getDouble(params, "min_distance", 8.0);
    _blockSize = yaml_utils::getInt(params, "block_size", 7);
    _windowSize = yaml_utils::getInt(params, "window_size", 21);
    _maxLevel = yaml_utils::getInt(params, "max_level", 3);
    _maxIterations = yaml_utils::getInt(params, "max_iterations", 30);
    _epsilon = yaml_utils::getDouble(params, "epsilon", 0.01);
    _gridEnabled = yaml_utils::getBool(params, "grid_enabled", true);
    _gridRows = yaml_utils::getInt(params, "grid_rows", 4);
    _gridCols = yaml_utils::getInt(params, "grid_cols", 4);
    _subpixRefine = yaml_utils::getBool(params, "subpix_refine", true);
    _subpixWindowSize = yaml_utils::getInt(params, "subpix_window_size", 5);
    _subpixMaxIterations = yaml_utils::getInt(params, "subpix_max_iterations", 20);
    _subpixEpsilon = yaml_utils::getDouble(params, "subpix_epsilon", 0.03);
    _forwardBackwardCheck = yaml_utils::getBool(params, "forward_backward_check", true);
    _fbThreshold = yaml_utils::getDouble(params, "fb_threshold", 1.5);
    _borderMargin = yaml_utils::getInt(params, "border_margin", 3);
    _maxTrackError = yaml_utils::getDouble(params, "max_track_error", -1.0);
    _ransacThreshold = yaml_utils::getDouble(params, "ransac_threshold", 3.0);
    _robustMethod = yaml_utils::getString(params, "robust_method", "RANSAC");
    _ransacMaxIters = yaml_utils::getInt(params, "ransac_max_iters", 2000);
    _ransacConfidence = yaml_utils::getDouble(params, "ransac_confidence", 0.99);
    _ransacRefineIters = yaml_utils::getInt(params, "ransac_refine_iters", 10);
    _minInliers = yaml_utils::getInt(params, "min_inliers", 6);
    _fitModel = yaml_utils::getString(params, "fit_model", "RIGID");
}

bool KltSparseAligner::align(RegistrationContext& ctx) {
    auto& dd = ctx.direct_data;
    auto& gd = ctx.geometry_data;
    dd.clear();
    gd.clear();
    dd.method = name();

    // 1. 检查输入灰度图，并在源图上挑选可跟踪角点作为稀疏 LK 种子。
    if (ctx.images.first_gray.empty() || ctx.images.second_gray.empty()) {
        dd.message = "KLT requires non-empty grayscale images";
        gd.message = dd.message;
        return false;
    }
    if (ctx.images.first_gray.size() != ctx.images.second_gray.size()) {
        dd.message = "KLT requires grayscale images with the same size";
        gd.message = dd.message;
        return false;
    }

    // 直接法入口只使用灰度图；先在源图选取稳定角点，作为 LK 光流的跟踪种子。
    std::vector<cv::Point2f> corners;
    detectCorners(ctx.images.first_gray,
                  _maxCorners,
                  _qualityLevel,
                  _minDistance,
                  _blockSize,
                  _gridEnabled,
                  _gridRows,
                  _gridCols,
                  corners);
    if (corners.empty()) {
        dd.message = "KLT found no trackable corners";
        gd.message = dd.message;
        return false;
    }

    if (_subpixRefine) {
        // 角点先做亚像素精修，减少后续 LK 初始位置的量化误差。
        refineCornersSubpix(ctx.images.first_gray,
                            _subpixWindowSize,
                            _subpixMaxIterations,
                            _subpixEpsilon,
                            corners);
    }

    // 2. 用金字塔 LK 从源图跟踪到目标图，并可选执行前后向一致性检查。
    std::vector<cv::Point2f> tracked;
    std::vector<unsigned char> status;
    std::vector<float> error;
    const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                    std::max(1, _maxIterations),
                                    _epsilon);
    const int win = std::max(3, _windowSize | 1);
    // 金字塔 LK 从源图角点直接跟踪到目标图，得到稀疏光流点对。
    cv::calcOpticalFlowPyrLK(ctx.images.first_gray,
                             ctx.images.second_gray,
                             corners,
                             tracked,
                             status,
                             error,
                             cv::Size(win, win),
                             std::max(0, _maxLevel),
                             criteria);

    std::vector<cv::Point2f> backTracked;
    std::vector<unsigned char> backStatus;
    std::vector<float> backError;
    if (_forwardBackwardCheck) {
        // 反向跟踪把 second->first 的结果投回去，只保留往返一致的稀疏点。
        cv::calcOpticalFlowPyrLK(ctx.images.second_gray,
                                 ctx.images.first_gray,
                                 tracked,
                                 backTracked,
                                 backStatus,
                                 backError,
                                 cv::Size(win, win),
                                 std::max(0, _maxLevel),
                                 criteria);
    }

    // 3. 整理通过检查的稀疏点对，并剔除越界、高误差或往返不一致的跟踪结果。
    // 将 KLT 成功跟踪的角点整理为全局几何估计点对；失败点不参与后续 RANSAC。
    std::vector<cv::Point2f> srcPts;
    std::vector<cv::Point2f> dstPts;
    srcPts.reserve(corners.size());
    dstPts.reserve(corners.size());
    dd.matches.clear();
    int trackedCount = 0;
    int fbPassedCount = 0;
    for (size_t i = 0; i < corners.size() && i < tracked.size() && i < status.size(); ++i) {
        if (!status[i]) {
            continue;
        }
        ++trackedCount;

        if (!direct_geometry_common::isPointInsideWithMargin(
                tracked[i], ctx.images.second_gray.size(), _borderMargin)) {
            continue;
        }
        if (_maxTrackError >= 0.0 && i < error.size() && error[i] > _maxTrackError) {
            continue;
        }
        if (_forwardBackwardCheck) {
            if (i >= backTracked.size() || i >= backStatus.size() || !backStatus[i]) {
                continue;
            }
            if (cv::norm(backTracked[i] - corners[i]) > std::max(0.0, _fbThreshold)) {
                continue;
            }
            ++fbPassedCount;
        }

        srcPts.push_back(corners[i]);
        dstPts.push_back(tracked[i]);
        const int idx = static_cast<int>(dd.matches.size());
        dd.matches.emplace_back(idx, idx, i < error.size() ? error[i] : 0.0f);
    }

    if (srcPts.empty()) {
        dd.message = "KLT tracked no valid points";
        gd.message = dd.message;
        return false;
    }

    // 4. 在有效跟踪点对上估计全局变换，并用内点数阈值过滤不可靠结果。
    cv::Mat A;
    cv::Mat H;
    std::vector<unsigned char> mask;
    GeometryType type = GeometryType::UNKNOWN;
    const RobustFitOptions fitOptions{
        _robustMethod, _ransacThreshold, _ransacMaxIters, _ransacConfidence, _ransacRefineIters};
    // 从跟踪点对估计全局变换，RIGID 配置会在 RANSAC 内点上强制回归无缩放矩阵。
    if (!direct_geometry_common::fitGlobalTransform(srcPts,
                                                    dstPts,
                                                    _fitModel,
                                                    fitOptions,
                                                    A,
                                                    H,
                                                    mask,
                                                    type,
                                                    "KLT sparse direct")) {
        dd.message = "KLT failed to fit global transform";
        gd.message = dd.message;
        return false;
    }

    int inliers = 0;
    for (unsigned char v : mask) {
        if (v) {
            ++inliers;
        }
    }
    if (inliers < _minInliers) {
        dd.message = "KLT global transform has " + std::to_string(inliers) +
                     " inliers, below min_inliers=" + std::to_string(_minInliers);
        gd.message = dd.message;
        IR_LOG_WARN("KLT rejected: ", dd.message);
        return false;
    }

    // 5. 同时写回 direct_data 与 geometry_data，供统一的直接法可视化和 warp 复用。
    // 同时写入 direct_data 与 geometry_data：前者用于直接法可视化，后者用于统一 warp/blend 输出。
    dd.points1 = srcPts;
    dd.points2 = dstPts;
    dd.inlier_mask = mask;
    dd.valid = true;
    dd.score = srcPts.empty() ? 0.0 : static_cast<double>(inliers) / srcPts.size();

    gd.type = type;
    gd.valid = true;
    gd.num_inliers = inliers;
    gd.inlier_ratio = dd.score;
    gd.inlier_mask = mask;
    gd.correspondence_source = "DIRECT";
    gd.num_correspondences = static_cast<int>(srcPts.size());
    if (type == GeometryType::HOMOGRAPHY) {
        H.convertTo(gd.H, CV_64F);
        dd.H = gd.H.clone();
    } else {
        A.convertTo(gd.A, CV_64F);
        dd.A = gd.A.clone();
    }

    IR_LOG_INFO("KLT sparse direct corners=",
                corners.size(),
                ", tracked=",
                trackedCount,
                ", fb_passed=",
                (_forwardBackwardCheck ? fbPassedCount : static_cast<int>(srcPts.size())),
                ", fitted=",
                srcPts.size(),
                ", inliers=",
                inliers);
    return true;
}

} // namespace ir

