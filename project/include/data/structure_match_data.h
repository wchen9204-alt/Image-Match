#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace ir {

/// 结构关联/匹配阶段的中间结果。
struct StructureMatchData {
    /// 当前使用的结构关联方法名称。
    std::string method;

    /// 结构关联得到的平移量。
    cv::Point2d translation{0.0, 0.0};

    /// 结构关联直接估计出的 2x3 仿射矩阵；为空时由 translation 退化生成平移矩阵。
    cv::Mat affine;

    /// 结构关联得分。
    double score = 0.0;

    /// 结构关联结果是否有效。
    bool valid = false;

    /// 结构关联阶段的详细状态信息。
    std::string message;

    /// 关联器输出的原始 KNN 匹配（每行的邻居列表）；供 RatioTest 等过滤器使用。
    std::vector<std::vector<cv::DMatch>> raw_matches_knn;

    /// 过滤链的工作输出；管道运行时由 filtered_matches 同步回 line_matches。
    std::vector<cv::DMatch> filtered_matches;

    /// 线段级候选匹配，queryIdx 对应 source lines，trainIdx 对应 target lines。
    std::vector<cv::DMatch> line_matches;

    /// 线段级内点匹配，用于线特征法的可视化和统计。
    std::vector<cv::DMatch> inlier_line_matches;

    /// 清空结构关联结果。
    void clear() {
        method.clear();
        translation = cv::Point2d(0.0, 0.0);
        affine.release();
        score = 0.0;
        valid = false;
        message.clear();
        raw_matches_knn.clear();
        filtered_matches.clear();
        line_matches.clear();
        inlier_line_matches.clear();
    }
};

} // namespace ir
