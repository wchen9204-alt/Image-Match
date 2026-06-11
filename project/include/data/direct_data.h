#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace ir {

/// 直接法算法附带的数值诊断项。
/// key 面向 JSON/CSV 等机器读取输出，label 面向终端和文本摘要显示。
struct DirectDiagnostic {
    std::string key;
    std::string label;
    double value = 0.0;
};

/// 直接法配准阶段的统一输出数据。
/// 该结构既承载直接法自身结果，也作为 DirectPipeline 向通用几何/可视化阶段传递数据的桥梁。
struct DirectData {
    /// 当前直接法算法名称，用于日志、摘要和输出文件命名。
    std::string method;

    /// 当前直接法阶段是否产出了可用结果。
    bool valid = false;

    /// 失败原因或回退说明；成功时通常为空。
    std::string message;

    /// 2x3 仿射族矩阵，可能表示平移、刚体、相似或完整仿射，具体语义由 geometry_data.type 标识。
    cv::Mat A;

    /// 3x3 单应矩阵，仅在 HOMOGRAPHY 模型下写入。
    cv::Mat H;

    /// 稠密光流场，类型通常为 CV_32FC2；由 Farneback / DIS 等稠密直接法写入。
    cv::Mat flow;

    /// 源图上的直接法点对起点；KLT 为跟踪角点，稠密光流方法为采样像素点。
    std::vector<cv::Point2f> points1;

    /// 目标图上的对应点；与 points1 按索引一一对应。
    std::vector<cv::Point2f> points2;

    /// 用于复用通用匹配可视化/统计流程的伪匹配，queryIdx/trainIdx 对应 points1/points2 索引。
    std::vector<cv::DMatch> matches;

    /// 直接法全局几何估计得到的内点掩码，与 points1/points2/matches 按索引对应。
    std::vector<unsigned char> inlier_mask;

    /// 算法置信度。KLT/稠密光流方法通常为内点比例，ECC/相位相关为 OpenCV 返回的响应或相关得分。
    double score = 0.0;

    /// 直接法专属诊断指标列表。
    /// 例如相位相关可写入峰值比和亚像素置信度，Fourier-Mellin 可写入旋转/尺度响应等。
    std::vector<DirectDiagnostic> diagnostics;

    /// 光度误差预留字段；未计算时保持 -1。
    double photometric_error = -1.0;

    /// 清空直接法阶段缓存，确保同一个上下文复用时不会带入上一轮结果。
    void clear() {
        method.clear();
        valid = false;
        message.clear();
        A.release();
        H.release();
        flow.release();
        points1.clear();
        points2.clear();
        matches.clear();
        inlier_mask.clear();
        score = 0.0;
        diagnostics.clear();
        photometric_error = -1.0;
    }

    /// 追加一个直接法诊断指标；label 为空时使用 key 作为文本显示名。
    void addDiagnostic(std::string key, std::string label, double value) {
        if (label.empty()) {
            label = key;
        }
        diagnostics.push_back({std::move(key), std::move(label), value});
    }
};

} // namespace ir
