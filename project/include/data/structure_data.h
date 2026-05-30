#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "core/types.h"

namespace ir {

/// 单张图像对应的结构特征数据。
struct StructureImageData {
    /// 用于配准估计和可视化的二值或灰度结构响应图。
    cv::Mat mask;

    /// 直线提取器输出的线段集合，格式为 `(x1, y1, x2, y2)`。
    std::vector<cv::Vec4i> lines;

    /// 轮廓提取器输出的轮廓点集。
    std::vector<std::vector<cv::Point>> contours;

    /// 清空当前图像的结构特征数据。
    void clear() {
        mask.release();
        lines.clear();
        contours.clear();
    }

    /// 判断当前图像是否没有任何有效结构特征。
    bool empty() const { return mask.empty() && lines.empty() && contours.empty(); }

    /// 按结构类型返回便于统计和摘要展示的数量。
    int primitiveCount(StructureType type) const {
        switch (type) {
        case StructureType::LINE:
            return static_cast<int>(lines.size());
        case StructureType::CONTOUR:
            return static_cast<int>(contours.size());
        case StructureType::EDGE:
            return mask.empty() ? 0 : cv::countNonZero(mask);
        case StructureType::UNKNOWN:
        default:
            return 0;
        }
    }
};

/// 一对待配准图像的结构特征数据。
struct StructureData {
    StructureImageData first;
    StructureImageData second;

    /// 本次结构提取使用的方法类型。
    StructureType type = StructureType::UNKNOWN;

    /// 清空两张图像的结构特征数据和类型信息。
    void clear() {
        first.clear();
        second.clear();
        type = StructureType::UNKNOWN;
    }

    /// 判断是否至少有一张图像缺少有效结构特征。
    bool empty() const { return first.empty() || second.empty(); }
};

} // namespace ir
