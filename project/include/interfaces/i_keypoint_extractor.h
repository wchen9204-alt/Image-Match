#pragma once

#include <string>

#include <opencv2/features2d.hpp>

#include "core/context.h"
#include "core/types.h"

namespace ir {

/// 点特征提取器接口。
///
/// 负责在输入图像上提取关键点并计算描述子，结果写入 `RegistrationContext`。
class IKeypointExtractor {
public:
    virtual ~IKeypointExtractor() = default;

    /// 返回当前提取器名称，用于日志与输出命名。
    virtual std::string name() const = 0;

    /// 返回当前提取器对应的关键点类型。
    virtual KeypointType type() const = 0;

    /// 返回当前描述子匹配默认使用的距离类型。
    virtual NormType normType() const = 0;

    /// 返回当前算法对象，供多层暗部流程分别执行 detect 与 compute。
    virtual cv::Ptr<cv::Feature2D> feature2D() const = 0;

    /// 执行关键点检测与描述子计算。
    virtual bool extract(RegistrationContext& ctx) = 0;
};

} // namespace ir
