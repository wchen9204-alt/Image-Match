#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <string>
#include <vector>

#include "core/context.h"

namespace ir {

/// 对应点来源类型。消费者只依赖统一视图，不直接关心点对来自哪个方法族。
enum class CorrespondenceSource {
    NONE,
    KEYPOINT,
    STRUCTURE,
    DIRECT,
    LEARNING
};

/// 将来源枚举转为日志/指标说明中的可读名称。
const char* toString(CorrespondenceSource source);

/// 从字符串解析对应点来源；未知时返回 NONE。
CorrespondenceSource correspondenceSourceFromString(const std::string& source);

/// 从上下文中解析当前对应点来源；优先读取显式来源，其次读取几何阶段记录。
CorrespondenceSource correspondenceSourceFromContext(const RegistrationContext& ctx);

/// 通用对应点视图，用于评估指标和可视化读取“可投影的点对”。
/// 该结构是只读快照，不回写 RegistrationContext，避免不同方法族互相伪装数据。
struct CorrespondenceView {
    /// 点对来源，用于日志和后续扩展分发。
    CorrespondenceSource source = CorrespondenceSource::NONE;

    /// 来源名称，便于指标 note 中说明当前采用的数据源。
    std::string source_name;

    /// 源图对应点，统一转换为 OpenCV keypoint 以复用 drawMatches。
    std::vector<cv::KeyPoint> first_keypoints;

    /// 目标图对应点，索引与 first_keypoints 对齐。
    std::vector<cv::KeyPoint> second_keypoints;

    /// 过滤后的候选对应关系。
    std::vector<cv::DMatch> filtered;

    /// 内点掩码，与 filtered 按索引对应；为空时表示调用方可按场景使用全部 filtered。
    std::vector<unsigned char> inlier_mask;

    /// 根据来源内点或 inlier_mask 推导出的内点对应关系。
    std::vector<cv::DMatch> inliers;

    bool empty() const { return filtered.empty(); }
    int filteredCount() const { return static_cast<int>(filtered.size()); }
    int inlierCount() const { return static_cast<int>(inliers.size()); }
};

/// 从点特征上下文构造对应点视图。
CorrespondenceView buildKeypointCorrespondenceView(const RegistrationContext& ctx);

/// 从深度学习上下文构造对应点视图；输入点对来自学习模型输出的显式转换结果。
CorrespondenceView buildLearningCorrespondenceView(const RegistrationContext& ctx);

/// 从直接法上下文构造对应点视图。
CorrespondenceView buildDirectCorrespondenceView(const RegistrationContext& ctx);

/// 从结构法上下文构造对应点视图。
CorrespondenceView buildStructureCorrespondenceView(const RegistrationContext& ctx);

/// 按显式来源构造对应点视图。
CorrespondenceView buildCorrespondenceView(const RegistrationContext& ctx,
                                           CorrespondenceSource source);

/// 兼容入口：在来源未知或未显式指定时，按当前上下文中实际可用的数据选择视图。
CorrespondenceView buildBestCorrespondenceView(const RegistrationContext& ctx);

} // namespace ir
