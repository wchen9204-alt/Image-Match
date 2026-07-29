#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <span>
#include <string>
#include <string_view>
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

/// 通用对应点轻量视图，用于评估指标和可视化读取“可投影的点对”。
/// 所有字段只借用原始数据，不复制关键点、匹配或内点掩码。
struct CorrespondenceView {
    /// 点对来源，用于日志和后续扩展分发。
    CorrespondenceSource source = CorrespondenceSource::NONE;

    /// 来源名称，便于指标 note 中说明当前采用的数据源。
    std::string_view source_name;

    /// 源图对应点，统一转换为 OpenCV keypoint 以复用 drawMatches。
    std::span<const cv::KeyPoint> first_keypoints;

    /// 目标图对应点，索引与 first_keypoints 对齐。
    std::span<const cv::KeyPoint> second_keypoints;

    /// OpenCV drawMatches 只接受 vector 引用；该指针指向与上述 span 相同的数据，绝不复制。
    const std::vector<cv::KeyPoint>* first_keypoints_storage = nullptr;
    const std::vector<cv::KeyPoint>* second_keypoints_storage = nullptr;

    /// 匹配器原始输出的候选对应关系，未经过过滤链筛选。
    std::span<const cv::DMatch> raw;

    /// 过滤后的候选对应关系。
    std::span<const cv::DMatch> filtered;

    /// 几何估计阶段输出的内点掩码，与 filtered 按索引对应。
    std::span<const unsigned char> inlier_mask;

    /// 几何阶段最终确认并写回的内点对应关系。
    std::span<const cv::DMatch> inliers;

    bool empty() const { return raw.empty() && filtered.empty(); }
    int filteredCount() const { return static_cast<int>(filtered.size()); }
    int inlierCount() const { return static_cast<int>(inliers.size()); }
};

/// 统一对应点快照。
/// KEYPOINT/LEARNING 仅保存 span 借用上下文；DIRECT/STRUCTURE 在此保存必要的转换结果，
/// 以保证返回的 CorrespondenceView 在当前 RegistrationContext 生命周期内始终有效。
struct CorrespondenceSnapshot {
    CorrespondenceSource source = CorrespondenceSource::NONE;
    std::string source_name = "NONE";

    /// true 时视图从 owned_* 读取；false 时从 borrowed_* 读取。
    bool owns_storage = false;

    /// DIRECT/STRUCTURE 没有独立 raw 匹配时，按原有语义将 filtered 作为 raw 读取。
    bool raw_falls_back_to_filtered = false;

    std::span<const cv::KeyPoint> borrowed_first_keypoints;
    std::span<const cv::KeyPoint> borrowed_second_keypoints;
    const std::vector<cv::KeyPoint>* borrowed_first_keypoints_storage = nullptr;
    const std::vector<cv::KeyPoint>* borrowed_second_keypoints_storage = nullptr;
    std::span<const cv::DMatch> borrowed_raw;
    std::span<const cv::DMatch> borrowed_filtered;
    std::span<const unsigned char> borrowed_inlier_mask;
    std::span<const cv::DMatch> borrowed_inliers;

    std::vector<cv::KeyPoint> owned_first_keypoints;
    std::vector<cv::KeyPoint> owned_second_keypoints;
    std::vector<cv::DMatch> owned_raw;
    std::vector<cv::DMatch> owned_filtered;
    std::vector<unsigned char> owned_inlier_mask;
    std::vector<cv::DMatch> owned_inliers;

    /// 将快照转换为仅含 span 的轻量读取视图。
    CorrespondenceView view() const;
    bool empty() const { return view().empty(); }
};

/// 从点特征上下文构造对应点快照；结果仅借用上下文中的数据。
CorrespondenceSnapshot buildKeypointCorrespondenceSnapshot(const RegistrationContext& ctx);

/// 从深度学习上下文构造对应点快照；输入点对来自学习模型输出的显式转换结果。
CorrespondenceSnapshot buildLearningCorrespondenceSnapshot(const RegistrationContext& ctx);

/// 从直接法上下文构造对应点快照。
CorrespondenceSnapshot buildDirectCorrespondenceSnapshot(const RegistrationContext& ctx);

/// 从结构法上下文构造对应点快照。
CorrespondenceSnapshot buildStructureCorrespondenceSnapshot(const RegistrationContext& ctx);

/// 按显式来源构造对应点快照。
CorrespondenceSnapshot buildCorrespondenceSnapshot(const RegistrationContext& ctx,
                                                    CorrespondenceSource source);

/// 在来源未知或未显式指定时，按当前上下文中实际可用的数据选择快照。
CorrespondenceSnapshot buildBestCorrespondenceSnapshot(const RegistrationContext& ctx);

/// 强制依据当前上下文刷新共享快照；应在几何估计前后调用，后者会纳入最新内点。
void refreshCorrespondenceSnapshot(RegistrationContext& ctx);

/// 为几何估计取得当前快照。缓存缺失或来源变化时自动构建一次。
CorrespondenceView ensureCorrespondenceView(RegistrationContext& ctx);

/// 为评测和可视化取得已缓存的轻量视图；缓存缺失时返回空视图，不额外复制数据。
CorrespondenceView cachedCorrespondenceView(const RegistrationContext& ctx);
} // namespace ir


