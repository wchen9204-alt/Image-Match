#pragma once

#include <filesystem>
#include <string>

#include <opencv2/core.hpp>

#include "core/context.h"

namespace ir::direct_pipeline_helpers {

/// 判断某个直接法是否会在算法内部自行消费点特征初始值。
bool alignerConsumesFeatureInitializerInternally(const std::string& alignerName);

/// 删除当前样本遗留的直接法专属可视化输出，避免上一轮结果残留。
void removeStaleDirectVisualization(const std::filesystem::path& path);

/// 将点特征初始值阶段的诊断信息同步到最终 RegistrationResult。
void syncFeatureInitializerDiagnostics(RegistrationContext& ctx);

/// 对还不会自行消费初始值的直接法，先把 source warp 到初始位姿，再让算法估计残差。
bool applyFeatureInitializerPrewarp(RegistrationContext& ctx,
                                    const std::string& alignerName,
                                    cv::Mat& initializerMatrix,
                                    cv::Mat& originalColor,
                                    cv::Mat& originalGray,
                                    bool& applied);

/// 将“点特征初始值 + 直接法残差”的两段变换合成为最终 source -> target 结果。
bool mergeFeatureInitializerAndDirectResult(RegistrationContext& ctx,
                                            const cv::Mat& initializerMatrix);

/// 基于点特征初始值单独生成一张 warp 后的 source 图，用于 false-color overlay 输出。
bool buildInitializerWarpedSource(const RegistrationContext& ctx, cv::Mat& warped);

/// 将最终判定采用的配准结果写入 warped_image，供 warped、blend 和通用 false 图统一输出。
/// 当最终来源为 INITIALIZER 时，重建点特征 warp；否则保留直接法的 warped_image。
bool applyFinalSelectedWarpedImage(RegistrationContext& ctx);

/// 基于最终选中的 warped_image 生成 direct 最终对外展示的 false-color overlay。
/// warped_image 已在通用输出前按最终判定来源统一为 initializer 或 direct 结果。
bool buildFinalSelectedFalseColorOverlay(const RegistrationContext& ctx,
                                         int foregroundThreshold,
                                         cv::Mat& overlay);

} // namespace ir::direct_pipeline_helpers
