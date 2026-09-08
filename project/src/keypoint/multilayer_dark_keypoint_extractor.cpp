#include "keypoint/multilayer_dark_keypoint_extractor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "core/factory.h"
#include "utils/image_utils.h"
#include "utils/logger.h"

namespace ir {

namespace {

cv::Mat buildDarkFeatureLayer(const cv::Mat& gray, int threshold);

/// 保存直方图局部峰在合并前的灰度范围、面积和峰高信息。
struct RawGrayDensityPeak {
    int peak_gray = 0;
    int left_peak_gray = 0;
    int right_peak_gray = 0;
    int left_gray = 0;
    int right_gray = 0;
    int span = 0;
    int area = 0;
    double peak_height = 0.0;
};

/// 保存合并后的可靠峰及其最终灰度区间和面积信息。
struct GrayDensityPeak {
    int peak_gray = 0;
    int left_gray = 0;
    int right_gray = 0;
    int span = 0;
    int area = 0;
    double peak_height = 0.0;
    double valley_height = 0.0;
    int merge_group = 0;
    int merge_source_begin = 0;
    int merge_source_end = 0;
    int merged_peak_count = 1;
    std::string merge_reason;
};

/// 保存单张图的选层模式、峰诊断信息和待检测图层。
struct LayerPlan {
    std::vector<cv::Mat> layers;
    std::vector<GrayDensityPeak> selected_peaks;
    std::vector<GrayDensityPeak> all_peaks;
    std::string mode = "fixed10";
};

constexpr int kMinimumPeakDistance = 15;
constexpr double kPeakHeightRatioForMerge = 0.10;
constexpr double kMinimumTop3AreaRatio = 0.95;

/// 返回灰度计数数组指定区间内的最小值，用于寻找相邻峰之间的谷底。
double rangeMinimum(const std::vector<double>& values, int left, int right) {
    if (left > right) {
        std::swap(left, right);
    }
    double result = values[static_cast<size_t>(left)];
    for (int index = left + 1; index <= right; ++index) {
        result = std::min(result, values[static_cast<size_t>(index)]);
    }
    return result;
}

/// 返回灰度计数数组指定区间内最小值所在的灰度索引。
int rangeMinimumIndex(const std::vector<double>& values, int left, int right) {
    int result = left;
    for (int index = left + 1; index <= right; ++index) {
        if (values[static_cast<size_t>(index)] < values[static_cast<size_t>(result)]) {
            result = index;
        }
    }
    return result;
}

/// 统计单张灰度图的原始局部峰，并计算每个原始峰的初始灰度区间和面积。
std::vector<RawGrayDensityPeak> analyzeRawGrayDensityPeaks(
    const cv::Mat& gray,
    std::vector<double>* density_output = nullptr) {
    std::array<int, 256> histogram{};
    int first_gray = 255;
    int last_gray = 0;
    for (int row = 0; row < gray.rows; ++row) {
        for (int column = 0; column < gray.cols; ++column) {
            const int value = gray.at<unsigned char>(row, column);
            if (value == 0) {
                continue;
            }
            ++histogram[static_cast<size_t>(value)];
            first_gray = std::min(first_gray, value);
            last_gray = std::max(last_gray, value);
        }
    }
    if (first_gray > last_gray) {
        return {};
    }

    const std::vector<double> density(histogram.begin(), histogram.end());
    if (density_output != nullptr) {
        *density_output = density;
    }

    std::vector<std::pair<int, double>> candidates;
    for (int gray_value = std::max(1, first_gray);
         gray_value <= std::min(254, last_gray); ++gray_value) {
        const double value = density[static_cast<size_t>(gray_value)];
        if (value > 0.0 &&
            value >= density[static_cast<size_t>(gray_value - 1)] &&
            value >= density[static_cast<size_t>(gray_value + 1)]) {
            candidates.emplace_back(gray_value, value);
        }
    }

    // 将同一段平坦峰顶的多个候选灰度合并为一个候选峰，代表点取平台中部。
    std::vector<std::pair<int, double>> plateau_candidates;
    for (size_t begin = 0; begin < candidates.size();) {
        size_t end = begin;
        while (end + 1 < candidates.size() &&
               candidates[end + 1].first == candidates[end].first + 1) {
            ++end;
        }
        size_t representative = begin;
        for (size_t index = begin + 1; index <= end; ++index) {
            if (candidates[index].second > candidates[representative].second) {
                representative = index;
            }
        }
        plateau_candidates.emplace_back(
            candidates[(begin + end) / 2].first, candidates[representative].second);
        begin = end + 1;
    }
    candidates = std::move(plateau_candidates);

    std::vector<RawGrayDensityPeak> peaks;
    peaks.reserve(candidates.size());
    for (size_t index = 0; index < candidates.size(); ++index) {
        const int peak = candidates[index].first;
        const int left_peak = index == 0 ? first_gray : candidates[index - 1].first;
        const int right_peak = index + 1 == candidates.size()
            ? last_gray
            : candidates[index + 1].first;
        peaks.push_back({peak, left_peak, right_peak, 0, 0, 0, 0, candidates[index].second});
    }
    for (size_t index = 0; index < peaks.size(); ++index) {
        auto& peak = peaks[index];
        peak.left_gray = index == 0
            ? first_gray
            : rangeMinimumIndex(density, peaks[index - 1].peak_gray, peak.peak_gray);
        peak.right_gray = index + 1 == peaks.size()
            ? last_gray
            : rangeMinimumIndex(density, peak.peak_gray, peaks[index + 1].peak_gray);
        peak.span = peak.right_gray - peak.left_gray + 1;
        for (int row = 0; row < gray.rows; ++row) {
            for (int column = 0; column < gray.cols; ++column) {
                const int value = gray.at<unsigned char>(row, column);
                if (value > 0 && value >= peak.left_gray && value <= peak.right_gray) {
                    ++peak.area;
                }
            }
        }
    }
    return peaks;
}

/// 按最高峰锚点合并相近原始峰，生成供图层选择使用的可靠峰集合。
std::vector<GrayDensityPeak> analyzeGrayDensityPeaks(
    const cv::Mat& gray,
    std::vector<RawGrayDensityPeak>* raw_output) {
    // 1. 先保留未合并的原始峰，同时保留完整灰度计数曲线供谷底计算。
    std::vector<double> density;
    const auto raw_peaks = analyzeRawGrayDensityPeaks(gray, &density);
    if (raw_output != nullptr) {
        *raw_output = raw_peaks;
    }

    std::vector<GrayDensityPeak> reliable;
    reliable.reserve(raw_peaks.size());
    for (const auto& peak : raw_peaks) {
        reliable.push_back({peak.peak_gray, peak.left_gray, peak.right_gray,
                            peak.span, peak.area, peak.peak_height, 0.0});
    }

    // 2. 每轮选择最高未归组峰作为锚点，向左右扩张并合并近邻峰。
    std::vector<GrayDensityPeak> merged;
    std::vector<bool> assigned(reliable.size(), false);
    int next_merge_group = 1;
    while (merged.size() < reliable.size()) {
        size_t anchor = reliable.size();
        for (size_t index = 0; index < reliable.size(); ++index) {
            if (!assigned[index] &&
                (anchor == reliable.size() ||
                 reliable[index].peak_height > reliable[anchor].peak_height)) {
                anchor = index;
            }
        }
        if (anchor == reliable.size()) {
            break;
        }

        size_t begin = anchor;
        size_t end = anchor;
        assigned[anchor] = true;
        const double minimum_neighbor_height =
            reliable[anchor].peak_height * kPeakHeightRatioForMerge;

        // 在当前边界 15 灰度范围内继续寻找满足高度条件的最远峰。
        // 即使中间峰自身高度不足，只要更远峰满足条件，中间峰也归入同一组。
        while (begin > 0) {
            size_t farthest_merge = begin;
            for (size_t index = begin; index-- > 0;) {
                if (assigned[index]) {
                    break;
                }
                const int distance =
                    reliable[begin].peak_gray - reliable[index].peak_gray;
                if (distance >= kMinimumPeakDistance) {
                    break;
                }
                if (reliable[index].peak_height >= minimum_neighbor_height) {
                    farthest_merge = index;
                }
            }
            if (farthest_merge == begin) {
                break;
            }
            for (size_t index = farthest_merge; index < begin; ++index) {
                assigned[index] = true;
            }
            begin = farthest_merge;
        }
        while (end + 1 < reliable.size()) {
            size_t farthest_merge = end;
            for (size_t index = end + 1; index < reliable.size(); ++index) {
                if (assigned[index]) {
                    break;
                }
                const int distance =
                    reliable[index].peak_gray - reliable[end].peak_gray;
                if (distance >= kMinimumPeakDistance) {
                    break;
                }
                if (reliable[index].peak_height >= minimum_neighbor_height) {
                    farthest_merge = index;
                }
            }
            if (farthest_merge == end) {
                break;
            }
            for (size_t index = end + 1; index <= farthest_merge; ++index) {
                assigned[index] = true;
            }
            end = farthest_merge;
        }

        GrayDensityPeak merged_peak = reliable[anchor];
        merged_peak.left_gray = reliable[begin].left_gray;
        merged_peak.right_gray = reliable[end].right_gray;
        merged_peak.merge_group = next_merge_group++;
        merged_peak.merge_source_begin = static_cast<int>(begin + 1);
        merged_peak.merge_source_end = static_cast<int>(end + 1);
        merged_peak.merged_peak_count = static_cast<int>(end - begin + 1);
        merged_peak.merge_reason = begin == end ? "anchor_only" : "height>=0.10";
        merged.push_back(merged_peak);
    }
    // 3. 恢复灰度顺序，并按合并后的边界重新统计跨度、面积和谷值。
    std::sort(merged.begin(), merged.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.left_gray < rhs.left_gray;
    });

    for (size_t index = 0; index < merged.size(); ++index) {
        auto& peak = merged[index];
        peak.span = peak.right_gray - peak.left_gray + 1;
        peak.area = 0;
        for (int row = 0; row < gray.rows; ++row) {
            for (int column = 0; column < gray.cols; ++column) {
                const int value = gray.at<unsigned char>(row, column);
                if (value > 0 && value >= peak.left_gray && value <= peak.right_gray) {
                    ++peak.area;
                }
            }
        }
        peak.valley_height = index + 1 < merged.size()
            ? rangeMinimum(density, peak.peak_gray, merged[index + 1].peak_gray)
            : 0.0;
    }
    return merged;
}

/// 计算所选峰值区间覆盖整幅图非零像素的比例。
double peakAreaRatio(const cv::Mat& gray, const std::vector<GrayDensityPeak>& peaks) {
    const int nonzero_area = cv::countNonZero(gray > 0);
    if (nonzero_area <= 0 || peaks.empty()) {
        return 0.0;
    }
    long long selected_area = 0;
    for (const auto& peak : peaks) {
        selected_area += peak.area;
    }
    return std::min(1.0, static_cast<double>(selected_area) / nonzero_area);
}

/// 根据可靠峰区间生成归一化的灰度密度特征层。
std::vector<cv::Mat> buildDensityLayers(
    const cv::Mat& gray,
    const std::vector<GrayDensityPeak>& peaks) {
    std::vector<cv::Mat> layers;
    layers.reserve(peaks.size());
    for (const auto& peak : peaks) {
        cv::Mat layer = cv::Mat::zeros(gray.size(), CV_8UC1);
        const cv::Mat mask = (gray >= peak.left_gray) &
                             (gray <= peak.right_gray) & (gray > 0);
        gray.copyTo(layer, mask);
        cv::Mat normalized;
        cv::normalize(layer, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
        normalized.setTo(0, layer == 0);
        layers.push_back(std::move(normalized));
    }
    return layers;
}

/// 生成固定十层方案中新检测的九个暗部层；255 原图层复用 FAST 特征。
std::vector<cv::Mat> buildFixedTenLayers(const cv::Mat& gray, const int layer_count) {
    std::vector<cv::Mat> layers;
    const int fixed_layer_count = std::max(1, layer_count);
    for (int index = 1; index < fixed_layer_count; ++index) {
        const int threshold = 255 * (fixed_layer_count - index) / fixed_layer_count;
        const cv::Mat layer = buildDarkFeatureLayer(gray, threshold);
        if (!layer.empty()) {
            layers.push_back(layer);
        }
    }
    return layers;
}

/// 根据开关、可靠峰数量和 top3 覆盖率为单张图选择密度层或固定十层。
LayerPlan chooseLayerPlan(const cv::Mat& gray,
                          const int layer_count,
                          const bool density_clustering_enabled,
                          const bool fixed_ten_layers_enabled) {
    // 1. 独立分析当前图像的灰度峰，并保留合并后的全部峰信息。
    LayerPlan plan;
    plan.all_peaks = analyzeGrayDensityPeaks(gray, nullptr);

    // 2. 关闭密度聚类时，固定使用 255 FAST 原图层和 9 个暗部层。
    if (!density_clustering_enabled) {
        plan.mode = fixed_ten_layers_enabled ? "fixed10" : "fast_only";
        plan.layers = fixed_ten_layers_enabled
            ? buildFixedTenLayers(gray, layer_count)
            : std::vector<cv::Mat>{};
        return plan;
    }

    // 3. 峰数为 2/3 时直接使用对应密度区间；峰数不足时固定十层。
    if (plan.all_peaks.size() == 2) {
        plan.mode = "density2";
        plan.selected_peaks = plan.all_peaks;
    } else if (plan.all_peaks.size() == 3) {
        plan.mode = "density3";
        plan.selected_peaks = plan.all_peaks;
    } else if (plan.all_peaks.size() >= 4) {
        plan.selected_peaks = plan.all_peaks;
        std::sort(plan.selected_peaks.begin(), plan.selected_peaks.end(),
                  [](const auto& lhs, const auto& rhs) {
                      if (lhs.area != rhs.area) {
                          return lhs.area > rhs.area;
                      }
                      return lhs.span > rhs.span;
                  });
        plan.selected_peaks.resize(3);
        std::sort(plan.selected_peaks.begin(), plan.selected_peaks.end(),
                  [](const auto& lhs, const auto& rhs) {
                      return lhs.left_gray < rhs.left_gray;
                  });
        if (peakAreaRatio(gray, plan.selected_peaks) >= kMinimumTop3AreaRatio) {
            plan.mode = "top3";
        } else {
            plan.mode = fixed_ten_layers_enabled ? "fixed10" : "fast_only";
            plan.selected_peaks.clear();
        }
    } else {
        plan.mode = fixed_ten_layers_enabled ? "fixed10" : "fast_only";
    }

    // 4. 峰数不少于 4 时，只有 top3 覆盖率达到阈值才使用密度层。
    plan.layers = plan.mode == "fixed10"
        ? buildFixedTenLayers(gray, layer_count)
        : buildDensityLayers(gray, plan.selected_peaks);
    return plan;
}

/// 将 source/target 的原始峰和合并峰选择结果写入当前用例 debug 目录。
void writeDensityPeakDebug(const RegistrationContext& ctx,
                           const LayerPlan& source,
                           const LayerPlan& target) {
    if (ctx.output_dir.empty()) {
        return;
    }
    // 1. 创建当前用例的 debug 目录；无法创建时不影响主配准流程。
    const std::filesystem::path debug_dir = ctx.output_dir / "debug";
    std::error_code error;
    std::filesystem::create_directories(debug_dir, error);
    if (error) {
        IR_LOG_WARN("无法创建灰度峰值诊断目录：", debug_dir.string(), ", ", error.message());
        return;
    }

    // 2. 输出合并前的原始局部峰，便于检查直方图噪声和候选峰数量。
    std::ofstream raw_output(debug_dir / "gray_density_peaks_raw.csv", std::ios::trunc);
    if (raw_output) {
        raw_output << "image,gray_order,peak_gray,left_peak_gray,right_peak_gray,span,area,peak_height\n";
        for (const auto& item : {std::pair<const char*, const LayerPlan&>{"source", source},
                                 {"target", target}}) {
            std::vector<RawGrayDensityPeak> raw_peaks;
            analyzeGrayDensityPeaks(
                item.first == std::string("source") ? ctx.images.first_gray : ctx.images.second_gray,
                &raw_peaks);
            for (size_t index = 0; index < raw_peaks.size(); ++index) {
                const auto& peak = raw_peaks[index];
                raw_output << item.first << ',' << index + 1 << ',' << peak.peak_gray << ','
                           << peak.left_peak_gray << ',' << peak.right_peak_gray << ','
                           << peak.span << ',' << peak.area << ',' << peak.peak_height << '\n';
            }
        }
    }

    // 3. 输出合并后的可靠峰、图层模式和选中状态。
    std::ofstream output(debug_dir / "gray_density_peaks.csv", std::ios::trunc);
    if (!output) {
        return;
    }
    output << "image,layer_mode,gray_order,selection_rank,peak_gray,left_gray,right_gray,span,area,"
              "peak_height,valley_height,merge_group,merge_source_begin,merge_source_end,"
              "merged_peak_count,merge_reason,selected\n";
    for (const auto& item : {std::pair<const char*, const LayerPlan&>{"source", source},
                             {"target", target}}) {
        for (size_t index = 0; index < item.second.all_peaks.size(); ++index) {
            const auto& peak = item.second.all_peaks[index];
            const auto selected = std::find_if(
                item.second.selected_peaks.begin(), item.second.selected_peaks.end(),
                [&](const auto& selected_peak) { return selected_peak.peak_gray == peak.peak_gray; });
            const bool is_selected = item.second.mode != "fixed10" &&
                                     selected != item.second.selected_peaks.end();
            output << item.first << ',' << item.second.mode << ',' << index + 1 << ',';
            if (is_selected) {
                output << std::distance(item.second.selected_peaks.begin(), selected) + 1;
            }
            output << ',' << peak.peak_gray << ',' << peak.left_gray << ',' << peak.right_gray << ','
                   << peak.span << ',' << peak.area << ',' << peak.peak_height << ','
                   << peak.valley_height << ',' << peak.merge_group << ','
                   << peak.merge_source_begin << ',' << peak.merge_source_end << ','
                   << peak.merged_peak_count << ','
                   << peak.merge_reason << ',' << (is_selected ? "yes" : "no") << '\n';
        }
    }
}

/// 多层暗部流程：保留阈值以下像素，再拉伸有效暗部灰度范围。
cv::Mat buildDarkFeatureLayer(const cv::Mat& gray, const int threshold) {
    cv::Mat layer = cv::Mat::zeros(gray.size(), CV_8UC1);
    const cv::Mat dark_mask = gray <= threshold;
    gray.copyTo(layer, dark_mask);

    double max_value = 0.0;
    cv::minMaxLoc(layer, nullptr, &max_value);
    if (max_value <= 0.0) {
        return {};
    }

    cv::Mat normalized;
    cv::normalize(layer, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    normalized.setTo(0, layer == 0);
    return normalized;
}

std::tuple<int, int, int, int> makeKeypointKey(const cv::KeyPoint& keypoint) {
    return {
        static_cast<int>(std::lround(keypoint.pt.x * 10.0f)),
        static_cast<int>(std::lround(keypoint.pt.y * 10.0f)),
        static_cast<int>(std::lround(keypoint.size * 10.0f)),
        static_cast<int>(std::lround(keypoint.angle))
    };
}

/// 将单层检测到的关键点追加到合并集合，并按关键点属性去重。
void appendUniqueKeypoints(const std::vector<cv::KeyPoint>& layer_keypoints,
                           std::vector<cv::KeyPoint>& merged_keypoints,
                           std::set<std::tuple<int, int, int, int>>& seen) {
    for (const cv::KeyPoint& keypoint : layer_keypoints) {
        if (!seen.insert(makeKeypointKey(keypoint)).second) {
            continue;
        }
        merged_keypoints.push_back(keypoint);
    }
}

/// 在指定图层上检测关键点，合并各层结果并丢弃重复关键点。
bool extractLayerKeypoints(const cv::Mat& gray,
                           const std::vector<cv::Mat>& layers,
                           const YAML::Node& keypoint_config,
                           std::vector<cv::KeyPoint>& keypoints) {
    std::set<std::tuple<int, int, int, int>> seen;
    (void)gray;
    for (const cv::Mat& layer : layers) {
        if (layer.empty()) {
            continue;
        }

        // 每层创建独立算法对象，避免不同层之间共享 OpenCV 内部状态。
        const auto layer_extractor = Factory::createKeypointExtractor(keypoint_config);
        if (!layer_extractor) {
            continue;
        }
        const auto detector = layer_extractor->feature2D();
        if (detector.empty()) {
            continue;
        }
        std::vector<cv::KeyPoint> layer_keypoints;
        cv::Mat discarded_descriptors;
        detector->detectAndCompute(
            layer, cv::noArray(), layer_keypoints, discarded_descriptors);
        appendUniqueKeypoints(layer_keypoints, keypoints, seen);
    }
    return !keypoints.empty();
}

/// 先合并暗部层关键点，再补充 FAST 原图层中的新增关键点并计算描述子。
bool mergeOriginalFeaturesAndComputeDescriptors(
    const cv::Mat& gray,
    const YAML::Node& keypoint_config,
    const KeypointImageData& original_features,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors) {
    // 1. 暗部层优先进入集合；重复关键点保留首次出现的暗部层版本。
    std::vector<cv::KeyPoint> dark_keypoints = std::move(keypoints);
    keypoints.clear();
    std::set<std::tuple<int, int, int, int>> seen;
    std::vector<int> original_descriptor_indices;
    original_descriptor_indices.reserve(
        dark_keypoints.size() + original_features.keypoints.size());
    const bool original_descriptors_aligned =
        !original_features.descriptors.empty() &&
        original_features.descriptors.rows ==
            static_cast<int>(original_features.keypoints.size());
    for (const cv::KeyPoint& keypoint : dark_keypoints) {
        seen.insert(makeKeypointKey(keypoint));
        keypoints.push_back(keypoint);
        original_descriptor_indices.push_back(-1);
    }
    for (size_t index = 0; index < original_features.keypoints.size(); ++index) {
        const cv::KeyPoint& keypoint = original_features.keypoints[index];
        if (!seen.insert(makeKeypointKey(keypoint)).second) {
            continue;
        }
        keypoints.push_back(keypoint);
        original_descriptor_indices.push_back(
            original_descriptors_aligned ? static_cast<int>(index) : -1);
    }

    if (keypoints.empty()) {
        return false;
    }

    // 2. 对合并后的关键点统一在原始灰度图上计算描述子。
    const auto extractor_holder = Factory::createKeypointExtractor(keypoint_config);
    if (!extractor_holder || extractor_holder->feature2D().empty()) {
        return false;
    }
    extractor_holder->feature2D()->compute(gray, keypoints, descriptors);
    if (descriptors.empty() ||
        descriptors.rows != static_cast<int>(keypoints.size())) {
        return false;
    }

    // 3. 仅覆盖只来自 FAST 原图层的关键点描述子；重复点保留暗部层的重算结果。
    if (original_descriptors_aligned &&
        descriptors.cols == original_features.descriptors.cols &&
        descriptors.type() == original_features.descriptors.type()) {
      for (size_t index = 0; index < keypoints.size(); ++index) {
        const int original_index = original_descriptor_indices[index];
        if (original_index >= 0) {
            original_features.descriptors.row(original_index).copyTo(
                descriptors.row(static_cast<int>(index)));
        }
      }
    }
    return !keypoints.empty() && !descriptors.empty() &&
           descriptors.rows == static_cast<int>(keypoints.size());
}

} // namespace

/// 创建多层暗部关键点提取器，并复用指定的基础关键点配置。
MultilayerDarkKeypointExtractor::MultilayerDarkKeypointExtractor(
    const YAML::Node& keypoint_config,
    const int layer_count,
    const bool density_clustering_enabled,
    const bool fixed_ten_layers_enabled)
    : _extractor(Factory::createKeypointExtractor(keypoint_config)),
      _keypoint_config(keypoint_config),
      _name(_extractor->name() + "_DARK_MULTILAYER"),
      _layer_count(std::max(1, layer_count)),
      _density_clustering_enabled(density_clustering_enabled),
      _fixed_ten_layers_enabled(fixed_ten_layers_enabled) {}

/// 执行灰度密度选层、关键点合并和原始灰度图描述子计算。
bool MultilayerDarkKeypointExtractor::extract(RegistrationContext& ctx) {
    // 1. 接管 FAST 特征，再初始化多层阶段输出。
    auto& features = ctx.keypoint_data;
    KeypointData original_features = std::move(features);
    features.clear();
    features.type = _extractor->type();
    features.norm_type = _extractor->normType();

    // 2. 准备原图灰度，暗部图层只用于本提取器，不改写配准流程中的原始图像。
    if (ctx.images.first.empty() || ctx.images.second.empty()) {
        IR_LOG_ERROR("多层暗部提取缺少输入图像。");
        return false;
    }
    if (!image_utils::ensureGray(ctx.images.first, ctx.images.first_gray) ||
        !image_utils::ensureGray(ctx.images.second, ctx.images.second_gray)) {
        IR_LOG_ERROR("多层暗部提取无法准备灰度图。");
        return false;
    }

    // 3. source、target 独立按灰度密度峰选择图层；某一张图缺少某个图层不影响另一张图。
    std::vector<cv::KeyPoint> first_keypoints;
    std::vector<cv::KeyPoint> second_keypoints;
    const LayerPlan first_plan = chooseLayerPlan(
        ctx.images.first_gray,
        _layer_count,
        _density_clustering_enabled,
        _fixed_ten_layers_enabled);
    const LayerPlan second_plan = chooseLayerPlan(
        ctx.images.second_gray,
        _layer_count,
        _density_clustering_enabled,
        _fixed_ten_layers_enabled);
    writeDensityPeakDebug(ctx, first_plan, second_plan);
    extractLayerKeypoints(ctx.images.first_gray, first_plan.layers, _keypoint_config, first_keypoints);
    extractLayerKeypoints(ctx.images.second_gray, second_plan.layers, _keypoint_config, second_keypoints);

    // 4. 先合并暗部层，再补充 FAST 原图层中的新增关键点并计算描述子。
    const bool first_ok = mergeOriginalFeaturesAndComputeDescriptors(
        ctx.images.first_gray, _keypoint_config, original_features.first,
        first_keypoints, features.first.descriptors);
    const bool second_ok = mergeOriginalFeaturesAndComputeDescriptors(
        ctx.images.second_gray, _keypoint_config, original_features.second,
        second_keypoints, features.second.descriptors);
    features.first.keypoints = std::move(first_keypoints);
    features.second.keypoints = std::move(second_keypoints);

    // 5. 输出的关键点与描述子始终按行对齐，可直接进入既有匹配、过滤和几何流程。
    IR_LOG_INFO("多层暗部 ",
                _extractor->name(),
                " 提取完成：",
                features.first.keypoints.size(),
                " / ",
                features.second.keypoints.size(),
                " 个关键点，图层模式=",
                first_plan.mode,
                " / ",
                second_plan.mode);
    return first_ok && second_ok && !features.empty();
}

} // namespace ir
