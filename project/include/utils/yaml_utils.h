#pragma once

#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace ir {

/// YAML 读取辅助工具。
namespace yaml_utils {

/// 从节点中读取指定类型的值，缺失或转换失败时返回默认值。
template <typename T>
inline T get(const YAML::Node& node, const std::string& key, const T& fallback) {
    if (!node || !node.IsMap()) return fallback;
    const YAML::Node child = node[key];
    if (!child || child.IsNull()) return fallback;
    try {
        return child.as<T>();
    } catch (const YAML::Exception&) {
        return fallback;
    }
}

/// 读取字符串值。
inline std::string getString(const YAML::Node& node,
                             const std::string& key,
                             const std::string& fallback = std::string{}) {
    return get<std::string>(node, key, fallback);
}

/// 读取整型值。
inline int getInt(const YAML::Node& node, const std::string& key, int fallback = 0) {
    return get<int>(node, key, fallback);
}

/// 读取浮点值。
inline double getDouble(const YAML::Node& node, const std::string& key, double fallback = 0.0) {
    return get<double>(node, key, fallback);
}

/// 读取单精度浮点值。
inline float getFloat(const YAML::Node& node, const std::string& key, float fallback = 0.0f) {
    return get<float>(node, key, fallback);
}

/// 读取布尔值。
inline bool getBool(const YAML::Node& node, const std::string& key, bool fallback = false) {
    return get<bool>(node, key, fallback);
}

/// 读取序列并转换为 `std::vector<T>`。
template <typename T>
inline std::vector<T> getVec(const YAML::Node& node,
                             const std::string& key,
                             const std::vector<T>& fallback = {}) {
    if (!node || !node.IsMap()) return fallback;
    const YAML::Node child = node[key];
    if (!child || !child.IsSequence()) return fallback;
    std::vector<T> out;
    out.reserve(child.size());
    try {
        for (const auto& v : child) out.push_back(v.as<T>());
    } catch (const YAML::Exception&) {
        return fallback;
    }
    return out;
}

} // namespace yaml_utils
} // namespace ir
