#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "core/registration.h"

namespace ir {

/// 管理从 YAML 配置文件加载的注册流水线，并提供按名称检索的能力。
///
/// PipelineManager 主要用于批处理场景：它负责创建 Registration 实例，
/// 保存成功加载的对象，并允许外部按名称查找。
class PipelineManager {
public:
    /// 构造一个空的管理器。
    PipelineManager() = default;

    /// 批量加载多个 YAML 配置文件，并返回成功加载的数量。
    ///
    /// 某个配置失败时会被跳过，不影响其他配置继续加载。
    int loadAll(const std::vector<std::filesystem::path>& yaml_paths);

    /// 加载单个 YAML 文件并创建一个已配置的 Registration 实例。
    std::shared_ptr<Registration> load(const std::filesystem::path& yaml_path);

    /// 返回当前已经成功加载的所有注册实例。
    const std::vector<std::shared_ptr<Registration>>& registrations() const {
        return regs_;
    }

    /// 按名称查找已加载的注册实例。
    std::shared_ptr<Registration> findByName(const std::string& name) const;

    /// 清空管理器中保存的所有注册实例。
    void clear() { regs_.clear(); }

private:
    /// 按加载顺序保存的注册实例。
    std::vector<std::shared_ptr<Registration>> regs_;
};

} // namespace ir
