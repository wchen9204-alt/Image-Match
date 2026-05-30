#pragma once

#include <string>

#include "core/context.h"
#include "core/types.h"

namespace ir {

/// 结构特征提取器接口。
///
/// 负责从两张输入图像中提取边缘、直线、轮廓等结构信息，并写入
/// `RegistrationContext::structure_data`。
class IStructureExtractor {
public:
    virtual ~IStructureExtractor() = default;

    /// 返回提取器名称，用于日志、输出文件名和调试摘要。
    virtual std::string name() const = 0;

    /// 返回当前提取器对应的结构特征类型。
    virtual StructureType type() const = 0;

    /// 执行结构特征提取。
    virtual bool extract(RegistrationContext& ctx) = 0;
};

} // namespace ir
