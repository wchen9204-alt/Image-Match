#pragma once

#include <string>

#include "core/context.h"

namespace ir {

/// 结构关联器接口。
///
/// 负责根据结构响应图、轮廓或线段等结构数据，建立两张图像之间的
/// 结构对应、相似性或对齐依据，并将结果写入
/// `RegistrationContext::structure_match_data`。
class IStructureAssociator {
public:
    virtual ~IStructureAssociator() = default;

    /// 返回结构关联方法名称。
    virtual std::string name() const = 0;

    /// 执行结构关联/匹配，并把结果写入 `structure_match_data`。
    virtual bool associate(RegistrationContext& ctx) = 0;
};

} // namespace ir

