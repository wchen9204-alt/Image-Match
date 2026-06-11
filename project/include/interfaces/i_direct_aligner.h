#pragma once

#include <string>

#include "core/context.h"

namespace ir {

/// 直接法配准器接口。
/// 实现类负责从图像灰度/光度关系直接估计变换，并把专属结果写入 ctx.direct_data。
class IDirectAligner {
public:
    virtual ~IDirectAligner() = default;

    /// 返回算法名称，用于日志、摘要和输出文件命名。
    virtual std::string name() const = 0;

    /// 直接从图像灰度/光度信息估计配准结果，并同步可用于 warp 的几何结果到 ctx.geometry_data。
    virtual bool align(RegistrationContext& ctx) = 0;
};

} // namespace ir
