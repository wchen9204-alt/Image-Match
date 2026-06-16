#pragma once

#include <string>
#include <string_view>

namespace ir {
namespace string_utils {

/// 将 ASCII 字符串转换为大写，保留非字母数字字符。
std::string toUpperAscii(std::string value);

/// 将枚举/方法名规整为只含大写字母和数字的 key。
std::string normalizedKey(std::string_view value);

} // namespace string_utils
} // namespace ir

