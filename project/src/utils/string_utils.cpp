#include "utils/string_utils.h"

#include <cctype>

namespace ir {
namespace string_utils {

std::string toUpperAscii(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return value;
}

std::string normalizedKey(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char c : value) {
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(std::toupper(c)));
        }
    }
    return out;
}

} // namespace string_utils
} // namespace ir

