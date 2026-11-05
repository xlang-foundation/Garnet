// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>

namespace Garnet::Tokenization::Utf8 {
// On failure, consumed counts a valid prefix of an incomplete/invalid
// sequence, matching the replacement boundaries used by UTF-8 lossy decode.
inline bool Character(const std::string& s, size_t pos, size_t& consumed) {
    const auto a = static_cast<unsigned char>(s[pos]);
    consumed = 1;
    if (a < 128) return true;
    const size_t n = a >= 0xc2 && a <= 0xdf ? 2 : a >= 0xe0 && a <= 0xef ? 3 : a >= 0xf0 && a <= 0xf4 ? 4 : 0;
    if (!n) return false;
    for (size_t i = 1; i < n; ++i) {
        if (pos + i >= s.size()) return false;
        const auto b = static_cast<unsigned char>(s[pos + i]);
        if (b < 0x80 || b > 0xbf || (i == 1 && ((a == 0xe0 && b < 0xa0) ||
            (a == 0xed && b > 0x9f) || (a == 0xf0 && b < 0x90) || (a == 0xf4 && b > 0x8f)))) return false;
        consumed = i + 1;
    }
    return true;
}
inline bool Valid(const std::string& s) {
    for (size_t p = 0, n; p < s.size(); p += n) if (!Character(s, p, n)) return false;
    return true;
}
inline std::string ReplaceInvalid(const std::string& s) {
    std::string out;
    for (size_t p = 0, n; p < s.size(); p += n) {
        if (Character(s, p, n)) out.append(s, p, n); else out += "\xef\xbf\xbd";
    }
    return out;
}
}
