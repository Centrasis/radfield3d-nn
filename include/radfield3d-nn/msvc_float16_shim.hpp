#pragma once
// ─────────────────────────────────────────────────────────────────────────────────────────────
// MSVC half-precision shim.
//
// RadFiled3D exposes its voxel half type as `Typing::float16 = _Float16`, but ONLY under
// `#if defined(__FLT16_MAX__)` (see RadFiled3D/helpers/Typing.hpp). GCC 12+/Clang define that
// macro and provide the `_Float16` arithmetic type; MSVC does neither, so on Windows
// `RadFiled3D::Typing::float16` and every fp16 voxel-layer specialization are compiled out and
// our fp16-aware deploy code fails to build (C3861/C2039 'float16' not found).
//
// This header gives MSVC a 2-byte `_Float16` type and advertises `__FLT16_MAX__` *before*
// RadFiled3D's headers are parsed, so RadFiled3D's `using float16 = _Float16;` and its
// dtype-name / dtype-size / DType specializations activate against this type. It is force-included
// into every C++ TU on MSVC (CMAKE_CXX_FLAGS /FI), including RadFiled3D's own sources, so
// RADFILED3D_HAS_FLOAT16 is consistent across the whole build (no ODR skew at the library
// boundary). GCC/Clang use the real `_Float16` and never see this file.
// ─────────────────────────────────────────────────────────────────────────────────────────────
#if defined(_MSC_VER) && !defined(__FLT16_MAX__)

#include <cstdint>
#include <cstring>

namespace rfnn_detail {

// IEEE-754 binary16 -> binary32.
inline float half_bits_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t f;
    if (exp == 0u) {
        if (man == 0u) {
            f = sign;  // +/- zero
        } else {
            // subnormal half -> normalized float
            exp = 1u;
            while ((man & 0x400u) == 0u) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = sign | ((exp + (127u - 15u)) << 23) | (man << 13);
        }
    } else if (exp == 0x1Fu) {
        f = sign | 0x7F800000u | (man << 13);  // Inf / NaN
    } else {
        f = sign | ((exp + (127u - 15u)) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// IEEE-754 binary32 -> binary16, round-to-nearest-even.
inline uint16_t float_to_half_bits(float value) {
    uint32_t f;
    std::memcpy(&f, &value, sizeof(f));
    const uint32_t sign = (f >> 16) & 0x8000u;
    const uint32_t biased = (f >> 23) & 0xFFu;
    uint32_t man = f & 0x7FFFFFu;

    if (biased == 0xFFu) {  // Inf / NaN
        return static_cast<uint16_t>(sign | 0x7C00u | (man ? 0x200u : 0u));
    }
    int32_t exp = static_cast<int32_t>(biased) - 127 + 15;
    if (exp >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> Inf
    }
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);  // underflow -> zero
        man |= 0x800000u;  // restore implicit 1
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint16_t halfman = static_cast<uint16_t>(man >> shift);
        const uint32_t rem = man & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (halfman & 1u))) ++halfman;
        return static_cast<uint16_t>(sign | halfman);
    }
    uint16_t out = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                                         (man >> 13));
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (out & 1u))) ++out;  // round (may carry into exp)
    return out;
}

}  // namespace rfnn_detail

// 2-byte half that RadFiled3D adopts as `Typing::float16`. Trivially copyable / standard layout so
// it serializes by raw bytes exactly like the native _Float16 on other platforms.
struct _Float16 {
    uint16_t bits_;
    _Float16() : bits_(0) {}
    _Float16(float v) : bits_(rfnn_detail::float_to_half_bits(v)) {}
    operator float() const { return rfnn_detail::half_bits_to_float(bits_); }
};
static_assert(sizeof(_Float16) == 2, "float16 shim must be exactly 2 bytes");

// Activate RadFiled3D's RADFILED3D_HAS_FLOAT16 branch (it keys off __FLT16_MAX__).
#define __FLT16_MAX__ 6.5504e+4

#endif  // _MSC_VER && !__FLT16_MAX__
