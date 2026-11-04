// fp16_selftest.cpp - self-test of the fp32 <-> fp16 rounding and of the header writer. No files, no dependencies.
//   fp16_selftest        (exit code 0 = all passed)
#include <cmath>
#include <cstdio>
#include <string>

#include "safetensors_io.hpp"

static int failures = 0;
static void check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

int main() {
    // 1. every non-NaN fp16 value survives fp16 -> fp32 -> fp16 unchanged
    int bad = 0;
    for (uint32_t h = 0; h < 0x10000; ++h) {
        if ((h & 0x7C00) == 0x7C00 && (h & 0x3FF)) continue;  // NaN
        if (st::f32_to_f16(st::f16_to_f32(static_cast<uint16_t>(h))) != h) ++bad;
    }
    check(bad == 0, "all 63,488 non-NaN fp16 values round-trip exactly");

    // 2. exact ties go to the even neighbour; just above / below go to the nearer one (positive and negative)
    int tie_bad = 0, near_bad = 0;
    for (uint32_t h = 0; h < 0x7BFF; ++h) {
        const float a = st::f16_to_f32(static_cast<uint16_t>(h)), b = st::f16_to_f32(static_cast<uint16_t>(h + 1));
        const float mid = static_cast<float>((static_cast<double>(a) + b) / 2);
        const uint16_t even = (h & 1) ? static_cast<uint16_t>(h + 1) : static_cast<uint16_t>(h);
        if (st::f32_to_f16(mid) != even) ++tie_bad;
        if (st::f32_to_f16(-mid) != (even | 0x8000)) ++tie_bad;
        if (st::f32_to_f16(std::nextafter(mid, 1e30f)) != h + 1) ++near_bad;
        if (st::f32_to_f16(std::nextafter(mid, -1e30f)) != h) ++near_bad;
    }
    check(tie_bad == 0, "exact ties between neighbouring fp16 values round to even (both signs, incl. subnormals)");
    check(near_bad == 0, "values just above/below a tie round to the nearer neighbour");

    // 3. limits
    check(st::f32_to_f16(65504.0f) == 0x7BFF, "65504 (largest fp16) stays 65504");
    check(st::f32_to_f16(65519.0f) == 0x7BFF, "65519 rounds down to 65504");
    check(st::f32_to_f16(65520.0f) == 0x7C00, "65520 becomes infinity (the converter refuses such files)");
    check(st::f32_to_f16(std::ldexp(1.0f, -24)) == 0x0001, "2^-24 is the smallest fp16 subnormal");
    check(st::f32_to_f16(std::ldexp(1.0f, -25)) == 0x0000, "2^-25 (a tie) rounds to zero");
    check(st::f32_to_f16(std::nextafter(std::ldexp(1.0f, -25), 1.0f)) == 0x0001, "just above 2^-25 rounds to 2^-24");
    check(st::f32_to_f16(std::ldexp(1.0f, -26)) == 0x0000, "2^-26 underflows to zero");
    check(st::f32_to_f16(-0.0f) == 0x8000 && st::f32_to_f16(0.0f) == 0x0000, "signed zeros keep their sign");

    // 4. header text identical to Python json.dumps(separators=(",", ":"), ensure_ascii=True)
    const std::string text = "{\"format\":\"pt\",\"note\":\"t\xC3\xA9st \xE0\xA6\xAC\xE0\xA6\xBE \"}";
    st::J j = st::Parser(text).parse();
    check(st::dump(j) == "{\"format\":\"pt\",\"note\":\"t\\u00e9st \\u09ac\\u09be \"}", "metadata text is written like Python's json.dumps");

    std::printf("\n%s\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return failures ? 1 : 0;
}
