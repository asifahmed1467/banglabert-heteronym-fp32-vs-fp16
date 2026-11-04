// safetensors_io.hpp - minimal safetensors reader/writer, fp32<->fp16 conversion. Standard library only.
// Format: 8 bytes little-endian header length N, N bytes JSON header, then raw tensor data.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static_assert(sizeof(float) == 4, "float must be 32 bit");

namespace st {

// ---------------- fp32 <-> fp16 (IEEE 754 binary16, round to nearest even) ----------------
inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp = (x >> 23) & 0xFFu;
    uint32_t man = x & 0x7FFFFFu;
    if (exp == 0xFFu) return static_cast<uint16_t>(sign | 0x7C00u | (man ? (0x200u | (man >> 13)) : 0u));  // inf / NaN
    const int32_t e = static_cast<int32_t>(exp) - 127 + 15;
    if (e >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);  // too large: infinity
    if (e <= 0) {                                                  // fp16 subnormal or zero
        if (e < -10) return static_cast<uint16_t>(sign);
        man |= 0x800000u;
        const int shift = 14 - e;
        uint32_t h = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1u), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return static_cast<uint16_t>(sign | h);
    }
    uint32_t h = sign | (static_cast<uint32_t>(e) << 10) | (man >> 13);
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;  // a carry correctly rolls into the exponent
    return static_cast<uint16_t>(h);
}

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, x;
    if (exp == 0) {
        if (man == 0) x = sign;
        else {  // subnormal: normalise
            int e = -1;
            do { ++e; man <<= 1; } while (!(man & 0x400u));
            man &= 0x3FFu;
            x = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (man << 13);
        }
    } else if (exp == 0x1Fu) x = sign | 0x7F800000u | (man << 13);
    else x = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// ---------------- tiny JSON (enough for safetensors headers) ----------------
struct J {
    enum T { NUL, NUM, STR, ARR, OBJ } t = NUL;
    long long i = 0;
    std::string s;
    std::vector<J> a;
    std::vector<std::pair<std::string, J>> o;
    const J* get(const std::string& k) const {
        for (auto& kv : o) if (kv.first == k) return &kv.second;
        return nullptr;
    }
};

inline void put_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) out += static_cast<char>(cp);
    else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
    else { out += static_cast<char>(0xF0 | (cp >> 18)); out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
}

class Parser {
public:
    explicit Parser(const std::string& s) : s_(s) {}
    J parse() { J v = value(); ws(); if (p_ != s_.size()) fail("trailing characters"); return v; }
private:
    const std::string& s_;
    size_t p_ = 0;
    [[noreturn]] void fail(const char* m) const { throw std::runtime_error(std::string("bad safetensors header JSON: ") + m); }
    void ws() { while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\n' || s_[p_] == '\t' || s_[p_] == '\r')) ++p_; }
    char peek() { ws(); if (p_ >= s_.size()) fail("unexpected end"); return s_[p_]; }
    void expect(char c) { if (peek() != c) fail("unexpected character"); ++p_; }
    std::string str() {
        expect('"');
        std::string out;
        while (true) {
            if (p_ >= s_.size()) fail("unterminated string");
            char c = s_[p_++];
            if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            if (p_ >= s_.size()) fail("bad escape");
            char e = s_[p_++];
            switch (e) {
                case '"': out += '"'; break; case '\\': out += '\\'; break; case '/': out += '/'; break;
                case 'b': out += '\b'; break; case 'f': out += '\f'; break; case 'n': out += '\n'; break;
                case 'r': out += '\r'; break; case 't': out += '\t'; break;
                case 'u': {
                    auto hex4 = [&]() { if (p_ + 4 > s_.size()) fail("bad \\u"); unsigned v = static_cast<unsigned>(std::stoul(s_.substr(p_, 4), nullptr, 16)); p_ += 4; return v; };
                    unsigned cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && p_ + 1 < s_.size() && s_[p_] == '\\' && s_[p_ + 1] == 'u') { p_ += 2; unsigned lo = hex4(); cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); }
                    put_utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }
    J value() {
        J v;
        char c = peek();
        if (c == '{') {
            ++p_; v.t = J::OBJ;
            if (peek() == '}') { ++p_; return v; }
            while (true) { std::string k = str(); expect(':'); v.o.emplace_back(k, value()); if (peek() == ',') { ++p_; continue; } expect('}'); return v; }
        }
        if (c == '[') {
            ++p_; v.t = J::ARR;
            if (peek() == ']') { ++p_; return v; }
            while (true) { v.a.push_back(value()); if (peek() == ',') { ++p_; continue; } expect(']'); return v; }
        }
        if (c == '"') { v.t = J::STR; v.s = str(); return v; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            size_t b = p_;
            while (p_ < s_.size() && (s_[p_] == '-' || s_[p_] == '+' || s_[p_] == '.' || s_[p_] == 'e' || s_[p_] == 'E' || (s_[p_] >= '0' && s_[p_] <= '9'))) ++p_;
            v.t = J::NUM; v.i = std::strtoll(s_.substr(b, p_ - b).c_str(), nullptr, 10);
            return v;
        }
        if (s_.compare(p_, 4, "null") == 0) { p_ += 4; return v; }
        fail("unsupported value");
    }
};

// Same text as Python json.dumps(..., separators=(",", ":")) with ensure_ascii=True.
inline std::string dump_str(const std::string& s) {
    static const char* hex = "0123456789abcdef";
    std::string o = "\"";
    auto u16 = [&](unsigned v) { o += "\\u"; for (int sh = 12; sh >= 0; sh -= 4) o += hex[(v >> sh) & 0xF]; };
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == '"') { o += "\\\""; ++i; }
        else if (c == '\\') { o += "\\\\"; ++i; }
        else if (c == '\n') { o += "\\n"; ++i; } else if (c == '\r') { o += "\\r"; ++i; } else if (c == '\t') { o += "\\t"; ++i; }
        else if (c == '\b') { o += "\\b"; ++i; } else if (c == '\f') { o += "\\f"; ++i; }
        else if (c < 0x20) { u16(c); ++i; }
        else if (c < 0x80) { o += static_cast<char>(c); ++i; }
        else {
            unsigned cp = 0; int n = (c >= 0xF0) ? 3 : (c >= 0xE0) ? 2 : 1;
            cp = c & (0x3F >> n);
            for (int k = 1; k <= n && i + k < s.size(); ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
            i += n + 1;
            if (cp >= 0x10000) { cp -= 0x10000; u16(0xD800 + (cp >> 10)); u16(0xDC00 + (cp & 0x3FF)); } else u16(cp);
        }
    }
    return o + "\"";
}
inline std::string dump(const J& v) {
    switch (v.t) {
        case J::NUL: return "null";
        case J::NUM: return std::to_string(v.i);
        case J::STR: return dump_str(v.s);
        case J::ARR: { std::string o = "["; for (size_t k = 0; k < v.a.size(); ++k) o += (k ? "," : "") + dump(v.a[k]); return o + "]"; }
        default: { std::string o = "{"; for (size_t k = 0; k < v.o.size(); ++k) o += (k ? "," : "") + dump_str(v.o[k].first) + ":" + dump(v.o[k].second); return o + "}"; }
    }
}

// ---------------- safetensors file ----------------
struct Tensor {
    std::string name, dtype;
    std::vector<long long> shape;
    uint64_t begin = 0, end = 0;
    uint64_t count() const { uint64_t n = 1; for (auto d : shape) n *= static_cast<uint64_t>(d); return n; }
    uint64_t bytes() const { return end - begin; }
};
struct File {
    std::vector<Tensor> tensors;  // sorted by data offset
    J metadata;                   // OBJ, possibly empty
    uint64_t data_start = 0;
};

inline File read_header(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    unsigned char b[8];
    if (!f.read(reinterpret_cast<char*>(b), 8)) throw std::runtime_error("file too short: " + path);
    uint64_t n = 0;
    for (int k = 7; k >= 0; --k) n = (n << 8) | b[k];
    if (n > (1ull << 30)) throw std::runtime_error("implausible header size in " + path);
    std::string text(n, '\0');
    if (!f.read(&text[0], static_cast<std::streamsize>(n))) throw std::runtime_error("truncated header in " + path);
    J root = Parser(text).parse();
    File out;
    out.data_start = 8 + n;
    out.metadata.t = J::OBJ;
    for (auto& kv : root.o) {
        if (kv.first == "__metadata__") { out.metadata = kv.second; continue; }
        const J* dt = kv.second.get("dtype");
        const J* sh = kv.second.get("shape");
        const J* off = kv.second.get("data_offsets");
        if (!dt || !sh || !off || off->a.size() != 2) throw std::runtime_error("bad tensor entry " + kv.first);
        Tensor t;
        t.name = kv.first; t.dtype = dt->s;
        for (auto& d : sh->a) t.shape.push_back(d.i);
        t.begin = static_cast<uint64_t>(off->a[0].i); t.end = static_cast<uint64_t>(off->a[1].i);
        out.tensors.push_back(t);
    }
    std::sort(out.tensors.begin(), out.tensors.end(), [](const Tensor& a, const Tensor& b) { return a.begin < b.begin; });
    return out;
}

inline std::vector<uint8_t> read_bytes(std::ifstream& f, uint64_t offset, uint64_t n) {
    std::vector<uint8_t> buf(n);
    f.clear();
    f.seekg(static_cast<std::streamoff>(offset));
    if (n && !f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(n))) throw std::runtime_error("short read");
    return buf;
}

inline uint64_t dtype_size(const std::string& d) {
    if (d == "F32" || d == "I32" || d == "U32") return 4;
    if (d == "F16" || d == "BF16" || d == "I16" || d == "U16") return 2;
    if (d == "F64" || d == "I64" || d == "U64") return 8;
    return 1;  // I8, U8, BOOL, F8_*
}

}  // namespace st
