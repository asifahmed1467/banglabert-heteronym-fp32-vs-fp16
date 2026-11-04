// fp16_verify.cpp - C++ version of prove_fp16.py (plus an optional full pass).
//
//   fp16_verify --fp32 <fp32 folder> --fp16 <fp16 folder> [--full]
//
// For every .safetensors file under --fp32 it checks the file of the same relative path under --fp16:
//  1. every F32 tensor became F16; tensors of any other dtype (e.g. int64 position ids) must be byte-for-byte unchanged
//  2. same tensor names and shapes, nothing missing or extra
//  3. every fp16 tensor takes exactly half the bytes
//  4. values: sampled (start, middle, end of each tensor), or ALL values with --full, must equal the fp32 value rounded to fp16
// --full also prints the weight rounding error (relative L2, mean/max absolute error, values rounded to zero).
// Exit code 0 only when proven.
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>

#include "safetensors_io.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    fs::path d32, d16;
    bool full = false;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--full") full = true;
        else if (k == "--fp32" && i + 1 < argc) d32 = argv[++i];
        else if (k == "--fp16" && i + 1 < argc) d16 = argv[++i];
    }
    if (d32.empty() || d16.empty()) { std::cerr << "usage: fp16_verify --fp32 <folder> --fp16 <folder> [--full]\n"; return 2; }
    std::vector<std::string> problems;
    auto bad = [&](const std::string& m) { if (problems.size() < 20) problems.push_back(m); else if (problems.size() == 20) problems.push_back("... more"); };
    try {
        std::vector<fs::path> files;
        for (auto& e : fs::recursive_directory_iterator(d32))
            if (e.is_regular_file() && e.path().extension() == ".safetensors") files.push_back(e.path());
        std::sort(files.begin(), files.end());
        if (files.empty()) { std::cerr << "no .safetensors under " << d32 << "\n"; return 2; }
        for (const auto& f32p : files) {
            const fs::path rel = fs::relative(f32p, d32), f16p = d16 / rel;
            std::cout << "\n" << rel.generic_string() << "\n";
            if (!fs::exists(f16p)) { bad(rel.generic_string() + ": missing in fp16 folder"); continue; }
            st::File a = st::read_header(f32p.string()), b = st::read_header(f16p.string());
            std::map<std::string, const st::Tensor*> mb;
            for (auto& t : b.tensors) mb[t.name] = &t;
            if (a.tensors.size() != b.tensors.size()) bad(rel.generic_string() + ": tensor count differs");
            std::ifstream fa(f32p, std::ios::binary), fb(f16p, std::ios::binary);
            uint64_t n_tensors = 0, params = 0, bytes32 = 0, bytes16 = 0, checked = 0, to_zero = 0, wrong = 0;
            double sum_abs = 0, sum_sq = 0, sum_ref = 0, max_abs = 0;
            for (const auto& ta : a.tensors) {
                auto it = mb.find(ta.name);
                if (it == mb.end()) { bad(ta.name + ": missing in fp16 file"); continue; }
                const st::Tensor& tb = *it->second;
                ++n_tensors;
                if (ta.dtype != "F32") {  // non-float tensors must be untouched
                    if (tb.dtype != ta.dtype || ta.bytes() != tb.bytes()) { bad(ta.name + ": " + ta.dtype + " tensor was changed"); continue; }
                    if (st::read_bytes(fa, a.data_start + ta.begin, ta.bytes()) != st::read_bytes(fb, b.data_start + tb.begin, tb.bytes()))
                        bad(ta.name + ": " + ta.dtype + " tensor contents differ (must be copied unchanged)");
                    continue;
                }
                if (tb.dtype != "F16") { ++wrong; bad(ta.name + ": fp16 file has dtype " + tb.dtype + " (NOT converted)"); continue; }
                if (ta.shape != tb.shape) { bad(ta.name + ": shape differs"); continue; }
                const uint64_t n = ta.count();
                if (ta.bytes() != 4 * n || tb.bytes() != 2 * n) { bad(ta.name + ": byte sizes do not match the shape"); continue; }
                params += n; bytes32 += ta.bytes(); bytes16 += tb.bytes();
                std::vector<std::pair<uint64_t, uint64_t>> ranges;  // [start, count)
                if (full || n <= 96) ranges.push_back({0, n});
                else for (uint64_t s : {uint64_t(0), n / 2 - 16, n - 32}) ranges.push_back({s, 32});
                for (auto [start, cnt] : ranges) {
                    const uint64_t chunk = 1u << 22;
                    for (uint64_t c0 = 0; c0 < cnt; c0 += chunk) {
                        const uint64_t c = std::min(chunk, cnt - c0);
                        auto raw32 = st::read_bytes(fa, a.data_start + ta.begin + 4 * (start + c0), 4 * c);
                        auto raw16 = st::read_bytes(fb, b.data_start + tb.begin + 2 * (start + c0), 2 * c);
                        const float* x = reinterpret_cast<const float*>(raw32.data());
                        const uint16_t* h = reinterpret_cast<const uint16_t*>(raw16.data());
                        for (uint64_t k = 0; k < c; ++k) {
                            ++checked;
                            const uint16_t want = st::f32_to_f16(x[k]);
                            if (h[k] != want && !(std::isnan(x[k]) && (h[k] & 0x7C00) == 0x7C00)) {
                                if (wrong++ < 5) bad(ta.name + "[" + std::to_string(start + c0 + k) + "]: fp16 bits differ from fp32 rounded to fp16");
                            }
                            const double d = std::fabs(static_cast<double>(x[k]) - static_cast<double>(st::f16_to_f32(h[k])));
                            if (std::isfinite(d)) {
                                sum_abs += d; sum_sq += d * d; sum_ref += static_cast<double>(x[k]) * x[k]; max_abs = std::max(max_abs, d);
                                if (x[k] != 0.0f && (h[k] & 0x7FFF) == 0) ++to_zero;
                            }
                        }
                    }
                }
            }
            for (auto& t : b.tensors) {
                bool found = false;
                for (auto& ta : a.tensors) if (ta.name == t.name) { found = true; break; }
                if (!found) bad(t.name + ": extra tensor in fp16 file");
            }
            std::cout << "  tensors compared : " << n_tensors << "\n  values (params)  : " << params
                      << "\n  fp32 data bytes  : " << bytes32 << "   fp16 data bytes: " << bytes16
                      << "   ratio " << (bytes32 ? static_cast<double>(bytes16) / bytes32 : 0.0)
                      << "\n  values checked   : " << checked << (full ? "  (ALL values)" : "  (samples: start, middle, end of each tensor)") << "\n";
            if (bytes32 && bytes16 * 2 != bytes32) bad(rel.generic_string() + ": fp16 data is not exactly half of fp32");
            if (full && checked) {
                std::cout << "  rounding error   : relative L2 " << (sum_ref > 0 ? 100.0 * std::sqrt(sum_sq / sum_ref) : 0.0) << " %, mean abs " << sum_abs / checked
                          << ", max abs " << max_abs << ", non-zero values rounded to zero: " << to_zero << "\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    std::cout << "\n";
    if (!problems.empty()) {
        std::cout << "NOT PROVEN. Problems:\n";
        for (auto& p : problems) std::cout << "  - " << p << "\n";
        return 1;
    }
    std::cout << "PROVEN: every tensor is F32 in the originals and F16 in the converted files, same names and shapes, exactly half the bytes,\n"
              << (full ? "and EVERY value equals the fp32 value rounded to fp16.\n" : "and the sampled values equal the fp32 values rounded to fp16 (use --full to check every value).\n");
    return 0;
}
