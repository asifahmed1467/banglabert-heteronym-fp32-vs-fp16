// fp16_convert.cpp - C++ version of make_fp16.py: rounds every float32 tensor of every .safetensors file to float16.
//
//   fp16_convert --src <fp32 folder> --out <fp16 folder>
//
// Copies every other file unchanged (config, tokenizer, ...). Refuses to write a file if a value would become infinity.
// Output files use the same header text as make_fp16.py, so for the same input both tools write IDENTICAL bytes.
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "safetensors_io.hpp"

namespace fs = std::filesystem;

struct Stats {
    long long converted = 0, kept = 0, overflow = 0, to_zero = 0;
    double max_abs_err = 0.0;
    uint64_t before = 0, after = 0;
};

static Stats convert_file(const fs::path& src, const fs::path& dst) {
    st::File in = st::read_header(src.string());
    Stats s;
    // new header, same layout rules as make_fp16.py
    std::string header = "{\"__metadata__\":";
    if (in.metadata.o.empty()) header += "{\"format\":\"pt\"}";
    else header += st::dump(in.metadata);
    uint64_t offset = 0;
    std::vector<uint64_t> new_size;
    for (const auto& t : in.tensors) {
        const bool f32 = t.dtype == "F32";
        const uint64_t size = f32 ? t.bytes() / 2 : t.bytes();
        if (f32 && t.bytes() != t.count() * 4) throw std::runtime_error(t.name + ": byte size does not match shape");
        new_size.push_back(size);
        header += "," + st::dump_str(t.name) + ":{\"dtype\":\"" + (f32 ? "F16" : t.dtype) + "\",\"shape\":[";
        for (size_t k = 0; k < t.shape.size(); ++k) header += (k ? "," : "") + std::to_string(t.shape[k]);
        header += "],\"data_offsets\":[" + std::to_string(offset) + "," + std::to_string(offset + size) + "]}";
        offset += size;
    }
    header += "}";
    header.append((8 - header.size() % 8) % 8, ' ');  // padded to 8 bytes

    fs::create_directories(dst.parent_path());
    const fs::path tmp = dst.string() + ".part";
    {
        std::ifstream fin(src, std::ios::binary);
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        uint64_t n = header.size();
        unsigned char le[8];
        for (int k = 0; k < 8; ++k) le[k] = static_cast<unsigned char>((n >> (8 * k)) & 0xFF);
        out.write(reinterpret_cast<const char*>(le), 8);
        out.write(header.data(), static_cast<std::streamsize>(header.size()));
        for (size_t idx = 0; idx < in.tensors.size(); ++idx) {
            const auto& t = in.tensors[idx];
            std::vector<uint8_t> raw = st::read_bytes(fin, in.data_start + t.begin, t.bytes());
            if (t.dtype != "F32") {
                out.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
                ++s.kept;
                continue;
            }
            const uint64_t n_val = t.count();
            std::vector<uint16_t> half(n_val);
            const float* x = reinterpret_cast<const float*>(raw.data());
            for (uint64_t k = 0; k < n_val; ++k) {
                const float v = x[k];
                const uint16_t h = st::f32_to_f16(v);
                half[k] = h;
                if (std::isfinite(v)) {
                    if ((h & 0x7FFF) == 0x7C00) { ++s.overflow; continue; }
                    if (v != 0.0f && (h & 0x7FFF) == 0) ++s.to_zero;
                    s.max_abs_err = std::max(s.max_abs_err, static_cast<double>(std::fabs(v - st::f16_to_f32(h))));
                }
            }
            out.write(reinterpret_cast<const char*>(half.data()), static_cast<std::streamsize>(half.size() * 2));
            ++s.converted;
        }
        if (!out) throw std::runtime_error("write failed: " + tmp.string());
    }
    if (s.overflow) {
        fs::remove(tmp);
        throw std::runtime_error(src.filename().string() + ": " + std::to_string(s.overflow) + " value(s) do not fit in fp16 (would become infinity). Refusing.");
    }
    fs::rename(tmp, dst);
    s.before = fs::file_size(src);
    s.after = fs::file_size(dst);
    return s;
}

int main(int argc, char** argv) {
    fs::path src, out;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i];
        if (k == "--src") src = argv[i + 1];
        else if (k == "--out") out = argv[i + 1];
    }
    if (src.empty() || out.empty()) { std::cerr << "usage: fp16_convert --src <fp32 folder> --out <fp16 folder>\n"; return 2; }
    try {
        src = fs::weakly_canonical(src);
        out = fs::weakly_canonical(out);
        if (src == out || !fs::is_directory(src)) { std::cerr << "--src must be an existing folder and differ from --out\n"; return 2; }
        std::vector<fs::path> files;
        for (auto& e : fs::recursive_directory_iterator(src))
            if (e.is_regular_file() && e.path().extension() != ".part") files.push_back(e.path());
        std::sort(files.begin(), files.end());
        bool any = false;
        std::ostringstream report;
        for (const auto& p : files) {
            const fs::path rel = fs::relative(p, src), dst = out / rel;
            if (p.filename() == "MANIFEST.txt") continue;
            if (p.extension() != ".safetensors") { fs::create_directories(dst.parent_path()); fs::copy_file(p, dst, fs::copy_options::overwrite_existing); continue; }
            any = true;
            auto t0 = std::chrono::steady_clock::now();
            Stats s = convert_file(p, dst);
            std::ostringstream line;
            line << rel.generic_string() << ": " << s.converted << " tensors -> fp16, " << s.kept << " kept as they were | "
                 << std::fixed << std::setprecision(1) << s.before / 1e6 << " MB -> " << s.after / 1e6 << " MB | max rounding error "
                 << std::scientific << std::setprecision(2) << s.max_abs_err << " | non-zero values rounded to zero: " << s.to_zero
                 << std::defaultfloat << "  (" << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s)";
            std::cout << line.str() << "\n";
            report << line.str() << "\n";
        }
        if (!any) { std::cerr << "no .safetensors files under " << src << "\n"; return 2; }
        std::ofstream m(out / "MANIFEST_FP16.txt", std::ios::binary);
        m << "fp16 copy made by fp16_convert (C++) from the fp32 files.\nOnly .safetensors float32 tensors were rounded to float16; all other files are unchanged.\n\n" << report.str();
        std::cout << "\nDONE -> " << out.string() << "\nnext: fp16_verify --fp32 " << src.string() << " --fp16 " << out.string() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
