// bench_ort.cpp - C++ inference + benchmark for the Bangla heteronym models (ONNX Runtime).
//
// What it does (the "model half" of src/predict.py, ported to C++):
//   read pre-tokenized inputs  ->  batch the rows of one sentence  ->  run the ONNX model
//   ->  mask the unused reading slots, softmax/argmax  ->  apply reading_fixes
//   ->  rebuild the phoneticized sentence  ->  time it, compare with the Python reference.
// Tokenization and Bangla normalization are NOT done here: make_inputs.py produces them with the
// authors' own Python code (see README.md for why).
//
// Timed scope per sentence (identical in cmp_worker.py for the Python engines):
//   pad the rows into one batch -> model call -> mask/argmax per heteronym -> reading_fixes -> rebuild text.
// Tokenization/normalization are outside the timer for every engine (same pre-tokenized input file).
//
// Run:
//   bench_ort --model m.onnx --inputs inputs.txt --tag cpp --repeat 5 --threads 4 --out result.json
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr float kMasked = -1e4f;  // same value as src/models.py MASKED
constexpr int kSlots = 3;         // N_SLOTS: at most three readings per word

struct Row {
    std::vector<int64_t> ids, mask, types, target;
    int64_t anchor = 0;
};
struct Item {
    std::string word, ref_pron;
    std::vector<std::string> cands;
    std::vector<Row> rows;  // encoder: 1 row; candidate: one row per reading
};
struct Sentence {
    std::string text, ref_output;
    std::vector<std::string> segs;  // n_items + 1 text pieces around the heteronyms
    std::vector<Item> items;
};
struct Inputs {
    std::string kind = "encoder";
    int64_t pad_id = 0;
    std::map<std::string, std::map<std::string, std::string>> fixes;
    std::vector<Sentence> sentences;
};

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == sep) { out.push_back(cur); cur.clear(); }
        else cur += c;
    }
    out.push_back(cur);
    return out;
}

std::vector<int64_t> numbers(const std::string& s) {
    std::vector<int64_t> v;
    std::istringstream in(s);
    long long x;
    while (in >> x) v.push_back(x);
    return v;
}

// Line format is documented in README.md (tab separated, UTF-8).
Inputs read_inputs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open inputs file: " + path);
    Inputs in;
    std::string line;
    Sentence* sent = nullptr;
    Item* item = nullptr;
    Row* row = nullptr;
    size_t want_items = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto p = split(line, '\t');
        const std::string& tag = p[0];
        if (tag == "KIND") in.kind = p.at(1);
        else if (tag == "PAD") in.pad_id = std::stoll(p.at(1));
        else if (tag == "FIX") in.fixes[p.at(1)][p.at(2)] = p.at(3);
        else if (tag == "SENT") {
            in.sentences.emplace_back();
            sent = &in.sentences.back();
            want_items = std::stoul(p.at(1));
            sent->ref_output = p.size() > 2 ? p[2] : "";
            sent->text = p.size() > 3 ? p[3] : "";
            item = nullptr;
            row = nullptr;
        } else if (tag == "SEG") sent->segs.push_back(p.size() > 1 ? p[1] : "");
        else if (tag == "ITEM") {
            sent->items.emplace_back();
            item = &sent->items.back();
            item->word = p.at(1);
            item->ref_pron = p.at(2);
            for (size_t i = 3; i < p.size(); ++i) item->cands.push_back(p[i]);
            row = nullptr;
        } else if (tag == "ROW") { item->rows.emplace_back(); row = &item->rows.back(); }
        else if (tag == "IDS") row->ids = numbers(p.size() > 1 ? p[1] : "");
        else if (tag == "MASK") row->mask = numbers(p.size() > 1 ? p[1] : "");
        else if (tag == "TYPES") row->types = numbers(p.size() > 1 ? p[1] : "");
        else if (tag == "TARGET") row->target = numbers(p.size() > 1 ? p[1] : "");
        else if (tag == "ANCHOR") row->anchor = std::stoll(p.at(1));
        else throw std::runtime_error("unknown line tag: " + tag);
        (void)want_items;
    }
    for (auto& s : in.sentences)
        if (s.segs.size() != s.items.size() + 1)
            throw std::runtime_error("sentence has " + std::to_string(s.segs.size()) + " segments for " +
                                     std::to_string(s.items.size()) + " items: " + s.text);
    return in;
}

class Model {
public:
    Model(const std::string& onnx, int threads, bool candidate)
        : env_(ORT_LOGGING_LEVEL_WARNING, "bench"), candidate_(candidate), mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        Ort::SessionOptions so;
        if (threads > 0) so.SetIntraOpNumThreads(threads);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(env_, std::filesystem::path(onnx).c_str(), so);
    }

    // Phoneticized text of one sentence (all its heteronyms predicted in one model call).
    std::string run(const Sentence& s, const Inputs& in) {
        if (s.items.empty()) return s.segs.empty() ? s.text : s.segs[0];
        std::vector<const Row*> rows;
        for (const auto& it : s.items)
            for (const auto& r : it.rows) rows.push_back(&r);
        const int64_t R = static_cast<int64_t>(rows.size());
        size_t L = 0;
        for (auto* r : rows) L = std::max(L, r->ids.size());
        std::vector<int64_t> ids(R * L, in.pad_id), mask(R * L, 0), types(R * L, 0), target(R * L, 0), anchor(R, 0);
        for (int64_t i = 0; i < R; ++i) {
            const Row& r = *rows[i];
            std::copy(r.ids.begin(), r.ids.end(), ids.begin() + i * L);
            std::copy(r.mask.begin(), r.mask.end(), mask.begin() + i * L);
            std::copy(r.types.begin(), r.types.end(), types.begin() + i * L);
            std::copy(r.target.begin(), r.target.end(), target.begin() + i * L);
            anchor[i] = r.anchor;
        }
        const int64_t shape2[2] = {R, static_cast<int64_t>(L)};
        const int64_t shape1[1] = {R};
        auto t2 = [&](std::vector<int64_t>& v) { return Ort::Value::CreateTensor<int64_t>(mem_, v.data(), v.size(), shape2, 2); };
        std::vector<Ort::Value> feeds;
        feeds.push_back(t2(ids));
        feeds.push_back(t2(mask));
        feeds.push_back(t2(types));
        feeds.push_back(Ort::Value::CreateTensor<int64_t>(mem_, anchor.data(), anchor.size(), shape1, 1));
        feeds.push_back(t2(target));
        static const char* in_names[] = {"input_ids", "attention_mask", "token_type_ids", "anchor", "target_mask"};
        static const char* out_names[] = {"score"};
        auto out = session_->Run(Ort::RunOptions{nullptr}, in_names, feeds.data(), feeds.size(), out_names, 1);
        const float* sc = out[0].GetTensorData<float>();
        const size_t n_out = static_cast<size_t>(out[0].GetTensorTypeAndShapeInfo().GetShape()[1]);

        std::string result = s.segs[0];
        size_t row_at = 0;
        for (size_t k = 0; k < s.items.size(); ++k) {
            const Item& it = s.items[k];
            float logits[kSlots];
            for (int j = 0; j < kSlots; ++j) logits[j] = kMasked;
            if (candidate_) {
                for (size_t j = 0; j < it.rows.size() && j < kSlots; ++j) logits[j] = sc[(row_at + j) * n_out];
            } else {
                for (size_t j = 0; j < it.cands.size() && j < std::min<size_t>(kSlots, n_out); ++j) logits[j] = sc[row_at * n_out + j];
            }
            row_at += it.rows.size();
            int best = 0;  // argmax of softmax == argmax of logits (first maximum)
            for (int j = 1; j < kSlots; ++j)
                if (logits[j] > logits[best]) best = j;
            std::string pron = best < static_cast<int>(it.cands.size()) ? it.cands[best] : "";
            auto f = in.fixes.find(it.word);
            if (f != in.fixes.end()) {
                auto g = f->second.find(pron);
                if (g != f->second.end()) pron = g->second;
            }
            result += pron.empty() ? it.word : pron;
            result += s.segs[k + 1];
        }
        return result;
    }

private:
    Ort::Env env_;
    bool candidate_;
    Ort::MemoryInfo mem_;
    std::unique_ptr<Ort::Session> session_;
};

// Resident memory of this process in MB from /proc (Linux); -1 when unavailable.
double proc_status_mb(const char* key) {
    std::ifstream f("/proc/self/status");
    std::string line;
    const std::string k = std::string(key) + ":";
    while (std::getline(f, line))
        if (line.compare(0, k.size(), k) == 0) return std::stod(line.substr(k.size())) / 1024.0;
    return -1.0;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += static_cast<char>(c);  // UTF-8 bytes pass through
    }
    return o;
}

double percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    if (v.size() == 1) return v[0];
    double pos = (v.size() - 1) * q;
    size_t lo = static_cast<size_t>(std::floor(pos)), hi = static_cast<size_t>(std::ceil(pos));
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

}  // namespace

int main(int argc, char** argv) {
    std::map<std::string, std::string> a{{"repeat", "5"}, {"threads", "0"}, {"tag", "run"}, {"out", "result.json"}};
    for (int i = 1; i + 1 < argc; i += 2) a[std::string(argv[i]).substr(2)] = argv[i + 1];
    if (!a.count("model") || !a.count("inputs")) {
        std::cerr << "usage: bench_ort --model m.onnx --inputs inputs.txt [--tag fp32] [--repeat 5] [--threads 0] [--out result.json]\n";
        return 2;
    }
    try {
        Inputs in = read_inputs(a["inputs"]);
        const int threads = std::stoi(a["threads"]);
        const int repeat = std::max(1, std::stoi(a["repeat"]));
        using clk = std::chrono::steady_clock;
        auto ms_since = [](clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); };

        const auto t_load = clk::now();
        Model model(a["model"], threads, in.kind == "candidate");
        const double load_s = ms_since(t_load) / 1000.0;

        // Warm-up: every sentence that reaches the model runs once, untimed (ORT sizes its buffers per input shape).
        double first_call_ms = 0.0;
        bool first = true;
        for (const auto& s : in.sentences) {
            if (s.items.empty()) continue;
            const auto t = clk::now();
            model.run(s, in);
            if (first) { first_call_ms = ms_since(t); first = false; }
        }
        const double rss_after_warmup = proc_status_mb("VmRSS");

        std::vector<std::string> outputs;
        std::vector<double> per_sentence, used;
        std::vector<std::vector<double>> per_call;
        size_t same = 0, compared = 0;
        for (const auto& s : in.sentences) {
            std::vector<double> times;
            std::string out;
            if (s.items.empty()) {  // never reaches the model: not timed, for every engine
                out = model.run(s, in);
                per_sentence.push_back(0.0);
                per_call.emplace_back();
            } else {
                for (int r = 0; r < repeat; ++r) {
                    const auto t0 = clk::now();
                    out = model.run(s, in);
                    times.push_back(ms_since(t0));
                }
                per_call.push_back(times);
                std::sort(times.begin(), times.end());
                per_sentence.push_back(times[times.size() / 2]);
                used.insert(used.end(), times.begin(), times.end());
            }
            outputs.push_back(out);
            if (!s.ref_output.empty()) { ++compared; same += (out == s.ref_output); }
        }
        if (used.empty()) used.push_back(0.0);
        double mean = std::accumulate(used.begin(), used.end(), 0.0) / used.size();
        double sq = 0;
        for (double x : used) sq += (x - mean) * (x - mean);

        std::ofstream o(a["out"], std::ios::binary);
        o << std::setprecision(10);
        o << "{\n \"tag\": \"" << json_escape(a["tag"]) << "\",\n \"engine\": \"onnxruntime-cpp\",\n"
          << " \"ort_version\": \"" << Ort::GetVersionString() << "\",\n \"threads\": " << threads << ",\n"
          << " \"import_s\": 0,\n \"load_s\": " << load_s << ",\n \"first_call_ms\": " << first_call_ms << ",\n"
          << " \"startup_s\": " << (load_s + first_call_ms / 1000.0) << ",\n"
          << " \"rss_mb\": " << rss_after_warmup << ",\n \"peak_rss_mb\": " << proc_status_mb("VmHWM") << ",\n"
          << " \"repeat\": " << repeat << ",\n \"n_sentences\": " << in.sentences.size() << ",\n"
          << " \"n_model_sentences\": " << std::count_if(in.sentences.begin(), in.sentences.end(), [](const Sentence& s) { return !s.items.empty(); }) << ",\n"
          << " \"median_ms\": " << percentile(used, 0.5) << ",\n \"mean_ms\": " << mean << ",\n"
          << " \"p95_ms\": " << percentile(used, 0.95) << ",\n \"min_ms\": " << *std::min_element(used.begin(), used.end())
          << ",\n \"max_ms\": " << *std::max_element(used.begin(), used.end()) << ",\n"
          << " \"stdev_ms\": " << std::sqrt(sq / used.size()) << ",\n \"throughput_sps\": " << (mean > 0 ? 1000.0 / mean : 0.0) << ",\n"
          << " \"same_as_python\": " << same << ",\n \"compared_with_python\": " << compared << ",\n \"per_sentence_ms\": [";
        for (size_t i = 0; i < per_sentence.size(); ++i) o << (i ? ", " : "") << per_sentence[i];
        o << "],\n \"per_call_ms\": [";
        for (size_t i = 0; i < per_call.size(); ++i) {
            o << (i ? ", [" : "[");
            for (size_t j = 0; j < per_call[i].size(); ++j) o << (j ? ", " : "") << per_call[i][j];
            o << "]";
        }
        o << "],\n \"outputs\": [";
        for (size_t i = 0; i < outputs.size(); ++i) o << (i ? ",\n  " : "\n  ") << "\"" << json_escape(outputs[i]) << "\"";
        o << "\n ]\n}\n";

        std::cout << "tag " << a["tag"] << ": median " << percentile(used, 0.5) << " ms, p95 " << percentile(used, 0.95)
                  << " ms, same output as Python reference on " << same << " of " << compared << " sentences -> " << a["out"] << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
