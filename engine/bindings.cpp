// pybind11 bindings. For now they expose CPU detection and the quantization formats so
// that Python tests can check the C++ and Python quantizers byte for byte, and check the
// SIMD kernels against their scalar reference.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "specdraft/cpu.hpp"
#include "specdraft/model.hpp"
#include "specdraft/model_file.hpp"
#include "specdraft/quant.hpp"
#include "specdraft/sampling.hpp"
#include "specdraft/speculative.hpp"
#include "specdraft/threadpool.hpp"

namespace py = pybind11;
using namespace specdraft;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using TokenArray = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

int checked_token_count(const TokenArray& tokens) {
    if (tokens.ndim() != 1 || tokens.size() == 0) {
        throw std::invalid_argument("tokens must be a non-empty 1-D int32 array");
    }
    return static_cast<int>(tokens.size());
}

py::dict config_as_dict(const ModelConfig& c) {
    py::dict out;
    out["vocab_size"] = c.vocab_size;
    out["hidden_size"] = c.hidden_size;
    out["intermediate_size"] = c.intermediate_size;
    out["num_hidden_layers"] = c.num_hidden_layers;
    out["num_attention_heads"] = c.num_attention_heads;
    out["num_key_value_heads"] = c.num_key_value_heads;
    out["head_dim"] = c.head_dim;
    out["vocab_limit"] = c.vocab_limit;
    out["output_vocab"] = c.output_vocab;
    out["tie_word_embeddings"] = c.tie_word_embeddings != 0;
    out["rms_norm_eps"] = c.rms_norm_eps;
    out["rope_theta"] = c.rope_theta;
    return out;
}

int checked_length(const FloatArray& x) {
    if (x.size() % QK != 0 || x.size() == 0) {
        throw std::invalid_argument("input length must be a positive multiple of 32");
    }
    return static_cast<int>(x.size());
}

// Python bytes are not guaranteed to be aligned for these structs, so copy into a
// vector before casting. This path is only used by tests and tooling.
template <typename Block>
std::vector<Block> blocks_from_bytes(const py::bytes& blob, const char* what) {
    const std::string data = blob;
    if (data.size() % sizeof(Block) != 0 || data.empty()) {
        throw std::invalid_argument(std::string("blob size is not a whole number of ") + what +
                                    " blocks");
    }
    std::vector<Block> blocks(data.size() / sizeof(Block));
    std::memcpy(blocks.data(), data.data(), data.size());
    return blocks;
}

template <typename Block>
py::bytes blocks_to_bytes(const std::vector<Block>& blocks) {
    return py::bytes(reinterpret_cast<const char*>(blocks.data()), blocks.size() * sizeof(Block));
}

template <typename Block, void (*Quantize)(const float*, int, Block*)>
py::bytes quantize_py(const FloatArray& x) {
    const int n = checked_length(x);
    std::vector<Block> out(n / QK);
    Quantize(x.data(), n, out.data());
    return blocks_to_bytes(out);
}

template <typename Block, void (*Dequantize)(const Block*, int, float*)>
FloatArray dequantize_py(const py::bytes& blob, const char* what) {
    const std::vector<Block> blocks = blocks_from_bytes<Block>(blob, what);
    const int n = static_cast<int>(blocks.size()) * QK;
    FloatArray out(n);
    Dequantize(blocks.data(), n, out.mutable_data());
    return out;
}

template <typename WBlock, float (*Dot)(const WBlock*, const BlockA8*, int)>
float dot_py(const py::bytes& weights, const py::bytes& activations, const char* what) {
    const std::vector<WBlock> w = blocks_from_bytes<WBlock>(weights, what);
    const std::vector<BlockA8> x = blocks_from_bytes<BlockA8>(activations, "A8");
    if (w.size() != x.size()) {
        throw std::invalid_argument("weight and activation blobs have different block counts");
    }
    return Dot(w.data(), x.data(), static_cast<int>(w.size()));
}

// One weight row against k activation vectors, which must be k whole copies of the row's
// block count, laid out one after another.
template <typename WBlock, void (*Dot)(const WBlock*, const BlockA8*, int, int, float*)>
py::array_t<float> dot_multi_py(const py::bytes& weights, const py::bytes& activations,
                                const char* what) {
    const std::vector<WBlock> w = blocks_from_bytes<WBlock>(weights, what);
    const std::vector<BlockA8> x = blocks_from_bytes<BlockA8>(activations, "A8");
    if (w.empty() || x.size() % w.size() != 0) {
        throw std::invalid_argument("activations must be a whole number of vectors");
    }
    const int tokens = static_cast<int>(x.size() / w.size());
    py::array_t<float> out(tokens);
    Dot(w.data(), x.data(), static_cast<int>(w.size()), tokens, out.mutable_data());
    return out;
}

py::dict stats_as_dict(const DecodeStats& stats) {
    py::dict out;
    out["emitted"] = stats.emitted;
    out["rounds"] = stats.rounds;
    out["proposed"] = stats.proposed;
    out["accepted"] = stats.accepted;
    out["rejections"] = stats.rejections;
    out["target_forwards"] = stats.target_forwards;
    out["draft_forwards"] = stats.draft_forwards;
    out["seconds"] = stats.seconds;
    out["accepted_lengths"] = stats.accepted_lengths;
    out["tokens_per_target_forward"] = stats.tokens_per_target_forward();
    out["alpha"] = stats.alpha();
    out["tokens_per_second"] = stats.tokens_per_second();
    return out;
}

GenerateOptions make_options(int max_new_tokens, int gamma, float temperature, int top_k,
                             float top_p, const std::vector<int32_t>& stop, uint64_t seed,
                             float confidence_threshold = 0.0f) {
    GenerateOptions options;
    options.max_new_tokens = max_new_tokens;
    options.gamma = gamma;
    options.sampling.temperature = temperature;
    options.sampling.top_k = top_k;
    options.sampling.top_p = top_p;
    options.stop = stop;
    options.seed = seed;
    options.confidence_threshold = confidence_threshold;
    return options;
}

}  // namespace

PYBIND11_MODULE(_engine, m) {
    m.doc() = "specdraft CPU engine";

    py::class_<Model>(m, "Model", "Qwen3 inference over a memory-mapped weights file")
        .def(py::init([](const std::string& path, int max_positions, int threads,
                         const std::string& cores, bool dynamic_schedule, int max_batch) {
                 EngineOptions options;
                 options.max_positions = max_positions;
                 options.threads = threads;
                 options.dynamic_schedule = dynamic_schedule;
                 options.max_batch = max_batch;
                 if (!parse_core_selection(cores.c_str(), &options.cores)) {
                     throw std::invalid_argument(
                         "cores must be any, performance, physical or logical");
                 }
                 return std::make_unique<Model>(ModelFile::open(path), options);
             }),
             py::arg("path"), py::arg("max_positions") = 2048, py::arg("threads") = 0,
             py::arg("cores") = "performance", py::arg("dynamic_schedule") = false,
             py::arg("max_batch") = 16)
        .def_property_readonly("max_batch", &Model::max_batch)
        .def_property_readonly("logit_count", &Model::logit_count,
                               "Logits per scored token: fewer when the output layer was trimmed.")
        .def_property_readonly("trimmed_vocabulary", &Model::trimmed_vocabulary)
        .def(
            "token_for_logit",
            [](const Model& model, uint32_t index) {
                if (index >= model.logit_count()) {
                    throw std::out_of_range("logit index out of range");
                }
                return model.token_for_logit(index);
            },
            py::arg("index"), "Which token a logit refers to; the identity unless trimmed.")
        .def_property_readonly("threads", &Model::threads)
        .def_property_readonly("core_selection", &Model::core_selection)
        .def_property_readonly("dynamic_schedule", &Model::dynamic_schedule)
        .def_property_readonly("weight_bytes_per_token", &Model::weight_bytes_per_token)
        .def_property_readonly("kv_bytes_per_token", &Model::kv_bytes_per_token)
        .def("set_timing", &Model::set_timing, py::arg("enabled"))
        .def("reset_timings", &Model::reset_timings)
        .def(
            "timings",
            [](const Model& model) {
                py::dict out;
                double total = 0.0;
                for (int i = 0; i < Model::kStageCount; ++i) {
                    const auto stage = static_cast<Model::Stage>(i);
                    const double seconds = model.stage_seconds(stage);
                    out[py::str(Model::stage_name(stage))] = seconds;
                    total += seconds;
                }
                out["total"] = total;
                out["tokens"] = model.timed_tokens();
                return out;
            },
            "Seconds spent in each stage since the last reset, plus the token count.")
        .def_property_readonly("config", [](const Model& model) { return config_as_dict(model.config()); })
        .def_property_readonly("max_positions", &Model::max_positions)
        .def_property_readonly("pos", &Model::pos, "how many positions the KV cache holds")
        .def("set_pos", &Model::set_pos, py::arg("position"),
             "Roll the cache back; attention then ignores everything past this point.")
        .def("reset", &Model::reset)
        .def(
            "forward",
            [](Model& model, const TokenArray& tokens, bool all_logits) {
                const int k = checked_token_count(tokens);
                const auto rows = all_logits ? k : 1;
                py::array_t<float> out({rows, static_cast<int>(model.logit_count())});
                const int32_t* ids = tokens.data();
                float* destination = out.mutable_data();
                {
                    py::gil_scoped_release unlocked;
                    model.forward(ids, k, destination, all_logits);
                }
                return out;
            },
            py::arg("tokens"), py::arg("all_logits") = false,
            "Run k tokens at the current position and return logits [k or 1, vocab_limit].")
        .def(
            "forward_capture",
            [](Model& model, const TokenArray& tokens) {
                const int k = checked_token_count(tokens);
                const ModelConfig& c = model.config();
                py::array_t<float> out({static_cast<int>(c.num_hidden_layers), k,
                                        static_cast<int>(c.hidden_size)});
                const int32_t* ids = tokens.data();
                float* destination = out.mutable_data();
                {
                    py::gil_scoped_release unlocked;
                    model.forward(ids, k, nullptr, false, destination);
                }
                return out;
            },
            py::arg("tokens"),
            "Run k tokens and return the hidden state after every layer [layers, k, hidden].");

    m.def(
        "dot_q4_a8_multi",
        [](const py::bytes& w, const py::bytes& x) {
            return dot_multi_py<BlockQ4, dot_q4_a8_multi>(w, x, "Q4");
        },
        py::arg("weights"), py::arg("activations"),
        "One weight row against several activation vectors, sharing one pass over the weights.");
    m.def(
        "dot_q8_a8_multi",
        [](const py::bytes& w, const py::bytes& x) {
            return dot_multi_py<BlockQ8, dot_q8_a8_multi>(w, x, "Q8");
        },
        py::arg("weights"), py::arg("activations"));

    m.def(
        "generate_plain",
        [](Model& model, const std::vector<int32_t>& prompt, int max_new_tokens, float temperature,
           int top_k, float top_p, const std::vector<int32_t>& stop, uint64_t seed) {
            const GenerateOptions options =
                make_options(max_new_tokens, 1, temperature, top_k, top_p, stop, seed);
            DecodeStats stats;
            std::vector<int32_t> tokens;
            {
                py::gil_scoped_release unlocked;
                tokens = generate_plain(model, prompt, options, &stats);
            }
            return py::make_tuple(tokens, stats_as_dict(stats));
        },
        py::arg("model"), py::arg("prompt"), py::arg("max_new_tokens") = 64,
        py::arg("temperature") = 0.0f, py::arg("top_k") = 0, py::arg("top_p") = 1.0f,
        py::arg("stop") = std::vector<int32_t>{}, py::arg("seed") = uint64_t{0},
        "Token-at-a-time decoding entirely inside the engine. Returns (tokens, stats).");

    m.def(
        "generate_speculative",
        [](Model& target, Model& draft, const std::vector<int32_t>& prompt, int max_new_tokens,
           int gamma, float temperature, int top_k, float top_p, const std::vector<int32_t>& stop,
           uint64_t seed, float confidence_threshold) {
            const GenerateOptions options = make_options(max_new_tokens, gamma, temperature, top_k,
                                                         top_p, stop, seed, confidence_threshold);
            DecodeStats stats;
            std::vector<int32_t> tokens;
            {
                py::gil_scoped_release unlocked;
                tokens = generate_speculative(target, draft, prompt, options, &stats);
            }
            return py::make_tuple(tokens, stats_as_dict(stats));
        },
        py::arg("target"), py::arg("draft"), py::arg("prompt"), py::arg("max_new_tokens") = 64,
        py::arg("gamma") = 4, py::arg("temperature") = 0.0f, py::arg("top_k") = 0,
        py::arg("top_p") = 1.0f, py::arg("stop") = std::vector<int32_t>{},
        py::arg("seed") = uint64_t{0}, py::arg("confidence_threshold") = 0.0f,
        "Speculative decoding entirely inside the engine. Returns (tokens, stats).");

    m.def(
        "generate_prompt_lookup",
        [](Model& target, const std::vector<int32_t>& prompt, int max_new_tokens, int gamma,
           float temperature, int top_k, float top_p, const std::vector<int32_t>& stop,
           uint64_t seed, int max_ngram) {
            const GenerateOptions options =
                make_options(max_new_tokens, gamma, temperature, top_k, top_p, stop, seed);
            DecodeStats stats;
            std::vector<int32_t> tokens;
            {
                py::gil_scoped_release unlocked;
                tokens = generate_prompt_lookup(target, prompt, options, max_ngram, &stats);
            }
            return py::make_tuple(tokens, stats_as_dict(stats));
        },
        py::arg("target"), py::arg("prompt"), py::arg("max_new_tokens") = 64, py::arg("gamma") = 4,
        py::arg("temperature") = 0.0f, py::arg("top_k") = 0, py::arg("top_p") = 1.0f,
        py::arg("stop") = std::vector<int32_t>{}, py::arg("seed") = uint64_t{0},
        py::arg("max_ngram") = 3,
        "Guesses copied from earlier in the text instead of from a draft model: no model work, "
        "so c is effectively zero. Returns (tokens, stats).");

    m.def(
        "warp_to_probs",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> logits, float temperature,
           int top_k, float top_p) {
            SamplingConfig config;
            config.temperature = temperature;
            config.top_k = top_k;
            config.top_p = top_p;
            py::array_t<float> out(logits.size());
            std::memcpy(out.mutable_data(), logits.data(),
                        static_cast<size_t>(logits.size()) * sizeof(float));
            std::vector<int> scratch;
            warp_to_probs(out.mutable_data(), static_cast<int>(out.size()), config, scratch);
            return out;
        },
        py::arg("logits"), py::arg("temperature") = 1.0f, py::arg("top_k") = 0,
        py::arg("top_p") = 1.0f, "The engine's sampling warps, for testing against Python's.");

    m.def(
        "accept_or_resample",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> p,
           py::array_t<float, py::array::c_style | py::array::forcecast> q,
           const std::vector<int32_t>& guesses, bool greedy, uint64_t seed) {
            const int gamma = static_cast<int>(guesses.size());
            if (p.ndim() != 2 || q.ndim() != 2 || p.shape(0) != gamma + 1 || q.shape(0) != gamma ||
                p.shape(1) != q.shape(1)) {
                throw std::invalid_argument("expected p [gamma+1, V] and q [gamma, V]");
            }
            const int vocab = static_cast<int>(p.shape(1));
            Rng rng(seed);
            std::vector<float> scratch;
            const Verdict verdict = accept_or_resample(p.data(), vocab, q.data(), vocab,
                                                       guesses.data(), gamma, vocab, greedy, rng,
                                                       scratch);
            return py::make_tuple(verdict.accepted, verdict.next_token);
        },
        py::arg("p"), py::arg("q"), py::arg("guesses"), py::arg("greedy") = false,
        py::arg("seed") = uint64_t{0},
        "The engine's acceptance rule, for testing against the Python oracle.");

    m.def(
        "core_topology",
        [] {
            py::list out;
            for (const CoreInfo& core : core_topology()) {
                py::dict entry;
                entry["logical_index"] = core.logical_index;
                entry["core_index"] = core.core_index;
                entry["efficiency_class"] = core.efficiency_class;
                entry["primary"] = core.primary;
                out.append(entry);
            }
            return out;
        },
        "Every logical processor, with its physical core and how fast a core it is.");

    m.def(
        "cores_for",
        [](const std::string& selection) {
            CoreSelection parsed;
            if (!parse_core_selection(selection.c_str(), &parsed)) {
                throw std::invalid_argument("unknown core selection");
            }
            return cores_for(parsed);
        },
        py::arg("selection"), "Which logical processors a selection would pin threads to.");

    m.def(
        "measure_read_bandwidth",
        [](size_t bytes, int threads, const std::string& selection, int repeats) {
            CoreSelection parsed;
            if (!parse_core_selection(selection.c_str(), &parsed)) {
                throw std::invalid_argument("unknown core selection");
            }
            py::gil_scoped_release unlocked;
            return measure_read_bandwidth(bytes, threads, parsed, repeats);
        },
        py::arg("bytes") = size_t{1} << 30, py::arg("threads") = 0,
        py::arg("selection") = "performance", py::arg("repeats") = 3,
        "Read bandwidth in GB/s: the ceiling tokens-per-second is measured against.");

    m.def(
        "measure_dispatch_overhead",
        [](int jobs, int threads, const std::string& selection) {
            CoreSelection parsed;
            if (!parse_core_selection(selection.c_str(), &parsed)) {
                throw std::invalid_argument("unknown core selection");
            }
            py::gil_scoped_release unlocked;
            return measure_dispatch_overhead(jobs, threads, parsed);
        },
        py::arg("jobs") = 2000, py::arg("threads") = 0, py::arg("selection") = "performance",
        "Seconds per empty parallel job: the cost of a synchronization point.");

    m.def(
        "model_file_info",
        [](const std::string& path) {
            ModelFile file = ModelFile::open(path);
            py::dict tensors;
            for (const std::string& name : file.names()) {
                const Tensor& tensor = file.tensor(name);
                tensors[py::str(name)] =
                    py::make_tuple(format_name(tensor.format), tensor.rows, tensor.cols, tensor.nbytes);
            }
            py::dict out;
            out["config"] = config_as_dict(file.config());
            out["size_bytes"] = file.size_bytes();
            out["tensors"] = tensors;
            return out;
        },
        py::arg("path"), "Read a weights file's header and directory without loading a model.");

    m.attr("QK") = QK;
    m.attr("BLOCK_SIZES") = std::map<std::string, size_t>{
        {"q4", sizeof(BlockQ4)}, {"q8", sizeof(BlockQ8)}, {"a8", sizeof(BlockA8)}};

    m.def("cpu_features", [] {
        const CpuFeatures& f = cpu_features();
        return std::map<std::string, bool>{{"avx2", f.avx2},
                                           {"fma", f.fma},
                                           {"f16c", f.f16c},
                                           {"avx_vnni", f.avx_vnni},
                                           {"avx512f", f.avx512f}};
    });
    m.def("kernel_path", [] { return std::string(active_kernel_path()); });
    m.def("set_force_scalar", &set_force_scalar, py::arg("force"),
          "Use the scalar kernels even where AVX-VNNI exists, so the reference path stays tested.");
    m.def("force_scalar", &force_scalar);

    m.def("fp32_to_fp16", &fp32_to_fp16, py::arg("value"));
    m.def("fp16_to_fp32", &fp16_to_fp32, py::arg("bits"));

    m.def("quantize_q4", &quantize_py<BlockQ4, quantize_q4>, py::arg("x"));
    m.def("quantize_q8", &quantize_py<BlockQ8, quantize_q8>, py::arg("x"));
    m.def("quantize_a8", &quantize_py<BlockA8, quantize_a8>, py::arg("x"));

    m.def(
        "dequantize_q4", [](const py::bytes& b) { return dequantize_py<BlockQ4, dequantize_q4>(b, "Q4"); },
        py::arg("blob"));
    m.def(
        "dequantize_q8", [](const py::bytes& b) { return dequantize_py<BlockQ8, dequantize_q8>(b, "Q8"); },
        py::arg("blob"));
    m.def(
        "dequantize_a8", [](const py::bytes& b) { return dequantize_py<BlockA8, dequantize_a8>(b, "A8"); },
        py::arg("blob"));

    m.def(
        "dot_q4_a8",
        [](const py::bytes& w, const py::bytes& x) { return dot_py<BlockQ4, dot_q4_a8>(w, x, "Q4"); },
        py::arg("weights"), py::arg("activations"));
    m.def(
        "dot_q4_a8_scalar",
        [](const py::bytes& w, const py::bytes& x) {
            return dot_py<BlockQ4, dot_q4_a8_scalar>(w, x, "Q4");
        },
        py::arg("weights"), py::arg("activations"));
    m.def(
        "dot_q8_a8",
        [](const py::bytes& w, const py::bytes& x) { return dot_py<BlockQ8, dot_q8_a8>(w, x, "Q8"); },
        py::arg("weights"), py::arg("activations"));
    m.def(
        "dot_q8_a8_scalar",
        [](const py::bytes& w, const py::bytes& x) {
            return dot_py<BlockQ8, dot_q8_a8_scalar>(w, x, "Q8");
        },
        py::arg("weights"), py::arg("activations"));
}
