// pybind11 bindings. For now they expose CPU detection and the quantization formats so
// that Python tests can check the C++ and Python quantizers byte for byte, and check the
// SIMD kernels against their scalar reference.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <chrono>
#include <cstring>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "specdraft/cpu.hpp"
#include "specdraft/model.hpp"
#include "specdraft/model_file.hpp"
#include "specdraft/quant.hpp"
#include "specdraft/simd.hpp"
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

// What attention's bandwidth goes on, separated from the rest of a forward pass.
//
// At context 2048 attention is 27% of a decode step and reads the KV cache at about 11 GB/s on a
// machine that streams 39.8. The layout is contiguous per (layer, kv head) and each cached row is
// already shared across its group of query heads, so neither of those is the cause. What is left is
// how the work is cut up and how the inner loop reads it, and those are what this measures.
//
//   scan     convert the cached rows and sum them, nothing else: the floor this access pattern
//            allows, and the number to compare the rest against
//   engine   what Model::attention does: one position at a time, convert K, a dot per query head
//            into a per-head scores row, then a second pass converting V and accumulating
//   blocked  the same work with positions taken in blocks, so several cache rows are in flight at
//            once instead of one
//
// `jobs_per_head` cuts each kv head's positions into that many jobs. The engine uses 1, which gives
// 8 jobs for 6 workers and cannot balance; this says what that costs.
enum class AttnShape { scan, engine, blocked, fused, grouped };

py::dict bench_attention(int positions, int layers, int kv_heads, int group, int head_dim,
                         int jobs_per_head, int block, int iters, int threads,
                         const std::string& cores, const std::string& shape) {
    if (positions < 1 || layers < 1 || kv_heads < 1 || group < 1 || head_dim < 1 ||
        jobs_per_head < 1 || block < 1 || iters < 1 || threads < 1) {
        throw std::invalid_argument("every size must be positive");
    }
    if (head_dim % 8 != 0) {
        throw std::invalid_argument("head_dim must be a multiple of 8");
    }
    AttnShape kind = AttnShape::engine;
    if (shape == "scan") {
        kind = AttnShape::scan;
    } else if (shape == "blocked") {
        kind = AttnShape::blocked;
    } else if (shape == "fused") {
        kind = AttnShape::fused;
    } else if (shape == "grouped") {
        kind = AttnShape::grouped;
    } else if (shape != "engine") {
        throw std::invalid_argument(
            "shape must be \"scan\", \"engine\", \"blocked\", \"fused\" or \"grouped\"");
    }
    CoreSelection selection = CoreSelection::performance;
    if (!parse_core_selection(cores.c_str(), &selection)) {
        throw std::invalid_argument("unknown core selection " + cores);
    }

    const int heads = kv_heads * group;
    const size_t per_layer = static_cast<size_t>(kv_heads) * positions * head_dim;
    const double cache_bytes = 2.0 * layers * per_layer * sizeof(uint16_t);  // keys and values

    // One buffer per cache, laid out exactly as the engine lays its own out: layer, then kv head,
    // then position. Sized for every layer so the sweep streams from memory rather than from L3,
    // which is the condition a real decode step runs in.
    std::vector<uint16_t> keys(static_cast<size_t>(layers) * per_layer);
    std::vector<uint16_t> values(keys.size());
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = fp32_to_fp16(uniform(rng));
        values[i] = fp32_to_fp16(uniform(rng));
    }
    std::vector<float> queries(static_cast<size_t>(heads) * head_dim);
    for (float& value : queries) {
        value = uniform(rng);
    }
    std::vector<float> scores(static_cast<size_t>(heads) * positions, 0.0f);

    // Per-worker scratch and output, each padded to its own cache line.
    constexpr size_t kLine = 16;  // floats
    const size_t scratch_stride = static_cast<size_t>(block) * head_dim * 2 + kLine;
    std::vector<float> scratch(static_cast<size_t>(threads) * scratch_stride);
    const size_t out_stride = ((static_cast<size_t>(heads) * head_dim + kLine - 1) / kLine) * kLine;
    std::vector<float> out(static_cast<size_t>(threads) * out_stride, 0.0f);
    std::vector<double> checksums(static_cast<size_t>(threads) * kLine, 0.0);

    const int jobs = kv_heads * jobs_per_head;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    double seconds = 0.0;
    {
        py::gil_scoped_release unlocked;
        ThreadPool pool(threads, selection);

        int layer = 0;
        const auto body = [&](int begin, int end, int worker) {
            float* keys_buffer = scratch.data() + static_cast<size_t>(worker) * scratch_stride;
            float* values_buffer = keys_buffer + static_cast<size_t>(block) * head_dim;
            float* output = out.data() + static_cast<size_t>(worker) * out_stride;
            double sum = 0.0;

            for (int job = begin; job < end; ++job) {
                const int kv = job / jobs_per_head;
                const int slice = job % jobs_per_head;
                const int lo = static_cast<int>(static_cast<int64_t>(positions) * slice / jobs_per_head);
                const int hi = static_cast<int>(static_cast<int64_t>(positions) * (slice + 1) / jobs_per_head);
                const size_t base = static_cast<size_t>(layer) * per_layer +
                                    static_cast<size_t>(kv) * positions * head_dim;
                const int first_head = kv * group;

                if (kind == AttnShape::scan) {
                    for (int t = lo; t < hi; ++t) {
                        const size_t at = base + static_cast<size_t>(t) * head_dim;
                        fp16_to_fp32_many(keys.data() + at, keys_buffer, head_dim);
                        fp16_to_fp32_many(values.data() + at, values_buffer, head_dim);
                        sum += keys_buffer[0] + values_buffer[0];
                    }
                } else if (kind == AttnShape::engine) {
                    for (int t = lo; t < hi; ++t) {
                        const size_t at = base + static_cast<size_t>(t) * head_dim;
                        fp16_to_fp32_many(keys.data() + at, keys_buffer, head_dim);
                        for (int g = 0; g < group; ++g) {
                            const int h = first_head + g;
                            scores[static_cast<size_t>(h) * positions + t] =
                                dot_f32(queries.data() + static_cast<size_t>(h) * head_dim,
                                        keys_buffer, head_dim) * scale;
                        }
                    }
                    for (int t = lo; t < hi; ++t) {
                        const size_t at = base + static_cast<size_t>(t) * head_dim;
                        fp16_to_fp32_many(values.data() + at, values_buffer, head_dim);
                        for (int g = 0; g < group; ++g) {
                            const int h = first_head + g;
                            accumulate_scaled(output + static_cast<size_t>(h) * head_dim,
                                              values_buffer,
                                              scores[static_cast<size_t>(h) * positions + t],
                                              head_dim);
                        }
                    }
                } else if (kind == AttnShape::grouped) {
                    // One conversion of each cached row, shared by the group of query heads that
                    // read it. The queries for a group are contiguous, so one stride reaches them.
                    float* row_scores = keys_buffer;  // group values, not a converted row
                    for (int t = lo; t < hi; ++t) {
                        const size_t at = base + static_cast<size_t>(t) * head_dim;
                        dot_f16_f32_group(keys.data() + at,
                                          queries.data() + static_cast<size_t>(first_head) * head_dim,
                                          head_dim, group, head_dim, row_scores);
                        for (int g = 0; g < group; ++g) {
                            scores[static_cast<size_t>(first_head + g) * positions + t] =
                                row_scores[g] * scale;
                        }
                    }
                    for (int t = lo; t < hi; ++t) {
                        const size_t at = base + static_cast<size_t>(t) * head_dim;
                        for (int g = 0; g < group; ++g) {
                            row_scores[g] =
                                scores[static_cast<size_t>(first_head + g) * positions + t];
                        }
                        accumulate_scaled_f16_group(
                            output + static_cast<size_t>(first_head) * head_dim, head_dim,
                            values.data() + at, row_scores, group, head_dim);
                    }
                } else if (kind == AttnShape::fused) {
                    // No scratch buffer at all: the cached row is converted in registers as it is
                    // multiplied. Same accumulator structure as the engine shape, so the same
                    // answer to the last bit; what goes away is a store and a load per element, and
                    // the dependency that made each position wait on the one before it.
                    for (int t = lo; t < hi; ++t) {
                        const size_t at = base + static_cast<size_t>(t) * head_dim;
                        for (int g = 0; g < group; ++g) {
                            const int h = first_head + g;
                            scores[static_cast<size_t>(h) * positions + t] =
                                dot_f16_f32(keys.data() + at,
                                            queries.data() + static_cast<size_t>(h) * head_dim,
                                            head_dim) * scale;
                        }
                    }
                    for (int t = lo; t < hi; ++t) {
                        const size_t at = base + static_cast<size_t>(t) * head_dim;
                        for (int g = 0; g < group; ++g) {
                            const int h = first_head + g;
                            accumulate_scaled_f16(output + static_cast<size_t>(h) * head_dim,
                                                  values.data() + at,
                                                  scores[static_cast<size_t>(h) * positions + t],
                                                  head_dim);
                        }
                    }
                } else {
                    // The same arithmetic, with `block` positions converted before any of them is
                    // used, so the loads for several cache rows are outstanding at once.
                    for (int t = lo; t < hi; t += block) {
                        const int span = std::min(block, hi - t);
                        for (int b = 0; b < span; ++b) {
                            fp16_to_fp32_many(keys.data() + base + static_cast<size_t>(t + b) * head_dim,
                                              keys_buffer + static_cast<size_t>(b) * head_dim, head_dim);
                        }
                        for (int g = 0; g < group; ++g) {
                            const int h = first_head + g;
                            const float* q = queries.data() + static_cast<size_t>(h) * head_dim;
                            float* row = scores.data() + static_cast<size_t>(h) * positions + t;
                            for (int b = 0; b < span; ++b) {
                                row[b] = dot_f32(q, keys_buffer + static_cast<size_t>(b) * head_dim,
                                                 head_dim) * scale;
                            }
                        }
                        for (int b = 0; b < span; ++b) {
                            fp16_to_fp32_many(values.data() + base + static_cast<size_t>(t + b) * head_dim,
                                              values_buffer + static_cast<size_t>(b) * head_dim, head_dim);
                        }
                        for (int g = 0; g < group; ++g) {
                            const int h = first_head + g;
                            float* accumulator = output + static_cast<size_t>(h) * head_dim;
                            const float* row = scores.data() + static_cast<size_t>(h) * positions + t;
                            for (int b = 0; b < span; ++b) {
                                accumulate_scaled(accumulator,
                                                  values_buffer + static_cast<size_t>(b) * head_dim,
                                                  row[b], head_dim);
                            }
                        }
                    }
                }
            }
            checksums[static_cast<size_t>(worker) * kLine] += sum + output[0];
        };

        // A barrier per layer, as the engine has: attention is one parallel region per layer.
        const auto sweep = [&] {
            for (layer = 0; layer < layers; ++layer) {
                if (threads == 1) {
                    body(0, jobs, 0);
                } else {
                    pool.run(jobs, body);
                }
            }
        };
        sweep();  // untimed, so the measurement is not paying for a cold cache
        const auto started = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            sweep();
        }
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }

    double checksum = 0.0;
    for (double value : checksums) {
        checksum += value;
    }
    py::dict result;
    result["seconds"] = seconds;
    result["cache_bytes"] = cache_bytes;
    result["cache_bytes_read"] = cache_bytes * iters;
    result["gb_per_second"] = cache_bytes * iters / seconds / 1e9;
    result["jobs"] = jobs;
    result["threads"] = threads;
    result["checksum"] = checksum;
    return result;
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
    out["prefill_seconds"] = stats.prefill_seconds;
    out["prefill_model_seconds"] = stats.prefill_model_seconds;
    out["rounds_seconds"] = stats.rounds_seconds();
    out["propose_seconds"] = stats.propose_seconds;
    out["draft_forward_seconds"] = stats.draft_forward_seconds;
    out["verify_seconds"] = stats.verify_seconds;
    out["accept_seconds"] = stats.accept_seconds;
    out["draft_sampling_seconds"] = stats.draft_sampling_seconds();
    out["bookkeeping_seconds"] = stats.bookkeeping_seconds();
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

// ---------------------------------------------------------------------- the kernel benchmark
//
// The k-token kernel on its own, with no model around it. Measuring v(k) through the engine mixes
// several things: how efficiently the kernel issues its multiply-accumulates, how often it stalls on
// memory, and whatever the rest of the forward pass costs. This isolates the first two -- choosing
// `rows` so the weights fit a given cache level separates them -- and `threads` matters more than it
// looks: on one core the memory system is nowhere near saturated, so a kernel that wins there need
// not win on six, where bandwidth is the constraint and the *layout* of the weights decides how much
// of it is reachable. Comparing layouts at one thread would answer the wrong question.
enum class BenchLayout {
    interleaved,   // each block's scale beside its bytes, as the file stored it before version 4
    split_tensor,  // every row's scales, then every row's bytes
    split_row,     // per row: that row's scales, then that row's bytes
    split_flat,    // the split layout read one block per load: the kernel before pair-packing
};

py::dict bench_kernel(int rows, int n_in, int tokens, int iters, int threads,
                      const std::string& format, BenchLayout layout, const std::string& cores) {
    if (rows < 1 || n_in < 1 || tokens < 1 || iters < 1 || threads < 1) {
        throw std::invalid_argument("rows, n_in, tokens, iters and threads must all be positive");
    }
    if (n_in % QK != 0) {
        throw std::invalid_argument("n_in must be a multiple of " + std::to_string(QK));
    }
    const bool four_bit = format == "q4";
    if (!four_bit && format != "q8") {
        throw std::invalid_argument("format must be \"q4\" or \"q8\"");
    }
    CoreSelection selection = CoreSelection::performance;
    if (!parse_core_selection(cores.c_str(), &selection)) {
        throw std::invalid_argument("unknown core selection " + cores);
    }

    const int nblocks = n_in / QK;
    const size_t width = static_cast<size_t>(nblocks);
    const size_t payload = four_bit ? QK / 2 : QK;  // quantized bytes a block
    const int zero_point = four_bit ? 8 : 128;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<float> scratch(static_cast<size_t>(n_in));
    const auto fill = [&] {
        for (float& value : scratch) {
            value = uniform(rng);
        }
    };

    // Activations, in whichever shape this layout's kernel reads.
    std::vector<BlockA8> act_blocks;
    std::vector<float> x_scales;
    std::vector<int8_t> x_qs;
    std::vector<int32_t> x_offsets;
    if (layout == BenchLayout::interleaved) {
        act_blocks.resize(width * tokens);
        for (int t = 0; t < tokens; ++t) {
            fill();
            quantize_a8(scratch.data(), n_in, act_blocks.data() + static_cast<size_t>(t) * width);
        }
    } else {
        x_scales.resize(width * tokens);
        x_qs.resize(width * tokens * QK);
        x_offsets.resize(width * tokens);
        for (int t = 0; t < tokens; ++t) {
            fill();
            const size_t offset = static_cast<size_t>(t) * width;
            const bool pair = four_bit && layout != BenchLayout::split_flat;
            quantize_a8_soa(scratch.data(), n_in, zero_point, x_scales.data() + offset,
                            x_qs.data() + offset * QK, x_offsets.data() + offset, pair);
        }
    }

    // Weights. One buffer whatever the layout, so no layout gets an allocation advantage; only the
    // arithmetic that finds a row inside it differs.
    const size_t row_stride = width * sizeof(uint16_t) + width * payload;
    std::vector<uint8_t> weights(static_cast<size_t>(rows) * row_stride);
    std::vector<uint8_t> staging(width * (four_bit ? sizeof(BlockQ4) : sizeof(BlockQ8)));
    for (int r = 0; r < rows; ++r) {
        fill();
        if (four_bit) {
            quantize_q4(scratch.data(), n_in, reinterpret_cast<BlockQ4*>(staging.data()));
        } else {
            quantize_q8(scratch.data(), n_in, reinterpret_cast<BlockQ8*>(staging.data()));
        }
        const size_t at = static_cast<size_t>(r) * row_stride;
        switch (layout) {
            case BenchLayout::interleaved:
                std::memcpy(weights.data() + at, staging.data(), staging.size());
                break;
            case BenchLayout::split_row:
                if (four_bit) {
                    repack_q4_soa(reinterpret_cast<const BlockQ4*>(staging.data()), nblocks,
                                  reinterpret_cast<uint16_t*>(weights.data() + at),
                                  weights.data() + at + width * sizeof(uint16_t));
                } else {
                    repack_q8_soa(reinterpret_cast<const BlockQ8*>(staging.data()), nblocks,
                                  reinterpret_cast<uint16_t*>(weights.data() + at),
                                  reinterpret_cast<int8_t*>(weights.data() + at +
                                                            width * sizeof(uint16_t)));
                }
                break;
            case BenchLayout::split_flat:
            case BenchLayout::split_tensor: {
                // All the scales first, then all the bytes, across the whole buffer.
                uint8_t* scales_at = weights.data() + static_cast<size_t>(r) * width * sizeof(uint16_t);
                uint8_t* qs_at = weights.data() + static_cast<size_t>(rows) * width * sizeof(uint16_t) +
                                 static_cast<size_t>(r) * width * payload;
                if (four_bit) {
                    repack_q4_soa(reinterpret_cast<const BlockQ4*>(staging.data()), nblocks,
                                  reinterpret_cast<uint16_t*>(scales_at), qs_at);
                } else {
                    repack_q8_soa(reinterpret_cast<const BlockQ8*>(staging.data()), nblocks,
                                  reinterpret_cast<uint16_t*>(scales_at),
                                  reinterpret_cast<int8_t*>(qs_at));
                }
                break;
            }
        }
    }

    // Each worker's output and checksum get a cache line to themselves. Without the padding the six
    // workers write their results into one line and every row invalidates it in the other five
    // cores, which costs more than the kernel and costs it unevenly across k -- it reads as though
    // the layouts differ by eightfold at k=2 and not at all at k=4.
    constexpr size_t kLineFloats = 16;  // 64 bytes
    const size_t out_stride = ((static_cast<size_t>(tokens) + kLineFloats - 1) / kLineFloats) * kLineFloats;
    std::vector<float> out(static_cast<size_t>(threads) * out_stride);
    std::vector<double> checksums(static_cast<size_t>(threads) * kLineFloats, 0.0);
    const uint8_t* w_base = weights.data();
    const size_t scales_region = static_cast<size_t>(rows) * width * sizeof(uint16_t);

    const auto body = [&](int begin, int end, int worker) {
        float* results = out.data() + static_cast<size_t>(worker) * out_stride;
        double sum = 0.0;
        for (int r = begin; r < end; ++r) {
            const size_t at = static_cast<size_t>(r) * row_stride;
            switch (layout) {
                case BenchLayout::interleaved:
                    if (four_bit) {
                        dot_q4_a8_multi(reinterpret_cast<const BlockQ4*>(w_base + at),
                                        act_blocks.data(), nblocks, tokens, results);
                    } else {
                        dot_q8_a8_multi(reinterpret_cast<const BlockQ8*>(w_base + at),
                                        act_blocks.data(), nblocks, tokens, results);
                    }
                    break;
                case BenchLayout::split_row: {
                    const auto* scales = reinterpret_cast<const uint16_t*>(w_base + at);
                    const uint8_t* qs = w_base + at + width * sizeof(uint16_t);
                    if (four_bit) {
                        dot_q4_soa_multi(scales, qs, x_scales.data(), x_qs.data(), x_offsets.data(),
                                         nblocks, tokens, results);
                    } else {
                        dot_q8_soa_multi(scales, reinterpret_cast<const int8_t*>(qs),
                                         x_scales.data(), x_qs.data(), x_offsets.data(), nblocks,
                                         tokens, results);
                    }
                    break;
                }
                case BenchLayout::split_flat: {
                    const auto* scales = reinterpret_cast<const uint16_t*>(
                        w_base + static_cast<size_t>(r) * width * sizeof(uint16_t));
                    const uint8_t* qs =
                        w_base + scales_region + static_cast<size_t>(r) * width * payload;
                    if (!four_bit) {
                        throw std::invalid_argument("split_flat is a q4-only comparison path");
                    }
                    dot_q4_soa_flat_multi(scales, qs, x_scales.data(), x_qs.data(), x_offsets.data(),
                                          nblocks, tokens, results);
                    break;
                }
                case BenchLayout::split_tensor: {
                    const auto* scales = reinterpret_cast<const uint16_t*>(
                        w_base + static_cast<size_t>(r) * width * sizeof(uint16_t));
                    const uint8_t* qs =
                        w_base + scales_region + static_cast<size_t>(r) * width * payload;
                    if (four_bit) {
                        dot_q4_soa_multi(scales, qs, x_scales.data(), x_qs.data(), x_offsets.data(),
                                         nblocks, tokens, results);
                    } else {
                        dot_q8_soa_multi(scales, reinterpret_cast<const int8_t*>(qs),
                                         x_scales.data(), x_qs.data(), x_offsets.data(), nblocks,
                                         tokens, results);
                    }
                    break;
                }
            }
            sum += results[0];  // so nothing in the timed loop can be dropped as dead
        }
        checksums[static_cast<size_t>(worker) * kLineFloats] += sum;
    };

    double seconds = 0.0;
    {
        py::gil_scoped_release unlocked;
        ThreadPool pool(threads, selection);
        const auto sweep = [&] {
            if (threads == 1) {
                body(0, rows, 0);
            } else {
                pool.run(rows, body);
            }
        };
        sweep();  // untimed, so the measurement is not paying for cold caches
        const auto started = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            sweep();
        }
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }

    double checksum = 0.0;
    for (double value : checksums) {
        checksum += value;
    }
    const double weight_bytes = static_cast<double>(rows) * row_stride;
    py::dict result;
    result["seconds"] = seconds;
    result["macs"] = static_cast<double>(rows) * n_in * tokens * iters;
    result["weight_bytes"] = weight_bytes;
    result["weight_bytes_read"] = weight_bytes * iters;
    result["activation_bytes"] =
        layout == BenchLayout::interleaved
            ? static_cast<double>(act_blocks.size() * sizeof(BlockA8))
            : static_cast<double>(x_qs.size() + 4 * x_offsets.size() + 4 * x_scales.size());
    result["checksum"] = checksum;
    result["threads"] = threads;
    return result;
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
        "argmax",
        [](FloatArray values) { return argmax(values.data(), static_cast<int>(values.size())); },
        py::arg("values"), "First index of the largest value: the engine's vectorized scan.");

    m.def(
        "argmax_scalar",
        [](FloatArray values) {
            return argmax_scalar(values.data(), static_cast<int>(values.size()));
        },
        py::arg("values"), "The scalar scan it replaced, kept so a test can pin them together.");

    // Both variants timed by one piece of code in one process, which is the only comparison this
    // machine supports: an unchanged baseline drifts by a third between runs.
    m.def(
        "bench_argmax",
        [](int n, int iters, const std::string& variant) {
            if (n <= 0 || iters <= 0) {
                throw std::invalid_argument("n and iters must be positive");
            }
            // A deterministic spread whose largest value sits near the end, so a scan that gave up
            // early would be caught by the index this returns rather than merely look fast.
            std::vector<float> values(static_cast<size_t>(n));
            uint32_t state = 12345u;
            for (int i = 0; i < n; ++i) {
                state = state * 1664525u + 1013904223u;
                values[static_cast<size_t>(i)] = static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
            }
            const int planted = n - 1 - n / 8;
            values[static_cast<size_t>(planted)] = 2.0f;
            if (variant != "scalar" && variant != "vector") {
                throw std::invalid_argument("variant must be \"vector\" or \"scalar\"");
            }
            int (*scan)(const float*, int) = variant == "scalar" ? &argmax_scalar : &argmax;
            long long found = 0;
            const auto started = std::chrono::steady_clock::now();
            for (int it = 0; it < iters; ++it) {
                found += scan(values.data(), n);  // used, so the call cannot be hoisted away
            }
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            py::dict out;
            out["seconds"] = seconds;
            out["elements_per_second"] = static_cast<double>(n) * iters / seconds;
            out["best"] = static_cast<int>(found / iters);
            out["planted"] = planted;
            return out;
        },
        py::arg("n"), py::arg("iters") = 1, py::arg("variant") = "vector",
        "Time one of the two argmax scans over a row of n floats.");

    m.def(
        "accept_or_resample_rows_read",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> p,
           py::array_t<float, py::array::c_style | py::array::forcecast> q,
           const std::vector<int32_t>& guesses, bool greedy, uint64_t seed) {
            const int gamma = static_cast<int>(guesses.size());
            if (p.ndim() != 2 || q.ndim() != 2 || p.shape(0) != gamma + 1 || q.shape(0) != gamma) {
                throw std::invalid_argument("expected p [gamma+1, V] and q [gamma, V]");
            }
            const int vocab = static_cast<int>(p.shape(1));
            Rng rng(seed);
            std::vector<float> scratch;
            std::vector<int> rows;
            const PrepareRow record = [&](int row) { rows.push_back(row); };
            const Verdict verdict =
                accept_or_resample(p.data(), vocab, q.data(), vocab, guesses.data(), gamma, vocab,
                                   greedy, rng, scratch, &record);
            return py::make_tuple(verdict.accepted, verdict.next_token, rows);
        },
        py::arg("p"), py::arg("q"), py::arg("guesses"), py::arg("greedy") = false,
        py::arg("seed") = uint64_t{0},
        "The acceptance rule, reporting which rows of p it asked for and in what order.");

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

    // Activations in the split layout, returned as the three arrays the kernels read, so a test can
    // check they hold the same bytes and scales quantize_a8 produces.
    m.def(
        "quantize_a8_soa",
        [](const FloatArray& x, int zero_point, bool paired) {
            if (x.ndim() != 1 || x.size() == 0 || x.size() % QK != 0) {
                throw std::invalid_argument("x must be a non-empty 1-D array of whole blocks");
            }
            const int n = static_cast<int>(x.size());
            const int nblocks = n / QK;
            py::array_t<float> scales(nblocks);
            py::array_t<int8_t> qs(n);
            py::array_t<int32_t> offsets(nblocks);
            quantize_a8_soa(x.data(), n, zero_point, scales.mutable_data(), qs.mutable_data(),
                            offsets.mutable_data(), paired);
            py::dict out;
            out["scales"] = scales;
            out["qs"] = qs;
            out["offsets"] = offsets;
            return out;
        },
        py::arg("x"), py::arg("zero_point") = 8, py::arg("paired") = true);

    // The same, done in `chunks` separate block ranges, which is how the engine spreads it over its
    // workers. Splitting must not change a byte -- every block's scale comes from its own 32 values --
    // and this is what a test checks, because the alternative is the engine quietly disagreeing with
    // the PyTorch twin only at whatever block boundary the thread count happens to produce.
    m.def(
        "quantize_a8_soa_chunked",
        [](const FloatArray& x, int zero_point, bool paired, int chunks) {
            if (x.ndim() != 1 || x.size() == 0 || x.size() % QK != 0) {
                throw std::invalid_argument("x must be a non-empty 1-D array of whole blocks");
            }
            if (chunks < 1) {
                throw std::invalid_argument("chunks must be positive");
            }
            const int n = static_cast<int>(x.size());
            const int nblocks = n / QK;
            py::array_t<float> scales(nblocks);
            py::array_t<int8_t> qs(n);
            py::array_t<int32_t> offsets(nblocks);
            std::memset(qs.mutable_data(), 0, static_cast<size_t>(n));
            for (int c = 0; c < chunks; ++c) {
                const int begin = static_cast<int>(static_cast<int64_t>(c) * nblocks / chunks);
                const int end = static_cast<int>(static_cast<int64_t>(c + 1) * nblocks / chunks);
                quantize_a8_soa_blocks(x.data(), nblocks, begin, end, zero_point,
                                      scales.mutable_data(), qs.mutable_data(),
                                      offsets.mutable_data(), paired);
            }
            py::dict out;
            out["scales"] = scales;
            out["qs"] = qs;
            out["offsets"] = offsets;
            return out;
        },
        py::arg("x"), py::arg("zero_point") = 8, py::arg("paired") = true, py::arg("chunks") = 1);

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

    // The split-layout kernels, taking the same interleaved blobs the other dot bindings take and
    // repacking them, so a test can check them against the scalar reference without knowing the
    // layout. The zero point differs by format: 8 for q4's nibbles, 128 for q8's bytes.
    m.def(
        "dot_q4_a8_soa",
        [](const py::bytes& weights, const py::bytes& activations) {
            std::string w = weights;
            std::string x = activations;
            if (w.size() % sizeof(BlockQ4) != 0 || w.empty()) {
                throw std::invalid_argument("weights must be a whole number of Q4 blocks");
            }
            const int nblocks = static_cast<int>(w.size() / sizeof(BlockQ4));
            const size_t width = static_cast<size_t>(nblocks);
            if (x.size() % (width * sizeof(BlockA8)) != 0 || x.empty()) {
                throw std::invalid_argument("activations must be k whole vectors of the same width");
            }
            const int tokens = static_cast<int>(x.size() / (width * sizeof(BlockA8)));

            std::vector<uint16_t> w_scales(width);
            std::vector<uint8_t> w_qs(width * (QK / 2));
            repack_q4_soa(reinterpret_cast<const BlockQ4*>(w.data()), nblocks, w_scales.data(),
                          w_qs.data());

            std::vector<float> x_scales(width * tokens);
            std::vector<int8_t> x_qs(width * tokens * QK);
            std::vector<int32_t> x_offsets(width * tokens);
            const auto* blocks = reinterpret_cast<const BlockA8*>(x.data());
            for (int t = 0; t < tokens; ++t) {
                const size_t offset = static_cast<size_t>(t) * width;
                repack_a8_soa(blocks + offset, nblocks, 8, x_scales.data() + offset,
                              x_qs.data() + offset * QK, x_offsets.data() + offset, true);
            }

            py::array_t<float> out(tokens);
            dot_q4_soa_multi(w_scales.data(), w_qs.data(), x_scales.data(), x_qs.data(),
                             x_offsets.data(), nblocks, tokens, out.mutable_data());
            return out;
        },
        py::arg("weights"), py::arg("activations"));

    m.def(
        "dot_q8_a8_soa",
        [](const py::bytes& weights, const py::bytes& activations) {
            std::string w = weights;
            std::string x = activations;
            if (w.size() % sizeof(BlockQ8) != 0 || w.empty()) {
                throw std::invalid_argument("weights must be a whole number of Q8 blocks");
            }
            const int nblocks = static_cast<int>(w.size() / sizeof(BlockQ8));
            const size_t width = static_cast<size_t>(nblocks);
            if (x.size() % (width * sizeof(BlockA8)) != 0 || x.empty()) {
                throw std::invalid_argument("activations must be k whole vectors of the same width");
            }
            const int tokens = static_cast<int>(x.size() / (width * sizeof(BlockA8)));

            std::vector<uint16_t> w_scales(width);
            std::vector<int8_t> w_qs(width * QK);
            repack_q8_soa(reinterpret_cast<const BlockQ8*>(w.data()), nblocks, w_scales.data(),
                          w_qs.data());

            std::vector<float> x_scales(width * tokens);
            std::vector<int8_t> x_qs(width * tokens * QK);
            std::vector<int32_t> x_offsets(width * tokens);
            const auto* blocks = reinterpret_cast<const BlockA8*>(x.data());
            for (int t = 0; t < tokens; ++t) {
                const size_t offset = static_cast<size_t>(t) * width;
                repack_a8_soa(blocks + offset, nblocks, 128, x_scales.data() + offset,
                              x_qs.data() + offset * QK, x_offsets.data() + offset, false);
            }

            py::array_t<float> out(tokens);
            dot_q8_soa_multi(w_scales.data(), w_qs.data(), x_scales.data(), x_qs.data(),
                             x_offsets.data(), nblocks, tokens, out.mutable_data());
            return out;
        },
        py::arg("weights"), py::arg("activations"));

    // Both benchmarks go through one implementation, so the only difference between them is the
    // layout being measured. `threads` defaults to one for a pure throughput figure; pass six to ask
    // the question the engine cares about, where the memory system is the constraint.
    // The float helpers attention and sampling are built from, exposed so their claims can be
    // checked rather than believed: that the grouped kernels are bit-for-bit what the per-head ones
    // were, and that the vectorized exponential is accurate enough for both oracles.
    m.def(
        "vector_exp",
        [](FloatArray values) {
            py::array_t<float> out(values.size());
            const int n = static_cast<int>(values.size());
            int i = 0;
            for (; i + 8 <= n; i += 8) {
                _mm256_storeu_ps(out.mutable_data() + i,
                                 exp256_ps(_mm256_loadu_ps(values.data() + i)));
            }
            for (; i < n; ++i) {  // the tail, one lane of a vector call
                alignas(32) float lane[8] = {values.data()[i]};
                _mm256_storeu_ps(lane, exp256_ps(_mm256_loadu_ps(lane)));
                out.mutable_data()[i] = lane[0];
            }
            return out;
        },
        py::arg("values"), "The engine's vectorized e^x, for checking against a known-good one.");

    m.def(
        "vector_softmax",
        [](FloatArray values) {
            py::array_t<float> out(values.size());
            std::memcpy(out.mutable_data(), values.data(),
                        static_cast<size_t>(values.size()) * sizeof(float));
            softmax_in_place(out.mutable_data(), static_cast<uint32_t>(out.size()));
            return out;
        },
        py::arg("values"), "The softmax attention and the sampling warps share.");

    m.def(
        "check_group_kernels",
        [](int n, int group, uint64_t seed) {
            if (n < 1 || group < 1 || group > 8) {
                throw std::invalid_argument("n must be positive and group within 1..8");
            }
            std::mt19937_64 rng(seed);
            std::uniform_real_distribution<float> uniform(-2.0f, 2.0f);
            std::vector<uint16_t> cached(static_cast<size_t>(n));
            std::vector<float> converted(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                cached[i] = fp32_to_fp16(uniform(rng));
                converted[i] = fp16_to_fp32(cached[i]);
            }
            std::vector<float> queries(static_cast<size_t>(group) * n);
            for (float& value : queries) {
                value = uniform(rng);
            }
            std::vector<float> weights(static_cast<size_t>(group));
            for (float& value : weights) {
                value = uniform(rng);
            }

            // The grouped kernels, against the per-head ones they replaced: a conversion into a
            // buffer followed by dot_f32 and accumulate_scaled.
            std::vector<float> grouped_dots(static_cast<size_t>(group));
            dot_f16_f32_group(cached.data(), queries.data(), static_cast<uint32_t>(n), group,
                              static_cast<uint32_t>(n), grouped_dots.data());
            std::vector<float> reference_dots(static_cast<size_t>(group));
            for (int g = 0; g < group; ++g) {
                reference_dots[static_cast<size_t>(g)] = dot_f32(
                    queries.data() + static_cast<size_t>(g) * n, converted.data(),
                    static_cast<uint32_t>(n));
            }

            std::vector<float> grouped_sum(static_cast<size_t>(group) * n, 0.0f);
            accumulate_scaled_f16_group(grouped_sum.data(), static_cast<uint32_t>(n), cached.data(),
                                        weights.data(), group, static_cast<uint32_t>(n));
            std::vector<float> reference_sum(static_cast<size_t>(group) * n, 0.0f);
            for (int g = 0; g < group; ++g) {
                accumulate_scaled(reference_sum.data() + static_cast<size_t>(g) * n,
                                  converted.data(), weights[static_cast<size_t>(g)],
                                  static_cast<uint32_t>(n));
            }

            py::dict out;
            out["grouped_dots"] = grouped_dots;
            out["reference_dots"] = reference_dots;
            out["grouped_sum"] = grouped_sum;
            out["reference_sum"] = reference_sum;
            return out;
        },
        py::arg("n"), py::arg("group") = 4, py::arg("seed") = uint64_t{1},
        "Attention's grouped kernels beside the per-head ones they replaced.");

    m.def(
        "bench_attention",
        [](int positions, int layers, int kv_heads, int group, int head_dim, int jobs_per_head,
           int block, int iters, int threads, const std::string& cores, const std::string& shape) {
            return bench_attention(positions, layers, kv_heads, group, head_dim, jobs_per_head,
                                   block, iters, threads, cores, shape);
        },
        py::arg("positions") = 1024, py::arg("layers") = 36, py::arg("kv_heads") = 8,
        py::arg("group") = 4, py::arg("head_dim") = 128, py::arg("jobs_per_head") = 1,
        py::arg("block") = 8, py::arg("iters") = 1, py::arg("threads") = 6,
        py::arg("cores") = "performance", py::arg("shape") = "engine",
        "The KV cache scan on its own: what attention's bandwidth goes on.");

    m.def(
        "bench_dot",
        [](int rows, int n_in, int tokens, int iters, const std::string& format, int threads,
           const std::string& cores) {
            return bench_kernel(rows, n_in, tokens, iters, threads, format,
                                BenchLayout::interleaved, cores);
        },
        py::arg("rows"), py::arg("n_in"), py::arg("tokens"), py::arg("iters") = 1,
        py::arg("format") = "q4", py::arg("threads") = 1, py::arg("cores") = "performance");

    m.def(
        "bench_dot_soa",
        [](int rows, int n_in, int tokens, int iters, const std::string& format, int threads,
           const std::string& cores, bool row_major, const std::string& variant) {
            BenchLayout layout = row_major ? BenchLayout::split_row : BenchLayout::split_tensor;
            if (variant == "flat") {
                layout = BenchLayout::split_flat;
            } else if (variant != "paired") {
                throw std::invalid_argument("variant must be \"paired\" or \"flat\"");
            }
            return bench_kernel(rows, n_in, tokens, iters, threads, format, layout, cores);
        },
        py::arg("rows"), py::arg("n_in"), py::arg("tokens"), py::arg("iters") = 1,
        py::arg("format") = "q4", py::arg("threads") = 1, py::arg("cores") = "performance",
        py::arg("row_major") = true, py::arg("variant") = "paired");
}
