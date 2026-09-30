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
#include "specdraft/quant.hpp"

namespace py = pybind11;
using namespace specdraft;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

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

}  // namespace

PYBIND11_MODULE(_engine, m) {
    m.doc() = "specdraft CPU engine";

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
