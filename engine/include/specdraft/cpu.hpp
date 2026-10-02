#pragma once

#include <cstdint>
#include <vector>

namespace specdraft {

// CPU features the kernels care about. AVX2 + FMA + F16C are the compile-time
// baseline; AVX-VNNI is detected at runtime and selects the fast integer kernels.
struct CpuFeatures {
    bool avx2 = false;
    bool fma = false;
    bool f16c = false;
    bool avx_vnni = false;
    bool avx512f = false;
};

const CpuFeatures& cpu_features();

// "vnni" or "scalar": which kernel path the dot products will actually take.
const char* active_kernel_path();

// Force the scalar kernels even on a CPU that has AVX-VNNI. The scalar path is the reference the
// SIMD kernels were written against, and on a machine with VNNI it would otherwise never run, so
// being able to select it is how it stays tested.
void set_force_scalar(bool force);
bool force_scalar();

// Whether the q4 kernel applies block scales eight at a time (the default) or one at a time. A
// measurement switch, not a tuning knob: adopting the grouped form came with a model-level slowdown
// that the isolated benchmarks contradicted, and the two have to be comparable inside one process to
// tell an arithmetic change from a layout change. Both compute the same dot product.
void set_scale_grouping(bool enabled);
bool scale_grouping();

// One logical processor. On a hybrid CPU like the i7-13620H there are two kinds of core,
// and which kind a thread lands on changes throughput a lot, so the engine needs to be
// able to see and choose.
struct CoreInfo {
    uint32_t logical_index = 0;
    uint32_t core_index = 0;  // physical core; two logical processors share one with SMT
    uint32_t efficiency_class = 0;  // higher is faster (Windows); 0 when unknown
    bool primary = false;  // the first logical processor of its physical core
};

// Every logical processor, in order. Empty if the OS would not say.
const std::vector<CoreInfo>& core_topology();

// The highest efficiency class present, i.e. which cores are the performance cores.
uint32_t fastest_efficiency_class();

}  // namespace specdraft
