#pragma once

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

}  // namespace specdraft
