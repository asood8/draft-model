#include "specdraft/cpu.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace specdraft {
namespace {

void cpuid_ex(int regs[4], int leaf, int subleaf) {
#if defined(_MSC_VER)
    __cpuidex(regs, leaf, subleaf);
#else
    unsigned int a = 0, b = 0, c = 0, d = 0;
    __cpuid_count(leaf, subleaf, a, b, c, d);
    regs[0] = static_cast<int>(a);
    regs[1] = static_cast<int>(b);
    regs[2] = static_cast<int>(c);
    regs[3] = static_cast<int>(d);
#endif
}

CpuFeatures detect() {
    CpuFeatures f;
    int r[4] = {0, 0, 0, 0};

    cpuid_ex(r, 1, 0);
    f.fma = (r[2] >> 12) & 1;   // ECX bit 12
    f.f16c = (r[2] >> 29) & 1;  // ECX bit 29

    cpuid_ex(r, 7, 0);
    f.avx2 = (r[1] >> 5) & 1;      // EBX bit 5
    f.avx512f = (r[1] >> 16) & 1;  // EBX bit 16

    cpuid_ex(r, 7, 1);
    f.avx_vnni = (r[0] >> 4) & 1;  // EAX bit 4 (VEX-encoded AVX-VNNI)

    return f;
}

}  // namespace

const CpuFeatures& cpu_features() {
    static const CpuFeatures features = detect();
    return features;
}

const char* active_kernel_path() {
    const CpuFeatures& f = cpu_features();
    return (f.avx2 && f.avx_vnni) ? "vnni" : "scalar";
}

}  // namespace specdraft
