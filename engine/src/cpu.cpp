#include "specdraft/cpu.hpp"

#include <algorithm>
#include <atomic>
#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX  // keep windows.h from defining min/max macros
#include <windows.h>
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

namespace {
std::atomic<bool> g_force_scalar{false};
}  // namespace

void set_force_scalar(bool force) {
    g_force_scalar.store(force, std::memory_order_relaxed);
}

bool force_scalar() {
    return g_force_scalar.load(std::memory_order_relaxed);
}


const char* active_kernel_path() {
    const CpuFeatures& f = cpu_features();
    return (f.avx2 && f.avx_vnni && !force_scalar()) ? "vnni" : "scalar";
}

namespace {

std::vector<CoreInfo> detect_cores() {
    std::vector<CoreInfo> cores;

#if defined(_WIN32)
    // GetSystemCpuSetInformation is the only place Windows reports EfficiencyClass, which
    // is what separates performance cores from efficiency cores.
    ULONG bytes = 0;
    GetSystemCpuSetInformation(nullptr, 0, &bytes, GetCurrentProcess(), 0);
    if (bytes > 0) {
        std::vector<uint8_t> buffer(bytes);
        if (GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()),
                                       bytes, &bytes, GetCurrentProcess(), 0)) {
            ULONG offset = 0;
            while (offset + sizeof(SYSTEM_CPU_SET_INFORMATION) <= bytes) {
                auto* entry = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data() + offset);
                if (entry->Size == 0) {
                    break;
                }
                if (entry->Type == CpuSetInformation) {
                    CoreInfo core;
                    core.logical_index = entry->CpuSet.LogicalProcessorIndex;
                    core.core_index = entry->CpuSet.CoreIndex;
                    core.efficiency_class = entry->CpuSet.EfficiencyClass;
                    cores.push_back(core);
                }
                offset += entry->Size;
            }
        }
    }
#endif

    if (cores.empty()) {  // no topology available: assume one thread per logical processor
        const unsigned count = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned i = 0; i < count; ++i) {
            CoreInfo core;
            core.logical_index = i;
            core.core_index = i;
            cores.push_back(core);
        }
    }

    std::sort(cores.begin(), cores.end(), [](const CoreInfo& a, const CoreInfo& b) {
        return a.logical_index < b.logical_index;
    });
    // Mark the first logical processor of each physical core, so callers can ask for one
    // thread per core instead of one per hyperthread.
    std::vector<uint32_t> seen;
    for (CoreInfo& core : cores) {
        if (std::find(seen.begin(), seen.end(), core.core_index) == seen.end()) {
            core.primary = true;
            seen.push_back(core.core_index);
        }
    }
    return cores;
}

}  // namespace

const std::vector<CoreInfo>& core_topology() {
    static const std::vector<CoreInfo> cores = detect_cores();
    return cores;
}

uint32_t fastest_efficiency_class() {
    uint32_t best = 0;
    for (const CoreInfo& core : core_topology()) {
        best = std::max(best, core.efficiency_class);
    }
    return best;
}

}  // namespace specdraft
