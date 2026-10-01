#include "specdraft/threadpool.hpp"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

#include "specdraft/cpu.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX  // keep windows.h from defining min/max macros
#include <windows.h>
#endif

namespace specdraft {
namespace {

// Spin, then yield, then sleep. Keeps hand-off latency low inside a token while letting an
// idle model stop burning cores.
class Backoff {
public:
    void wait() {
        ++spins_;
        if (spins_ < kPauseSpins) {
            _mm_pause();
        } else if (spins_ < kYieldSpins) {
            std::this_thread::yield();
        } else {
            // Only reached when the model has been idle between calls for a long time.
            // Never sleep on the job-to-job path: Windows' default timer resolution is
            // about 15 ms, so a request to sleep for microseconds costs milliseconds, which
            // is far longer than the work a job contains.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    void reset() { spins_ = 0; }

private:
    static constexpr uint64_t kPauseSpins = 8192;
    static constexpr uint64_t kYieldSpins = 1 << 20;
    uint64_t spins_ = 0;
};

void pin_to_core(uint32_t logical_index) {
#if defined(_WIN32)
    if (logical_index < 64) {
        SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << logical_index);
    }
#else
    (void)logical_index;
#endif
}

}  // namespace

const char* core_selection_name(CoreSelection selection) {
    switch (selection) {
        case CoreSelection::any:
            return "any";
        case CoreSelection::performance:
            return "performance";
        case CoreSelection::physical:
            return "physical";
        case CoreSelection::logical:
            return "logical";
    }
    return "?";
}

bool parse_core_selection(const char* name, CoreSelection* out) {
    const std::string wanted = name == nullptr ? "" : name;
    for (CoreSelection candidate : {CoreSelection::any, CoreSelection::performance,
                                   CoreSelection::physical, CoreSelection::logical}) {
        if (wanted == core_selection_name(candidate)) {
            *out = candidate;
            return true;
        }
    }
    return false;
}

std::vector<uint32_t> cores_for(CoreSelection selection) {
    const std::vector<CoreInfo>& cores = core_topology();
    const uint32_t fastest = fastest_efficiency_class();
    std::vector<uint32_t> chosen;
    for (const CoreInfo& core : cores) {
        const bool wanted = selection == CoreSelection::logical ||
                            (selection == CoreSelection::physical && core.primary) ||
                            (selection == CoreSelection::performance && core.primary &&
                             core.efficiency_class == fastest) ||
                            selection == CoreSelection::any;
        if (wanted) {
            chosen.push_back(core.logical_index);
        }
    }
    if (chosen.empty()) {
        chosen.push_back(0);
    }
    return chosen;
}

ThreadPool::ThreadPool(int threads, CoreSelection selection) : selection_(selection) {
    const std::vector<uint32_t> cores = cores_for(selection);
    int count = threads > 0 ? threads : static_cast<int>(cores.size());
    count = std::max(1, count);

    if (selection != CoreSelection::any) {
        pin_to_core(cores[0]);  // the calling thread is worker 0
    }
    for (int i = 1; i < count; ++i) {
        const uint32_t core = cores[static_cast<size_t>(i) % cores.size()];
        workers_.emplace_back([this, i, core, selection] {
            if (selection != CoreSelection::any) {
                pin_to_core(core);
            }
            worker_loop(i);
        });
    }
}

ThreadPool::~ThreadPool() {
    stopping_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::worker_loop(int index) {
    uint64_t seen = 0;
    Backoff backoff;
    while (true) {
        while (generation_.load(std::memory_order_acquire) == seen) {
            if (stopping_.load(std::memory_order_relaxed)) {
                return;
            }
            backoff.wait();
        }
        backoff.reset();
        seen = generation_.load(std::memory_order_acquire);
        if (stopping_.load(std::memory_order_acquire)) {
            return;
        }
        execute(index);
        remaining_.fetch_sub(1, std::memory_order_release);
    }
}

void ThreadPool::execute(int worker) {
    if (chunk_ > 0) {  // dynamic: keep taking chunks until none are left
        while (true) {
            const int begin = next_chunk_.fetch_add(chunk_, std::memory_order_relaxed);
            if (begin >= total_) {
                return;
            }
            job_(context_, begin, std::min(begin + chunk_, total_), worker);
        }
    }
    // static: one contiguous slice each, with the remainder spread over the first workers
    const int workers = size();
    const int base = total_ / workers;
    const int extra = total_ % workers;
    const int begin = worker * base + std::min(worker, extra);
    const int end = begin + base + (worker < extra ? 1 : 0);
    if (begin < end) {
        job_(context_, begin, end, worker);
    }
}

void ThreadPool::run_raw(int total, JobFn job, void* context) {
    if (total <= 0) {
        return;
    }
    if (workers_.empty() || total == 1) {
        job(context, 0, total, 0);
        return;
    }
    job_ = job;
    context_ = context;
    total_ = total;
    chunk_ = 0;
    remaining_.store(static_cast<int>(workers_.size()), std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);

    execute(0);  // the caller pulls its weight

    Backoff backoff;
    while (remaining_.load(std::memory_order_acquire) != 0) {
        backoff.wait();
    }
}

void ThreadPool::run_dynamic_raw(int total, int chunk, JobFn job, void* context) {
    if (total <= 0) {
        return;
    }
    if (workers_.empty() || total == 1) {
        job(context, 0, total, 0);
        return;
    }
    job_ = job;
    context_ = context;
    total_ = total;
    chunk_ = std::max(1, chunk);
    next_chunk_.store(0, std::memory_order_relaxed);
    remaining_.store(static_cast<int>(workers_.size()), std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);

    execute(0);

    Backoff backoff;
    while (remaining_.load(std::memory_order_acquire) != 0) {
        backoff.wait();
    }
}

double measure_read_bandwidth(size_t bytes, int threads, CoreSelection selection, int repeats) {
    bytes = std::max<size_t>(bytes, 1 << 20);
    const size_t count = bytes / sizeof(float);
    std::vector<float> buffer(count, 1.0f);

    ThreadPool pool(threads, selection);
    std::vector<double> partial(static_cast<size_t>(pool.size()), 0.0);

    // Sum rather than copy: inference reads weights and writes almost nothing, so a
    // read-only kernel is the honest ceiling. Four accumulators keep the floating-point
    // add latency from being the limit instead of memory.
    const auto sweep = [&](int begin, int end, int worker) {
        __m256 a = _mm256_setzero_ps(), b = _mm256_setzero_ps();
        __m256 c = _mm256_setzero_ps(), d = _mm256_setzero_ps();
        const float* data = buffer.data();
        int i = begin;
        for (; i + 32 <= end; i += 32) {
            a = _mm256_add_ps(a, _mm256_loadu_ps(data + i));
            b = _mm256_add_ps(b, _mm256_loadu_ps(data + i + 8));
            c = _mm256_add_ps(c, _mm256_loadu_ps(data + i + 16));
            d = _mm256_add_ps(d, _mm256_loadu_ps(data + i + 24));
        }
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d)));
        double sum = 0.0;
        for (float lane : lanes) {
            sum += lane;
        }
        for (; i < end; ++i) {
            sum += data[i];
        }
        partial[static_cast<size_t>(worker)] += sum;
    };

    pool.run(static_cast<int>(count), sweep);  // warm up: fault the pages in, settle the clocks

    double best = 0.0;
    for (int attempt = 0; attempt < std::max(1, repeats); ++attempt) {
        const auto started = std::chrono::steady_clock::now();
        pool.run(static_cast<int>(count), sweep);
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        best = std::max(best, static_cast<double>(count * sizeof(float)) / seconds / 1e9);
    }

    // Keep the sums observable so nothing above can be optimized away.
    if (std::accumulate(partial.begin(), partial.end(), 0.0) == 0.0) {
        throw std::runtime_error("bandwidth benchmark read nothing");
    }
    return best;
}

}  // namespace specdraft
