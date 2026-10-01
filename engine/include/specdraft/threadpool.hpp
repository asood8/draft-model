#pragma once

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

// A pool whose threads stay alive between operations and wait at a spin barrier.
//
// A 0.6B model hits well over a hundred synchronization points per token, so handing work
// back to the OS scheduler each time would cost more than the work itself. Threads here
// spin briefly, then yield, then sleep, so a model that is idle between calls does not peg
// the CPU while a model mid-token never waits on a wake-up.
//
// Every job splits an index range across the workers, and the calling thread takes a share
// too. Each output row is always computed start to finish by one worker, so results do not
// depend on the thread count or the schedule: the engine stays bit-exact and reproducible.

namespace specdraft {

enum class CoreSelection {
    any,          // let the OS place threads
    performance,  // one thread per performance core (the default)
    physical,     // one thread per physical core, both kinds
    logical,      // one thread per logical processor, hyperthreads included
};

const char* core_selection_name(CoreSelection selection);
bool parse_core_selection(const char* name, CoreSelection* out);

// Logical processors a selection would use, in the order threads are assigned to them.
std::vector<uint32_t> cores_for(CoreSelection selection);

class ThreadPool {
public:
    // threads <= 0 means "as many as the selection offers".
    ThreadPool(int threads, CoreSelection selection);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return 1 + static_cast<int>(workers_.size()); }  // the caller counts
    CoreSelection selection() const { return selection_; }

    using JobFn = void (*)(void* context, int begin, int end, int worker);

    // Split [0, total) into one contiguous slice per worker.
    void run_raw(int total, JobFn job, void* context);
    // Hand out chunks from a shared counter, so faster cores take more work. This is what
    // a hybrid CPU needs when a static split would leave performance cores waiting.
    void run_dynamic_raw(int total, int chunk, JobFn job, void* context);

    // The body is passed by address, never wrapped in a std::function, so dispatching a job
    // allocates nothing: there are over a hundred of them per token.
    template <typename F>
    void run(int total, const F& body) {
        run_raw(total, &trampoline<F>, const_cast<void*>(static_cast<const void*>(&body)));
    }

    template <typename F>
    void run_dynamic(int total, int chunk, const F& body) {
        run_dynamic_raw(total, chunk, &trampoline<F>,
                        const_cast<void*>(static_cast<const void*>(&body)));
    }

private:
    template <typename F>
    static void trampoline(void* context, int begin, int end, int worker) {
        (*static_cast<const F*>(context))(begin, end, worker);
    }

    void worker_loop(int index);
    void execute(int worker);

    std::vector<std::thread> workers_;
    CoreSelection selection_;

    // Job state. Workers read it only after seeing a new generation number.
    JobFn job_ = nullptr;
    void* context_ = nullptr;
    int total_ = 0;
    int chunk_ = 0;  // 0 means the static split
    std::atomic<int> next_chunk_{0};
    std::atomic<uint64_t> generation_{0};
    std::atomic<int> remaining_{0};
    std::atomic<bool> stopping_{false};
};

// Measured read bandwidth in GB/s: the ceiling every tokens-per-second number is compared
// against, since decoding is dominated by streaming weights once per token.
double measure_read_bandwidth(size_t bytes, int threads, CoreSelection selection, int repeats = 3);

// Seconds to dispatch one empty job and collect every worker again. A token involves well
// over a hundred of these, so this number multiplied by that count is pure overhead.
double measure_dispatch_overhead(int jobs, int threads, CoreSelection selection);

}  // namespace specdraft
