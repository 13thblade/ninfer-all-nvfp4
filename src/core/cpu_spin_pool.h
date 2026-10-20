#pragma once

// CPU workers for routed experts computed on the host while the GPU runs (every MoE family's
// host and disk experts). Workers spin between spin(true) and spin(false), so a decode call's
// products start without a wake-up, and sleep otherwise. run(count, job) executes job(i) for every
// i < count on the workers and the calling thread and returns when all are done, rethrowing the
// first failure. A claim is one CAS on a ticket holding the run's generation and next index, so a
// worker still leaving an earlier run can never claim an index of a later one.
//
// The expert products read stored weights at memory speed, which one thread per physical core
// saturates: an SMT sibling shares its core's loads. The workers are pinned one to a physical core
// of the process's CPUs (its affinity mask), the calling thread pins itself to the first of them
// when it first runs, and pool_cpu_count() bounds the default by the container's CPU quota.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer {

// Logical CPUs this process may keep busy: std::thread's count, bounded on Linux by the affinity
// mask and by a container's CPU quota (cgroup v2 cpu.max, v1 cpu.cfs_quota_us).
[[nodiscard]] unsigned usable_cpus();

// One logical CPU of each physical core this process may run on, in CPU order (Linux sysfs
// topology, Windows processor information; every allowed CPU where neither is readable).
[[nodiscard]] std::vector<int> physical_core_cpus();

// The default thread count of a CPU expert pool, the calling thread included: one per physical
// core, at most usable_cpus(), at least one.
[[nodiscard]] std::uint32_t pool_cpu_count();

// One turn of a spin wait: the processor's pause hint where it has one, a yield elsewhere.
void spin_hint() noexcept;

class CpuSpinPool {
public:
    // `threads` counts the calling thread: threads - 1 workers. `pin`: one physical core each.
    CpuSpinPool(std::uint32_t threads, bool pin);
    ~CpuSpinPool();
    CpuSpinPool(const CpuSpinPool&)            = delete;
    CpuSpinPool& operator=(const CpuSpinPool&) = delete;

    [[nodiscard]] std::uint32_t threads() const noexcept { return std::uint32_t(workers_.size()) + 1; }
    [[nodiscard]] bool pinned() const noexcept { return pinned_; }

    void spin(bool on);

    template <class Job>
    void run(std::size_t count, Job&& job) {
        if (count == 0) { return; }
        pin_caller();
        context_ = &job;
        call_    = [](void* context, std::size_t i) {
            (*static_cast<std::remove_reference_t<Job>*>(context))(i);
        };
        const std::uint64_t generation = (ticket_.load(std::memory_order_relaxed) >> 32) + 1;
        done_.store(0, std::memory_order_relaxed);
        // A worker claims only when the ticket's generation matches this one.
        size_.store(generation << 32 | count, std::memory_order_release);
        {
            std::lock_guard lock(mutex_);
            ticket_.store(generation << 32, std::memory_order_release);
        }
        wake_.notify_all();
        drain();
        while (done_.load(std::memory_order_acquire) != count) { spin_hint(); }
        if (failure_) {
            auto failure = std::exchange(failure_, nullptr);
            std::rethrow_exception(failure);
        }
    }

private:
    void drain();
    void work(int cpu);
    void pin_caller();

    std::vector<std::thread> workers_;
    int caller_cpu_     = -1;
    bool pinned_        = false;
    std::thread::id caller_;
    void (*call_)(void*, std::size_t) = nullptr;
    void* context_                    = nullptr;
    std::atomic<std::uint64_t> ticket_{0}; // generation << 32 | next index
    std::atomic<std::uint64_t> size_{0};   // generation << 32 | count
    std::atomic<std::size_t> done_{0};
    std::atomic<bool> spinning_{false}, stop_{false};
    std::mutex mutex_, failure_mutex_;
    std::condition_variable wake_;
    std::exception_ptr failure_;
};

} // namespace ninfer
