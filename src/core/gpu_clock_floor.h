#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer {

// Holds the graphics clock of CUDA devices at or above a floor for its lifetime (NVML locked
// clocks between the floor and the device's maximum) and gives them back to the driver when it
// ends. A layer pipeline leaves each stage waiting on its neighbours between short kernels, and
// the driver reads the waits as idle: on four RTX 3060 under Windows a Flash-Next decode ran its
// kernels at 210-480 MHz, and the longer the answer the lower its rate. Changing clocks needs
// administrator rights (root on Linux); a device whose clocks cannot be held is reported and left
// as it is. A process that is killed rather than stopped leaves the floor in place until the
// driver is told otherwise (`nvidia-smi -rgc`) or the machine restarts.
class GpuClockFloor {
public:
    GpuClockFloor() noexcept = default;
    GpuClockFloor(std::span<const int> devices, std::uint32_t floor_mhz);
    ~GpuClockFloor();

    GpuClockFloor(const GpuClockFloor&)            = delete;
    GpuClockFloor& operator=(const GpuClockFloor&) = delete;
    GpuClockFloor(GpuClockFloor&& other) noexcept;
    GpuClockFloor& operator=(GpuClockFloor&& other) noexcept;

    // One line per device: the range it is held in, or why it was left alone.
    [[nodiscard]] const std::vector<std::string>& report() const noexcept { return report_; }
    // Whether every device asked for is held.
    [[nodiscard]] bool complete() const noexcept { return complete_; }

private:
    struct Nvml;

    static std::shared_ptr<Nvml> load_nvml(std::string& problem);
    void release() noexcept;

    std::shared_ptr<Nvml> nvml_;
    std::vector<void*> held_;
    std::vector<std::string> report_;
    bool complete_ = false;
};

} // namespace ninfer
