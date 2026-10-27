#pragma once

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer {

// The all-reduce of a tensor-parallel group. Every rank of a DeviceContext holds a replica of the
// BF16 residual and an FP32 partial of one row-split projection; each rank adds the sum of all
// partials, taken in rank order, to its replica and rounds once:
//
//   x_r <- bf16(float(x_r) + ((p_0 + p_1) + ... + p_{N-1}))
//
// Every rank evaluates the same expression on the same values, so the replicas stay identical, and
// the one rounding is the one a single device's residual-adding projection makes. A projection only
// rank 0 runs is an exchange with one contributor: every replica adds rank 0's output.
//
// A rank writes its partial into its slot (`partial`), publishes it (`publish`) and, once every
// rank has published, reduces (`reduce`). Every rank publishes in every exchange, one outside the
// contributors with zero bytes. Two slots alternate between exchanges. A rank writes a
// slot again two exchanges later, after its reduce of the exchange in between, which waits for
// every other rank's publish of that exchange, which each rank issues after its own reduce of the
// earlier one: so no rank overwrites a slot another is still reading. Consecutive passes are
// separate in the same way as long as a pass ends in a synchronize, as every round does.
//
// Records and waits pair within one exchange, so the protocol is legal eagerly and inside one
// capture that spans every rank's stream.
//
// A pass may cut its columns into pieces (up to kPieces) and exchange each piece on the rank's
// side stream while its compute stream runs the next piece: a piece writes, publishes and reduces
// its own region of the slot (`offset`), the side stream waits `piece_ready` before publishing and
// records `piece_done` after reducing, and the compute stream waits `piece_done` before the piece's
// next use of its residual. Publishes and reduces of one piece are issued for every rank before the
// next piece's, so each reduce waits for the publish of its own piece.
//
// Transports, chosen once from the topology:
//   Peer        every pair of ranks has peer access or shares a device: a reduce reads the other
//               ranks' device slots directly.
//   StagedHost  a publish copies the slot into a pinned, mapped host slot, and a reduce reads the
//               other ranks' host slots across the bus. Works between any two cards.
enum class TensorParallelTransport { Peer, StagedHost };

struct TensorParallelExchangeOptions {
    // Largest partial one slot holds.
    std::size_t slot_bytes = 0;
    // Use the staged protocol even where peer access exists (tests).
    bool force_staged = false;
};

class TensorParallelExchange {
public:
    static constexpr std::size_t kSlots  = 2;
    static constexpr std::size_t kPieces = 4;

    TensorParallelExchange(const DeviceContext& context, TensorParallelExchangeOptions options);
    ~TensorParallelExchange();

    TensorParallelExchange(const TensorParallelExchange&)            = delete;
    TensorParallelExchange& operator=(const TensorParallelExchange&) = delete;
    TensorParallelExchange(TensorParallelExchange&&)                 = delete;
    TensorParallelExchange& operator=(TensorParallelExchange&&)      = delete;

    [[nodiscard]] std::size_t ranks() const noexcept { return ranks_.size(); }
    [[nodiscard]] TensorParallelTransport transport() const noexcept { return transport_; }
    [[nodiscard]] std::size_t slot_bytes() const noexcept { return slot_bytes_; }

    // Where `rank` writes its FP32 partial for an exchange in `slot`: device memory of that rank.
    [[nodiscard]] float* partial(std::size_t rank, std::size_t slot) const;
    // Makes `rank`'s partial of `bytes` in `slot`, from `offset` values in, readable by the other
    // ranks.
    void publish(std::size_t rank, std::size_t slot, std::size_t bytes, cudaStream_t stream,
                 std::size_t offset = 0);
    // Adds the partials ranks [0, contributors) published in `slot`, from `offset` values in, to
    // `rank`'s BF16 residual of `count` values.
    void reduce(std::size_t rank, std::size_t slot, __nv_bfloat16* residual, std::int64_t count,
                std::size_t contributors, cudaStream_t stream, std::size_t offset = 0);
    // A piece's exchange stream and fences on `rank` (see above).
    [[nodiscard]] cudaStream_t side_stream(std::size_t rank) const;
    [[nodiscard]] cudaEvent_t piece_ready(std::size_t rank, std::size_t piece) const;
    [[nodiscard]] cudaEvent_t piece_done(std::size_t rank, std::size_t piece) const;

private:
    struct Rank {
        int device = 0;
        std::array<float*, kSlots> device_slot{};
        std::array<float*, kSlots> host_slot{};
        std::array<cudaEvent_t, kSlots> published{};
        cudaStream_t side = nullptr;
        std::array<cudaEvent_t, kPieces> ready{};
        std::array<cudaEvent_t, kPieces> done{};
    };

    void release() noexcept;
    void check_slot(std::size_t rank, std::size_t slot, std::size_t bytes,
                    std::size_t offset_bytes = 0) const;
    void check_piece(std::size_t rank, std::size_t piece) const;

    std::vector<Rank> ranks_;
    TensorParallelTransport transport_ = TensorParallelTransport::StagedHost;
    std::size_t slot_bytes_            = 0;
};

// Each rank's speed for dividing work by: the device-to-device copy bandwidth of its card in bytes
// per second, shared evenly by the ranks one card holds (`--devices 0,0`). A decode step reads every
// weight once, so a rank's share of the weights takes time in proportion to this. Measured with a
// short copy on each card.
[[nodiscard]] std::vector<double> rank_copy_bandwidth(const DeviceContext& context);

// Each rank's free device memory now, shared evenly by the ranks one card holds.
[[nodiscard]] std::vector<std::uint64_t> rank_free_bytes(const DeviceContext& context);

namespace detail {

// One rank's reduction: residual[i] <- bf16(float(residual[i]) + ((part[0][i] + part[1][i]) + ...)).
void tensor_parallel_reduce(__nv_bfloat16* residual, const float* const* parts, std::size_t count,
                            std::int64_t values, cudaStream_t stream);

} // namespace detail

} // namespace ninfer
