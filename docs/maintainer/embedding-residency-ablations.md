# Host token embedding: ablations to run

Temporary note for active work. Delete it once each item below is measured and decided; move a
lasting result into `docs/cli.md`, `docs/serving.md` or the Qwen3.5 model reference.

`--embedding-residency host` (Qwen3.5 family) binds `text/token_embedding` as
`artifact::Residency::Mapped`. The table is read into its own page-locked host block, registered
with `cudaHostRegisterMapped`, and its view addresses the device's mapping of that block
(`cudaHostGetDevicePointer`). Every consumer (prefill and its vision scatter, decode, verify, the
MTP stem, the DFlash and DFlash2 drafters) runs the unchanged `ops::embedding` gather, which now
reads the rows over PCIe in place. The baseline below is that design. Each item is a choice
that could change its performance, but not its output.

Shared measurement setup:
- Target: Qwen3.8-27B, `/extra/data/models/qwen3.8-27B/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`
  (IQ2_S embedding, 1640 B/row, 388 MiB).
- Card: RTX 5070 Ti 16 GB, `120a` build.
- Baseline: `--embedding-residency device` on the same build.
- Workloads:
  - decode tok/s at C1 and C8, with `--spec mtp` on and off;
  - TTFT for 8K and 32K prompts;
  - an nsys trace for kernel attribution: the embedding kernels are `dequantize_rows_kernel`
    (GGUF) and `embed_gather_*` (RowSplit/FP8).
- Report the distribution and the worst case, not only the mean.

## A1. Prefill: zero-copy gather vs. CPU gather plus copy engine

- **Hypothesis.** Prompt ids are known on the host, so the CPU could gather a chunk's raw rows
  into pinned staging. One `cudaMemcpyAsync` on the copy engine would then move them at full
  link bandwidth and overlap compute. Zero-copy instead uses SMs to issue small reads.
- **Scope.** Prefill only. Decode ids are sampled on the device inside CUDA graphs, so a CPU
  gather there would need a sync or a host node each round. Decode stays zero-copy.
- **Measure.** Embedding time per 2048-token chunk (nsys), and TTFT at 8K and 32K.
- **Decide.** Adopt only if the embedding's share of a prefill chunk is measurable, i.e. above
  run-to-run noise.

## A2. Read shape: in-place decode vs. stage-then-decode

- **Hypothesis.** The GGUF `dequantize_rows_kernel` reads byte fields scattered within each
  256-value superblock. Over PCIe each warp access becomes a small (32 B) read request, and
  threads re-read shared block headers.
- **Alternative (format-agnostic).**
  1. One CTA per token copies the row's raw bytes (each plane for RowSplit tables) from the
     mapped table into a planned device scratch with 16 B loads. This is one wide pass over PCIe.
  2. The existing kernel then decodes from device memory, with ids `0..T-1` over the scratch.
  - Same arithmetic, so the outputs are the same bits.
  - The scratch is `T × row bytes` (≈ 3.4 MB for a 2048-token IQ2_S chunk). It is planned in
    `execution/workspace.h` and startup planning, not allocated on the side.
- **Measure.** Embedding kernel time at T = 1, 8, 32, 2048 for both routes. A microbenchmark
  shows the effect at Op level; end-to-end decode and TTFT confirm it.
- **Decide.** Adopt where the Op gain also shows up end to end. Expect IQ2_S to gain the most;
  Q8 RowSplit already reads 4 B words.

## A3. Overlap

- **Prefill.** Issue chunk N+1's gather on a side stream into a double buffer while chunk N runs,
  vs. the serial gather at the start of each chunk.
- **Decode.** Programmatic dependent launch (PDL) of the first layer-0 kernel behind the
  embedding in the decode graph, so its prologue overlaps the PCIe round trip.
- **Measure.** Per-step decode latency at C1, where the gather is pure latency, and TTFT.

## A4. Host page size

- **Hypothesis.** Random row reads touch a new 4 KiB page each. With an IOMMU these miss the
  IOTLB. `madvise(MADV_HUGEPAGE)` on the mapped block before it is faulted in would cut those
  misses.
- **Measure.** Embedding kernel time at C8 decode and at 2048-token prefill, with and without
  huge pages. Record whether the IOMMU is enabled on the host.

## A5. Hot-row device cache

- **Hypothesis.** Natural text concentrates on a few thousand tokens. Keeping those rows on the
  device behind a remap table would avoid the bus for most reads. It would matter for MTP
  drafting, where each draft step's gather is serial and dependent.
- **Precondition.** Pursue only if A1–A4 leave a measurable decode or MTP cost. It adds a
  resource, a policy and a second read path.

## A6. Host format (Q8 recipes only)

- **Hypothesis.** A Q4 host transcode halves the bytes per row (5440 → 2720 B) at a quality cost.
- **Precondition.** Only if the measurements show bytes, not latency, dominate the host gather.
  The flag currently refuses `--embedding-q4/q6`.

## A7. PCIe contention

- **Measure.** Repeat the decode and TTFT measurements with the host KV tier and hybrid
  prefix-cache restores active, which share the link with the zero-copy reads.
