# MiniMax H3 Sol RT — Technical Findings

This document preserves the H3-only technical findings from the MiniMax H3 Sol Attention RT-core research project.

## Goal

Use RT cores for useful routing work during real MiniMax H3 generation while preserving Sol Attention quality and semantics, with the end goal of reducing end-to-end generation time.

Reference target:

- Base Sol INT8: ~104.33 s/it
- RTX 5060 Ti 16 GB (SM120)
- tau = 1.30
- block size = 64
- group size = 32
- 56 heads
- head_dim = 128
- INT8 QK/PV enabled

## H3 Sol Attention

For each query block/head, Sol Attention routes KV blocks between:

- exact full QK/PV
- approximate kc/vc summary attention

A block becomes exact when its route score passes threshold, it is local (±1 block), or sink-conditioning requires exact attention.

A useful routing identity is:

`mean_i(q_i · kc) = mean(q_i) · kc`

This allows centroid-based route-score calculation, though floating-point threshold-edge equivalence still needs direct validation against the original Sol predicate.

## Native OptiX Routing

A native OptiX backend was successfully compiled and loaded on Windows for RTX 5060 Ti.

Async export:

```cpp
extern "C" API int sol_rt_route_async(
    uint64_t route_scores_ptr,
    uint64_t thresholds_ptr,
    uint64_t out_groups_ptr,
    uint64_t out_counts_ptr,
    int rows,
    int kv_blocks,
    uint64_t stream_ptr
);
```

The async path uses a supplied CUDA stream and avoids global device synchronization inside the async function.

## Key Experimental Results

| Build | RT in generation | Structure | Quality observation | Approx. speed |
|---|---:|---|---|---:|
| Base Sol INT8 | No | Original Sol path | Good | **104.33 s/it** |
| H3 RT shadow | Shadow only | Original output path | Good | ~147–150 s/it |
| Active RT path | Yes | RT-derived routing | Good in tested run | ~149.33 s/it |
| v3.1 serialized RT split | Yes | One full INT8 attention launch | Good | **~119.55 s/it** |
| v3.2 overlap | Yes | 28 attention slabs/block | Localized quality issue | **~123.34 s/it** |
| v3.3 router pipeline | Yes | One full attention launch | No major issue reported | **~118.02 s/it** |
| v3.4 safe overlap | Intended | One full attention launch | Not benchmarked cleanly | Incomplete |

## Strongest Quality-Good RT Reference

v3.1 serialized split:

```text
backend = base_rt_split
rt_proof_mode = native_rt
tau = 1.30
min_tokens = 4096
q_chunk_blocks = 32
output_check_calls = 0
strict = true
thresh_type = diag
int8_qk = true
int8_pv = true
sink_conditioning = exact_kv
```

Observed warm step: ~119.55 s/it.

The user reported no visible quality issue in this serialized configuration.

## Why the Slabbed Overlap Lost

v3.2 split each H3 attention call into 28 query slabs.

At 50 H3 attention blocks per step, this produced roughly 1,400 attention launches per denoise step, plus per-slab synchronization and compact/finalize work.

The result was slower (~123.34 s/it) and introduced one localized visual quality issue.

Conclusion: overlap must preserve large-kernel execution and avoid fragmenting attention.

## Important Router Finding

The native RT trace itself was often around ~12 ms, while total router wall time was commonly ~150–170 ms and could rise to ~430 ms on stalled calls.

Therefore, the dominant cost was often outside OptiX:

- centroid/route-score preparation
- finalization
- synchronization
- metadata conversion
- memory movement
- CSR/mask construction

The project demonstrated that:

`RT participation != automatic RT acceleration`

For RT to produce a net win, it must replace existing Sol work rather than add a sidecar pipeline around it.

## Long-Context Finding

Dense `[BH, N, N]` route masks do not scale well toward 100k–200k tokens.

A compact representation such as CSR is required.

However, tested exact-entry counts were also very high, meaning the current exact/sink policy can become close to dense in many rows. Sparse routing has limited benefit when the effective exact set is already very dense.

## Hardware-Proof Methodology

Application logs prove that the native path was invoked, but external NVIDIA profiling is the correct method for physical RT-core proof.

Recommended A/B:

- `native_rt`: native OptiX route + NVTX marker
- `cuda_control`: same logical predicate without native OptiX

Use Nsight Systems / Nsight Graphics for external proof.

## Final Architecture Insight

The most promising future direction is not a separate CSR attention architecture.

It is:

```text
Original Base Sol attention kernel
+
precomputed RT route decision
-
original in-kernel route predicate work
```

while preserving:

- original Sol accumulation order
- one large attention launch
- exact/approximate semantics
- INT8 QK/PV
- no duplicate routing computation
- no dense full-mask reconstruction

## Finalization

The research phase is finalized.

The project proved that native OptiX routing can participate in real MiniMax H3 generation on RTX 5060 Ti and clarified the integration bottlenecks.

Best validated production reference:

- Base Sol INT8: ~104.33 s/it

Best quality-good RT-assisted reference:

- v3.1 serialized RT split: ~119.55 s/it

The main result is the architecture knowledge gained: RT routing can be fast, but orchestration, synchronization, representation conversion, and attention execution structure determine whether that hardware participation becomes an end-to-end speedup.
