# MiniMax H3 Sol RT

Experimental MiniMax H3 Sol Attention research exploring RT-core-assisted routing, native OptiX integration, INT8 attention, and CUDA/RT workload overlap on RTX GPUs.

## Project Goal

The goal of this project was to accelerate MiniMax H3 generation on an RTX 5060 Ti 16 GB by using RT cores for routing work while preserving Sol Attention quality and semantics.

The target was not to reduce model quality or skip computation. The intent was to redistribute suitable work across RTX hardware so RT cores could participate in real generation while CUDA/Tensor-core work continued in parallel.

Primary target:

- preserve Base Sol quality
- preserve Sol Attention routing semantics
- use RT cores in real generation, not only validation
- reduce end-to-end denoise time
- ideally beat the Base Sol INT8 reference of about **104.33 s/it**

## Test Platform

- GPU: NVIDIA GeForce RTX 5060 Ti 16 GB
- Architecture: SM120
- CUDA build toolchain: CUDA 12.8
- Runtime: PyTorch 2.10.0 + cu130
- OptiX: 9.1.0
- CMake: 4.4.2
- Visual Studio: 2022 BuildTools
- ComfyUI: 0.34.6

## H3 Sol Attention Structure

The H3 path tested in this project uses custom Sol Attention with:

- Q/K/V: `[B, T, 56, 128]`
- 56 heads
- head dimension 128
- block size 64
- group size 32
- K block summaries -> `kc`
- V block summaries -> `vc`

For each query block/head and KV block, a routing score determines whether the block uses exact or approximate attention.

A KV block is exact when one of the following is true:

- route score is above the threshold
- local block distance is within ±1
- sink-conditioning requires exact attention

Exact blocks use full QK/PV. Non-exact blocks use block summaries.

The main tuning values used throughout the project were:

```text
tau = 1.30
min_tokens = 4096
q_chunk_blocks = 32
thresh_type = diag
sink_conditioning = exact_kv
int8_qk = true
int8_pv = true
```

## Routing Identity

A useful identity for the routing path is:

```text
mean_i(q_i · kc) = mean(q_i) · kc
```

This means a query-block centroid can reproduce the mean routing score algebraically and can reduce routing work.

Floating-point accumulation order can still affect threshold-edge decisions, so exact production equivalence requires direct comparison against original Sol route decisions.

## Native OptiX Backend

A native Windows OptiX router was successfully compiled and loaded.

The async exported function was:

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

The async path uses the supplied CUDA stream for:

- group-max work
- async memory operations
- OptiX launch
- device-to-device result copies

It does not require a global `cudaDeviceSynchronize()` in the async function.

### Working Windows build approach

The ComfyUI installation path contained parentheses, which caused CUDA device-link issues. Building in an external directory was reliable.

```powershell
Remove-Item -Recurse -Force "E:\Img_Gen\sol_rt_build" -ErrorAction SilentlyContinue

cmd /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cmake -S "E:\Img_Gen\ComfyUI-Latest\ComfyUI (1)\ComfyUI\custom_nodes\ComfyUI-sol-attn-RT-v3.2-INT8-OVERLAP\sol_rt_native" -B "E:\Img_Gen\sol_rt_build" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=120 -DOPTIX_ROOT="C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0" && cmake --build "E:\Img_Gen\sol_rt_build"'
```

Useful OptiX include order:

```cpp
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
```

The Windows DLL export `sol_rt_route_async` was verified successfully.

## Development Progression

### 1. Shadow validation

The first H3 implementation left original Sol Attention in control of the output while RT routing ran as a validator.

Observed:

- 102 validated calls
- 0 routing mismatches in the tested representation
- about 56,831 tokens
- N about 888 blocks

This established that the RT routing path could reproduce the tested route representation, but it was intentionally slow because both original Sol and RT routing work were running.

### 2. Active RT routing

The next version allowed RT-derived routing to participate in generation.

Observed:

- 200 active calls
- `false_neg=0` in tested runs
- generation around **149.33 s/it**

The OptiX trace itself was not the main cost. Data preparation, routing finalization, masks, synchronization, and the changed attention path dominated total overhead.

### 3. Serialized CSR split

This became the strongest quality-good RT reference.

Execution:

```text
route
-> finalize
-> CSR exact list
-> one full INT8 attention launch
```

Representative performance:

- about **119.5 s/it**
- attention commonly about 1.15–1.30 s per H3 block
- route wall commonly about 150–170 ms
- some route calls around 430 ms
- `false_neg=0`

No visible quality issue was reported for this serialized configuration.

Representative configuration:

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

### 4. Slabbed overlap

An overlap version split attention into 28 query slabs and tried to overlap RT routing with slab-level attention.

Conceptually:

```text
route slab
-> wait
-> compact slab
-> attention slab
-> repeat
```

This increased launch and synchronization overhead.

At 28 slabs across 50 attention blocks, the design created roughly 1,400 slab-level attention launches per denoise step.

A scalar synchronization pattern such as:

```python
total_exact = int(offsets[-1].item())
```

also forced CPU/GPU synchronization in the slab loop.

Observed:

- about **123.34 s/it**
- slower than serialized split
- one localized quality issue was observed in the generated video

This demonstrated that overlap must preserve large-kernel execution and avoid fragmenting attention.

### 5. Whole-mask router pipeline

A later version preserved one full attention launch while pipelining routing stages.

Representative log marker:

```text
[Base-Sol RT PIPE] whole-mask router active: tokens=56831 N=888 slabs=28 q_chunk=32; attention remains ONE full launch
```

Observed warm result:

- about **118.02 s/it**

This recovered most of the serialized split behavior while keeping async routing, but it still did not reach the Base Sol reference.

## Main Performance Finding

The most important finding was:

```text
RT participation != automatic RT acceleration
```

The native OptiX trace could be very fast, often around **12 ms**, while total router wall time could be around **150 ms** or even **430 ms** on stalled calls.

That means the expensive parts were often outside the RT trace itself:

- centroid/route-score preparation
- group preparation
- route-result finalization
- synchronization
- memory movement
- metadata conversion
- CSR/mask construction

Moving a predicate to RT cores is not enough if the CUDA-side route work, conversion work, and a second attention data path are still paid.

The RT path must replace existing work rather than sit beside it.

## Overlap Finding

The serialized logs exposed a promising scheduling pattern:

```text
route call N
then
CSR + attention N
then
next route
```

In theory, if route work could overlap independent compute, visible time could approach:

```text
max(route, attention)
```

instead of:

```text
route + attention
```

With roughly 150–170 ms of normal route time across 50 H3 blocks, full hiding could theoretically save around 8 seconds per step.

The important dependency limitation is that routing inputs for the next transformer block are normally not available until that block's Q/K have been produced. Therefore, true cross-transformer-block overlap is constrained by the transformer dependency chain.

The safest overlap opportunities are inside the same block, after route inputs exist and before route results are consumed.

## Quality Findings

### Base Sol INT8

- known-good quality
- about **104.33 s/it**

### Serialized RT split

- no visible quality issue reported
- about **119.5 s/it**

### Slabbed RT overlap

- one localized quality issue observed
- about **123.34 s/it**

A `false_neg=0` result does not by itself prove full equivalence to original Base Sol routing. It proves consistency with the routing reference used by that RT implementation.

For strict equivalence, original Sol route decisions must be compared directly against RT-derived decisions.

## Memory and Long-Context Findings

Dense route masks become increasingly expensive as the number of blocks grows.

At around 200k tokens, dense `[BH, N, N]` route representations become impractical.

A compact representation such as CSR is required for long-context scaling.

Another important observation was that many tested CSR rows became highly exact/dense. If most blocks are exact, sparse routing has less opportunity to create meaningful speedup.

## Benchmark Summary

| Build | RT used in generation | Attention structure | Quality observation | Approx. speed |
|---|---:|---|---|---:|
| Base Sol INT8 | No | Original Sol path | Good | **104.33 s/it** |
| H3 RT shadow | Shadow only | Original output path | Good | ~147–150 s/it |
| Active RT path | Yes | RT-derived route path | Good in tested run | ~149.33 s/it |
| Serialized RT split | Yes | One full INT8 attention launch | Good | **~119.5 s/it** |
| Slabbed RT overlap | Yes | 28 attention slabs/block | Localized issue | **~123.34 s/it** |
| Whole-mask pipeline | Yes | One full attention launch | No major issue reported | **~118.02 s/it** |

## What Was Proven

This project established that:

- RTX 5060 Ti can compile and load a custom H3 OptiX routing backend
- native RT routing can participate in real MiniMax H3 generation
- the async native route export works in the Windows runtime
- RT route traces themselves can be very fast
- serialized RT split can preserve acceptable visual quality
- fragmenting attention into many small overlap slabs is counterproductive
- excessive CPU-visible scalar synchronization damages GPU overlap
- dense route masks are not suitable for very long contexts
- integration/orchestration overhead is more important than isolated RT trace time

## What Still Requires Formal Proof

The following items were not fully established:

- bitwise or decision-for-decision equivalence with original Base Sol routing
- external-profiler proof of physical RT-core occupancy for every tested path
- a production RT path that beats Base Sol end-to-end
- safe 200k-token production behavior
- practical cross-layer route/attention overlap

For hardware-unit proof, external NVIDIA profiling such as Nsight Systems / Nsight Graphics is preferable to self-reported application logs.

## Architecture That Still Looks Most Promising

The experiments point toward a much tighter design:

```text
Original Base Sol attention kernel
+
precomputed native RT route decision
-
original in-kernel route predicate work
```

The key is to preserve:

- original Sol accumulation order
- one large attention launch
- exact/approximate semantics
- INT8 QK/PV path

while eliminating duplicate routing work and avoiding CSR/mask reconstruction where possible.

## Finalization

This research phase is finalized.

The project successfully demonstrated native OptiX routing inside real MiniMax H3 generation and clarified the practical constraints of CUDA/RT workload splitting on an RTX 5060 Ti.

The strongest validated production reference from this work remains Base Sol INT8 at approximately **104.33 s/it**.

The strongest quality-good RT-assisted reference was the serialized RT split at approximately **119.5 s/it**.

The most valuable result is the architecture knowledge gained: RT routing itself can be fast, but the surrounding data preparation, synchronization, metadata conversion, and attention execution structure determine whether RT participation becomes a real end-to-end speedup.

This repository preserves those findings for future work on tighter Base-Sol/RT integration.