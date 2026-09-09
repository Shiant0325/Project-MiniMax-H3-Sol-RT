# Native RT Backend

The H3 RT experiments use a native OptiX routing backend invoked from the ComfyUI custom node.

Key async export:

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

The async function is designed to run group-max preparation, OptiX launch, and device-side copies on a supplied CUDA stream without a global device synchronize.

For strict hardware-unit validation, use Nsight Systems or Nsight Graphics rather than relying only on application logs.
