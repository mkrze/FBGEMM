# XPU inference recovery reference

This branch pins FBGEMM v1.8.0, source base
`99a76f4ed785d2d00579eae5fd45440a1242d657`, used with `fbgemm-gpu-cpu==1.8.0`.

No FBGEMM Python or kernel changes were necessary for the validated non-sharded
INT4/INT8 inference path. Pass `device=torch.device("xpu:0")` explicitly to
`IntNBitTableBatchedEmbeddingBagsCodegen`; omitted-device CUDA defaults are not
changed by this branch. Import `fbgemm_xpu` to register the XPU operators.

Validated configuration: `PoolingMode.NONE`, `EmbeddingLocation.DEVICE`,
CPU-prepared packed weights, row alignment 16, bounds WARNING version 1,
no cache/pruning. The plugin supplies PTXPULIB-151 bounds and PTXPULIB-150
quantized lookup. This does not enable training FUSED TBE or all FBGEMM devices.

Reproduction entry point is in the user's Generative Recommenders fork:
`https://github.com/mkrze/generative-recommenders/tree/xpu/mlperf-dlrmv3-inference/scripts/mlperf-dlrm-v3-xpu`.
Use the CPU wheel, not a CUDA FBGEMM wheel or an unpinned checkout of main.