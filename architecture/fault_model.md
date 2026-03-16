# Fault Model Definition

## Fault Types Considered

| Fault Type | Location | Description | Detection Method | Recovery Method |
|------------|----------|-------------|------------------|------------------|
| Transient Compute Fault | GEMM output tensor | Single-value corruption during matrix multiply | Checksum detection | Replay layer execution |
| Soft Error (SRAM/DRAM) | Activation buffer | Bit flip in stored activation | Checksum mismatch | Recompute from checkpoint |
| Timing Fault | Accelerator pipeline | Incorrect intermediate result due to voltage droop | Output validation | Replay computation |

## Assumptions

- Faults are transient (not permanent hardware failure)
- Only a subset of tensor elements are corrupted
- Checksum detection can detect most large deviations
- Replay is possible because clean input activations are checkpointed

## Impact

- Without protection → silent data corruption propagates across layers
- With protection → error detected and corrected before propagation