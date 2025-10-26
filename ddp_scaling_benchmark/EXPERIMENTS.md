# DDP Scaling Benchmark Experiments

This document captures the end-to-end setup, results, and takeaways from the distributed data parallel (DDP) scaling experiments conducted on the `ddp_scaling_benchmark` project, along with recommended follow-up improvements for future iterations.

## Experiment Setup

- **Hardware:** 2× NVIDIA Tesla V100-PCIE-16GB (PCIe interconnect, no NVLink)
- **Model:** Feed-forward MLP with `input_dim=4096`, `hidden_dim=4096`, `output_dim=2048`
- **DDP Configuration:** `backend="nccl"`, `bucket_cap_mb=50`, `broadcast_buffers=False`, `static_graph=True`
- **Precision:** FP32 (no mixed precision or gradient compression)
- **Training Loop:** `epochs=3`, `steps_per_epoch=20`, `warmup_steps=5`
- **Batch Sizes Tested:** 256 and 4096
- **World Sizes Tested:** 1 GPU and 2 GPUs (DDP)

## Results Summary

| GPUs | Batch | Avg. Step Time (ms) | Throughput (samples/s) | Speedup vs. 1 GPU | Scaling Efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 256 | 7.60 | 33,700 | – | – |
| 2 | 256 | 41.40 | 12,369 | 0.37× | – |
| 1 | 4,096 | 99.77 | 41,052 | – | – |
| 2 | 4,096 | 114.24 | 71,710 | 1.75× | 87% |

## Interpretation

### 1. Small batch (256) ⇒ communication dominated
- Short compute time per step (~7.6 ms) leaves little opportunity to hide the latency of NCCL collectives.
- Synchronizing numerous small gradient tensors introduces substantial overhead, so the two-GPU run underperforms the single-GPU baseline (33k → 12k samples/s).
- Outcome: Multi-GPU training is slower than single GPU, characteristic of a latency-bound regime.

### 2. Large batch (4,096) ⇒ compute dominated
- Compute time per step increases to ~100 ms, making it much easier to amortize communication costs.
- Two GPUs deliver 71.7k samples/s, a 1.75× speedup with ~87% scaling efficiency — strong for PCIe-connected V100s.
- Outcome: DDP scaling approaches the hardware limit when the compute/communication ratio is favorable.

### 3. Scaling pattern
- Larger batch sizes reduce the fraction of time spent on communication, improving scaling efficiency from <40% to ~87%.
- For this model and hardware, batches around 4k provide a good balance between throughput and efficiency; higher batches may yield diminishing but still positive returns.

## Key Insights

1. **Batch size dictates scaling efficiency.** Increasing the per-step workload is essential to reap the benefits of multi-GPU training.
2. **Interconnect topology matters.** PCIe bandwidth limits ultimate scaling, so expect even better efficiency on NVLink-equipped systems.
3. **DDP settings are correctly tuned.** The observed placement (`cuda:0`, `cuda:1`) and throughput ratios confirm the benchmark is functioning as expected.
4. **Maintain sufficient compute intensity.** Keep per-GPU compute time above ~30–50 ms to hide communication latency in distributed training or inference workloads.

## Further Improvements

To make the benchmark more educational and actionable, consider the following enhancements:

1. **Visualize scaling trends.** Plot throughput versus number of GPUs for multiple batch sizes and include an ideal linear-scaling reference.
2. **Quantify communication overhead.** Report communication time explicitly (e.g., `T_comm = T_multi_gpu - T_single_gpu`) and express it as a fraction of total step time.
3. **Expose model hyperparameters via CLI.** Add flags such as `--hidden-dim` and `--num-layers` to demonstrate how model depth affects scaling.
4. **Add profiling mode.** Integrate `torch.profiler` to capture kernel and NCCL timelines on demand (e.g., via a `--profile` flag).
5. **Connect results to Amdahl's Law.** Estimate the communication fraction `f` from measurements and compare observed speedups to theoretical limits.
6. **Experiment with environment knobs.** Encourage exploring settings like `NCCL_DEBUG`, `bucket_cap_mb`, and `OMP_NUM_THREADS` to observe their impact on scaling.
7. **Generate an efficiency report.** Summarize scaling efficiency and communication overhead automatically at the end of each run.
8. **Compare interconnects when available.** Re-run the benchmark on NVLink-enabled hardware to highlight how topology shifts efficiency from ~87% toward ~95–98%.

These additions turn the raw measurements into a comprehensive learning resource for understanding DDP performance trade-offs.

