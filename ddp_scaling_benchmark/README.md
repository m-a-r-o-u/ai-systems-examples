# DDP Scaling Benchmark

This minimal example provides a synthetic workload for benchmarking single-node GPU scaling with PyTorch DistributedDataParallel (DDP). It can be run with a single GPU (or CPU for testing) as well as multiple GPUs on the same host.

## Features

- Uses `torch.nn.parallel.DistributedDataParallel` for multi-GPU training.
- Supports single-process execution for quick functional checks.
- Measures per-step latency and derived samples/second throughput after a configurable warmup.
- Ready to launch with `torchrun` (recommended) or `python -m torch.distributed.run`.

## Quickstart

Install PyTorch with GPU support that matches your system. Then run:

```bash
# Single process (falls back to CPU if no GPU is available)
python benchmark.py --epochs 1 --steps-per-epoch 10 --warmup-steps 2

# Multi-GPU on a single node
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 benchmark.py --epochs 3 --steps-per-epoch 20 --warmup-steps 5
```

You can also pass the same arguments through `python -m torch.distributed.run` if preferred.

By default the script picks the local GPU assigned by the launcher. Use `--device cpu` together with `--backend gloo` for CPU-only tests.

### Verify GPU placement

Poor scaling is often caused by multiple ranks sharing a single GPU. The benchmark now prints the per-rank mapping, e.g.:

```
[Rank 0] CUDA_VISIBLE_DEVICES=0,1, LOCAL_RANK=0, current_device=cuda:0, name=NVIDIA A100-SXM4-40GB
[Rank 1] CUDA_VISIBLE_DEVICES=0,1, LOCAL_RANK=1, current_device=cuda:1, name=NVIDIA A100-SXM4-40GB
```

If both ranks report `cuda:0`, the launch configuration is wrong. Re-run the job with an explicit visibility mask as shown above and confirm with `watch -n 0.5 nvidia-smi` that each process occupies a different GPU.

## Output

The rank-0 process prints a summary similar to:

```
=== DDP Scaling Benchmark ===
World size       : 4
Device           : cuda:0
Config           : {...}
Warmup steps     : 5
Measured steps   : 55 / 60
Avg step time    : 8.12 ms
Throughput       : 27,400 samples/s
Measured samples : 140,800
```

Compare the reported throughput across different `--nproc_per_node` values to understand how well the workload scales with additional GPUs.

## Experiment report

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed measurements, interpretations, and suggested follow-up improvements from the latest DDP scaling studies.

## Scaling tips

- The default configuration increases the per-rank batch size and model width so that each step spends ~15–30 ms in compute before communication. This yields more reliable scaling measurements than the tiny default model.
- Use `--bucket-cap-mb` to increase the DDP gradient bucket size (50 MB by default) when the model contains many small tensors.
- `--broadcast-buffers` is disabled by default to avoid redundant synchronisation of random buffers in this synthetic workload. Enable it if your model relies on buffer state (e.g. BatchNorm running stats).
- The script enables `static_graph=True` and `gradient_as_bucket_view=True` for DDP. Disable static graph optimisations with `--no-static-graph` when the model structure changes across iterations.
- Adjust model size with `--hidden-dim` and explore deeper or shallower networks with `--num-layers` to evaluate how computation-heavy workloads scale.
