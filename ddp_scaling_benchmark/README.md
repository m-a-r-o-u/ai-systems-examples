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
torchrun --nproc_per_node=4 benchmark.py --epochs 3 --steps-per-epoch 20 --warmup-steps 5
```

You can also pass the same arguments through `python -m torch.distributed.run` if preferred.

By default the script picks the local GPU assigned by the launcher. Use `--device cpu` together with `--backend gloo` for CPU-only tests.

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
