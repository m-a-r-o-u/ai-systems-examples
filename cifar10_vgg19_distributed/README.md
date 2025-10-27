# Distributed CIFAR-10 + VGG19 Training Example

This example demonstrates a practical workflow for training VGG19 on CIFAR-10 with PyTorch, `torchrun`, and multi-GPU DistributedDataParallel (DDP). The goal is to provide an educational baseline that can be referenced in HPC documentation and easily adapted to Slurm job scripts.

## Background

VGG19 is a classic convolutional neural network (CNN) architecture introduced in 2014 that stacks many small 3×3 filters to build deep feature hierarchies. You can read the original paper on the [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) for historical context. CIFAR-10 is a widely used dataset of 60,000 tiny 32×32 color images across ten categories that dates back to 2009 and remains a standard educational benchmark. CNN-based image classification workloads like this have been a mainstay in the deep learning community for over a decade, making them ideal for demonstrating distributed training fundamentals before moving on to more modern models.

## Key features

* **`torchrun` orchestration** – single launcher that plays nicely with Slurm.
* **Rank-aware I/O** – main rank owns downloads, checkpoints, and logs.
* **Pretrained VGG19** – ImageNet weights with a CIFAR-10 head and resizing pipeline.
* **Automatic mixed precision (AMP)** – enabled by default, toggle with `--no-amp`.
* **Deterministic seeding** – aligned Python/NumPy/PyTorch seeds with cuDNN notes.
* **Clear batch-size semantics** – CLI accepts global batch size; LR scales linearly.
* **CSV & TensorBoard metrics** – quick console summaries plus structured artifacts.
* **Checkpointing & resume** – latest/best snapshots and broadcasted restarts.

## Repository layout

```
cifar10_vgg19_distributed/
├── README.md              # This document
├── requirements.txt       # Minimal dependencies
└── train.py               # torchrun entrypoint
```

Artifacts (checkpoints, CSV, TensorBoard logs) are written to `./artifacts` by default. Adjust this with `--output-dir` if your HPC workflow prefers scratch or parallel filesystems.

## Environment setup

1. Create or activate a Python environment that includes CUDA-enabled PyTorch (a virtualenv, conda env, or container image all work).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The requirements file pins high-level libraries only; prefer the official [PyTorch install selector](https://pytorch.org/get-started/locally/) to match your CUDA toolkit.

## Launching with `torchrun`

```bash
torchrun --nproc_per_node=4 cifar10_vgg19_distributed/train.py \
  --data-dir /path/to/datasets \
  --output-dir /path/to/artifacts \
  --global-batch-size 256 \
  --epochs 20
```

For a quick local run, start with defaults:

```bash
torchrun --nproc_per_node=1 cifar10_vgg19_distributed/train.py --data-dir ./data --output-dir ./output --global-batch-size 256 --epochs 20
```

* `torchrun` automatically populates `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`. The script refuses to run without them to catch accidental `python train.py` executions.
* The **global** batch size (256 by default) is split across ranks. For example, 4 GPUs → 64 images per GPU. The learning rate follows the linear scaling rule: `actual_lr = base_lr * (global_batch / 256)`.
* Increase `--num-workers` to match available CPU cores; 4–8 workers per GPU is a common sweet spot.
* Tune CPU affinity by passing `--omp-num-threads` if the automatic choice (CPU cores ÷ local world size) does not match your job layout.
* Disable AMP for debugging with `--no-amp`.

### Example Slurm snippet

```bash
#!/bin/bash
#SBATCH --partition=lrz-dgx-a100-80x8
#SBATCH --job-name=cifar10-vgg19
#SBATCH --gres=gpu:1                  # Change manually when using more GPUs
#SBATCH --time=01:00:00
#SBATCH --output=log_%j.log

# --- Manual settings ---
NPROC_PER_NODE=1                      # Change manually when using more GPUs
IMAGE="$HOME/nvidia+pytorch+23.10-py3.sqsh"
MOUNT="$HOME/ai-systems-examples/cifar10_vgg19_distributed:/workspace"
# -----------------------

srun --container-image="$IMAGE" \
     --container-mounts="$MOUNT" \
     torchrun --nproc_per_node=$NPROC_PER_NODE ./train.py \
              --data-dir ./data \
              --output-dir ./output \
              --epochs 5 \
              --global-batch-size 512
```

The script uses `env://` initialization, so it is compatible with multi-node Slurm jobs when `torchrun` is invoked with `--nnodes`/`--node-rank` (or `--standalone` for single node tests).

## Reproducibility

* `--seed` controls Python, NumPy, and PyTorch RNGs across all ranks.
* `--deterministic` toggles `torch.backends.cudnn.deterministic = True` and disables `cudnn.benchmark`. Deterministic mode is slower but reproducible; the default favors performance.

## Logging & artifacts

All I/O occurs on rank 0 and is stored under `--output-dir`:

* `metrics.csv` – epoch-level metrics with columns `epoch,phase,loss,accuracy,throughput`.
* `tensorboard/` – scalar summaries for train/val/test (loss, accuracy, throughput). Launch `tensorboard --logdir <output-dir>/tensorboard` to visualize.
* `checkpoints/latest.pth` – always-updated checkpoint with epoch, optimizer, and AMP scaler state.
* `checkpoints/best.pth` – highest validation accuracy so far.
* `checkpoints/epoch_XXX.pth` – optional periodic snapshots controlled by `--save-frequency` (disabled by default).

A sample CSV row:

```
epoch,phase,loss,accuracy,throughput
0,train,0.8215,0.7132,834.5
0,val,0.6124,0.7910,1182.3
```

## Resuming training

Use the `--resume` flag with any checkpoint (latest, best, or numbered). The checkpoint is loaded on rank 0 and broadcast to the rest of the world size.

```bash
torchrun --nproc_per_node=4 cifar10_vgg19_distributed/train.py \
  --resume /path/to/artifacts/checkpoints/latest.pth
```

The script restores the epoch counter, optimizer, and AMP scaler. Training automatically proceeds from the next epoch.

## Data pipeline

* Training pipeline: resize to 256 → random crop 224 × 224 → random horizontal flip → tensor conversion → ImageNet normalization.
* Validation/test pipeline: resize to 256 → center crop 224 × 224 → tensor conversion → ImageNet normalization.
* By default, the CIFAR-10 training set is split 45k/5k into train/validation using a fixed generator seed, so results are comparable across runs.

## Extending the example

* Swap models by editing `build_model` (e.g., ResNet or ViT backbones).
* Integrate LR schedulers or optimizers – plug them into the training loop and checkpoint state dict.
* Add extra logging sinks (W&B, MLflow) by hooking into the `is_main_process()` guard.
* Scale to multiple nodes by passing the appropriate `--nnodes`/`--node-rank` parameters to `torchrun` or letting Slurm handle rendezvous via `MASTER_ADDR`/`MASTER_PORT`.

## Troubleshooting

* **Hangs at startup** – ensure every rank can reach `MASTER_ADDR:MASTER_PORT` and that `torchrun` is used instead of `python`.
* **Imbalanced batches** – the script drops the last partial batch during training to keep per-rank batch sizes aligned; disable `drop_last` if you need every sample.
* **AMP underflows** – rerun with `--no-amp` to diagnose, or reduce learning rate.
* **Slow data loading** – increase `--num-workers` or move `--data-dir` to a local SSD on each node.

For deeper experimentation, the code is organized to be easily extended: helper functions are pure and unit-test friendly, and all configuration surfaces through the CLI for shell scripting.

### Eliminating `OMP_NUM_THREADS` warnings

`torchrun` prints a warning and forces `OMP_NUM_THREADS=1` when the variable is unset. The training script now computes a sensible default (CPU cores ÷ local world size) on startup so runs launched with `--nproc_per_node` no longer inherit the slow default. You can override the heuristic explicitly:

```bash
torchrun --nproc_per_node=2 cifar10_vgg19_distributed/train.py \
  --omp-num-threads 8
```

Setting the flag (or exporting `OMP_NUM_THREADS` yourself) suppresses the launcher warning while keeping data loading responsive.
