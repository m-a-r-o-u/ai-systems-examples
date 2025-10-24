# Distributed CIFAR-10 + VGG19 Training Example

This example demonstrates a production-ready workflow for training VGG19 on CIFAR-10 with PyTorch, `torchrun`, and multi-GPU DistributedDataParallel (DDP). The goal is to provide an educational baseline that can be referenced in HPC documentation and easily adapted to Slurm, PBS, or other schedulers.

## Key features

* **`torchrun` orchestration** – no `mp.spawn`; launches integrate cleanly with Slurm and other MPI-style runners.
* **Rank-aware I/O** – rank 0 handles dataset downloads, checkpointing, CSV metrics, and TensorBoard logging to avoid races on shared filesystems.
* **Pretrained VGG19** – loads ImageNet weights, replaces the classifier head, and resizes CIFAR-10 inputs to 224×224 with ImageNet normalization for fast convergence.
* **Automatic mixed precision (AMP)** – enabled by default with a `--no-amp` flag for debugging.
* **Deterministic seeding** – consistent seeds across Python, NumPy, and PyTorch with explicit cuDNN notes.
* **Clear batch-size semantics** – CLI uses the **global** batch size (`per_gpu * world_size`); the learning rate scales linearly with that value.
* **CSV & TensorBoard metrics** – short, grep-friendly logs plus structured artifacts for post-run analysis.
* **Checkpointing & resume** – latest, best, and optional per-epoch checkpoints with a `--resume` flag that broadcasts weights to all ranks.

## Repository layout

```
cifar10_vgg19_distributed/
├── README.md              # This document
├── requirements.txt       # Minimal dependencies
└── train.py               # torchrun entrypoint
```

Artifacts (checkpoints, CSV, TensorBoard logs) are written to `./artifacts` by default. Adjust this with `--output-dir` if your HPC workflow prefers scratch or parallel filesystems.

## Environment setup

1. Create or activate a Python environment that includes CUDA-enabled PyTorch.
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

* `torchrun` automatically populates `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`. The script refuses to run without them to catch accidental `python train.py` executions.
* The **global** batch size (256 by default) is split across ranks. For example, 4 GPUs → 64 images per GPU. The learning rate follows the linear scaling rule: `actual_lr = base_lr * (global_batch / 256)`.
* Increase `--num-workers` to match available CPU cores; 4–8 workers per GPU is a common sweet spot.
* Tune CPU affinity by passing `--omp-num-threads` if the automatic choice (CPU cores ÷ local world size) does not match your job layout.
* Disable AMP for debugging with `--no-amp`.

### Example Slurm snippet

```bash
#!/bin/bash
#SBATCH --job-name=cifar10-vgg19
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log

module purge
module load cuda
source /path/to/venv/bin/activate

export OMP_NUM_THREADS=8

torchrun --nproc_per_node=$SLURM_NTASKS_PER_NODE \
  cifar10_vgg19_distributed/train.py \
  --data-dir /scratch/$USER/datasets \
  --output-dir /scratch/$USER/cifar10-vgg19 \
  --epochs 30 \
  --global-batch-size 512 \
  --num-workers 8 \
  --save-frequency 5
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
