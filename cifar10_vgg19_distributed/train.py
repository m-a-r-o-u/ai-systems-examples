"""Distributed CIFAR-10 training with VGG19 and torchrun.

This script is designed for educational HPC documentation. It demonstrates
how to train an ImageNet-pretrained VGG19 model on CIFAR-10 using
DistributedDataParallel (DDP) with torchrun, automatic mixed precision (AMP),
and rank-aware logging/checkpointing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class EpochStats:
    """Container for metrics aggregated across ranks."""

    loss: float
    accuracy: float
    throughput: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed VGG19 training on CIFAR-10")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"), help="Directory used to store CIFAR-10")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./artifacts"),
        help="Directory where checkpoints and logs are written",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--global-batch-size", type=int, default=256, help="Global batch size across all ranks")
    parser.add_argument(
        "--base-lr",
        type=float,
        default=0.01,
        help="Learning rate for a global batch size of 256. Scaled linearly with the actual global batch size.",
    )
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable automatic mixed precision")
    parser.set_defaults(amp=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force cuDNN deterministic convolution (slower but reproducible). If not set, cuDNN benchmark is enabled.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per process")
    parser.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--save-frequency", type=int, default=0, help="Save a numbered checkpoint every N epochs (0 disables)")
    parser.add_argument("--log-frequency", type=int, default=10, help="How often (in steps) to log training progress")
    args = parser.parse_args()

    # torchrun sets LOCAL_RANK/ RANK / WORLD_SIZE
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("This script must be launched with torchrun so distributed metadata is provided.")
    return args


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def set_random_seeds(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def broadcast_object(obj):
    """Broadcast a Python object from rank 0 to all ranks."""

    if get_world_size() == 1:
        return obj
    object_list = [obj] if is_main_process() else [None]
    dist.broadcast_object_list(object_list)
    return object_list[0]


def init_distributed() -> Tuple[int, int, torch.device]:
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = get_rank()
    world_size = get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return rank, world_size, device


def build_datasets(data_dir: Path) -> Tuple[Dataset, Dataset, Dataset]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    if is_main_process():
        datasets.CIFAR10(root=data_dir, train=True, download=True)
        datasets.CIFAR10(root=data_dir, train=False, download=True)
    if dist.is_initialized():
        dist.barrier()

    train_full = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=None)
    val_size = 5000
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(0)
    train_subset, val_subset = torch.utils.data.random_split(train_full, [train_size, val_size], generator=generator)
    train_dataset = _wrap_dataset_with_transform(train_subset, train_transform)
    val_dataset = _wrap_dataset_with_transform(val_subset, eval_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=eval_transform)
    return train_dataset, val_dataset, test_dataset


class _WrappedDataset(Dataset):
    def __init__(self, subset: torch.utils.data.Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.subset)

    def __getitem__(self, idx):  # type: ignore[override]
        x, y = self.subset[idx]
        return self.transform(x), y


def _wrap_dataset_with_transform(subset, transform):
    return _WrappedDataset(subset, transform)


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    per_device_batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)

    common_kwargs = dict(num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        sampler=train_sampler,
        drop_last=True,
        **common_kwargs,
    )
    val_loader = DataLoader(val_dataset, batch_size=per_device_batch_size, sampler=val_sampler, **common_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=per_device_batch_size, sampler=test_sampler, **common_kwargs)
    return train_loader, val_loader, test_loader


def build_model(num_classes: int = 10) -> nn.Module:
    weights = models.VGG19_Weights.IMAGENET1K_V1
    model = models.vgg19(weights=weights)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    return model


def accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        correct = (pred == target).sum()
        return correct.float() / target.size(0)


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    if get_world_size() == 1:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor /= get_world_size()
    return tensor


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    log_frequency: int,
    use_amp: bool,
) -> EpochStats:
    model.train()
    running_loss = 0.0
    running_correct = 0.0
    num_batches = 0
    epoch_start = time.time()
    last_log_time = epoch_start
    last_log_step = 0

    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_loss = reduce_tensor(loss.detach(), average=True)
        batch_acc = reduce_tensor(accuracy(outputs, targets).detach(), average=True)

        running_loss += batch_loss.item()
        running_correct += batch_acc.item()
        num_batches += 1

        if is_main_process() and (step % log_frequency == 0 or step == len(loader)):
            now = time.time()
            steps_since_log = step - last_log_step
            throughput = loader.batch_size * get_world_size() * steps_since_log / max(now - last_log_time, 1e-9)
            print(
                f"Epoch {epoch:03d} Step {step:04d}/{len(loader):04d} "
                f"Loss {batch_loss.item():.4f} Acc {batch_acc.item():.4f} "
                f"Throughput {throughput:.1f} img/s"
            )
            last_log_time = now
            last_log_step = step

    epoch_loss = running_loss / max(num_batches, 1)
    epoch_acc = running_correct / max(num_batches, 1)
    epoch_time = time.time() - epoch_start
    samples_per_sec = loader.batch_size * get_world_size() * num_batches / max(epoch_time, 1e-9)
    return EpochStats(loss=epoch_loss, accuracy=epoch_acc, throughput=samples_per_sec)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, use_amp: bool) -> EpochStats:
    model.eval()
    running_loss = 0.0
    running_correct = 0.0
    num_batches = 0
    start_time = time.time()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            batch_loss = reduce_tensor(loss.detach(), average=True)
            batch_acc = reduce_tensor(accuracy(outputs, targets).detach(), average=True)

            running_loss += batch_loss.item()
            running_correct += batch_acc.item()
            num_batches += 1

    epoch_loss = running_loss / max(num_batches, 1)
    epoch_acc = running_correct / max(num_batches, 1)
    elapsed = time.time() - start_time
    throughput = loader.batch_size * get_world_size() * num_batches / max(elapsed, 1e-9)
    return EpochStats(loss=epoch_loss, accuracy=epoch_acc, throughput=throughput)


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    best_acc: float,
    is_best: bool,
    save_frequency: int,
) -> None:
    if not is_main_process():
        return

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_acc": best_acc,
    }

    latest_path = checkpoint_dir / "latest.pth"
    torch.save(state, latest_path)

    if is_best:
        torch.save(state, checkpoint_dir / "best.pth")

    if save_frequency > 0 and epoch % save_frequency == 0:
        torch.save(state, checkpoint_dir / f"epoch_{epoch:03d}.pth")


def load_checkpoint_if_available(
    resume_path: Path | None,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> Tuple[int, float]:
    start_epoch = 0
    best_acc = 0.0

    if resume_path is None:
        return start_epoch, best_acc

    if is_main_process():
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint {resume_path} not found")
        checkpoint = torch.load(resume_path, map_location="cpu")
    else:
        checkpoint = None

    checkpoint = broadcast_object(checkpoint)

    model.module.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_acc = checkpoint.get("best_acc", 0.0)
    if is_main_process():
        print(f"Resumed from {resume_path} at epoch {start_epoch}")
    return start_epoch, best_acc


def create_metric_loggers(output_dir: Path) -> Tuple[SummaryWriter | None, csv.writer | None, callable]:
    if not is_main_process():
        return None, None, lambda *args, **kwargs: None

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    file = open(csv_path, "a", newline="", buffering=1)
    csv_writer = csv.writer(file)

    if file.tell() == 0:
        csv_writer.writerow(
            [
                "epoch",
                "phase",
                "loss",
                "accuracy",
                "throughput",
            ]
        )

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    def finalize():
        writer.flush()
        writer.close()
        file.flush()
        file.close()

    return writer, csv_writer, finalize


def log_metrics(
    writer: SummaryWriter | None,
    csv_writer: csv.writer | None,
    epoch: int,
    phase: str,
    stats: EpochStats,
) -> None:
    if not is_main_process():
        return
    if writer:
        writer.add_scalar(f"{phase}/loss", stats.loss, epoch)
        writer.add_scalar(f"{phase}/accuracy", stats.accuracy, epoch)
        writer.add_scalar(f"{phase}/throughput", stats.throughput, epoch)
    if csv_writer:
        csv_writer.writerow([epoch, phase, stats.loss, stats.accuracy, stats.throughput])


def main() -> None:
    args = parse_args()
    rank, world_size, device = init_distributed()

    if args.global_batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size {args.global_batch_size} must be divisible by world size {world_size}."
        )

    per_device_batch_size = args.global_batch_size // world_size
    lr = args.base_lr * (args.global_batch_size / 256.0)

    if is_main_process():
        print("Configuration", json.dumps({**vars(args), "per_device_batch_size": per_device_batch_size, "lr": lr}, default=str, indent=2))

    set_random_seeds(args.seed, args.deterministic)

    train_dataset, val_dataset, test_dataset = build_datasets(args.data_dir)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, per_device_batch_size, args.num_workers
    )

    model = build_model()
    model.to(device)
    model = DDP(model, device_ids=[device], output_device=device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch, best_acc = load_checkpoint_if_available(args.resume, model, optimizer, scaler)

    writer, csv_writer, finalize_logs = create_metric_loggers(args.output_dir)

    try:
        for epoch in range(start_epoch, args.epochs):
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                epoch,
                args.log_frequency,
                args.amp,
            )
            val_stats = evaluate(model, val_loader, criterion, device, args.amp)

            log_metrics(writer, csv_writer, epoch, "train", train_stats)
            log_metrics(writer, csv_writer, epoch, "val", val_stats)

            is_best = val_stats.accuracy > best_acc
            best_acc = max(best_acc, val_stats.accuracy)
            save_checkpoint(args.output_dir, epoch, model, optimizer, scaler, best_acc, is_best, args.save_frequency)

        test_loader.sampler.set_epoch(args.epochs)
        test_stats = evaluate(model, test_loader, criterion, device, args.amp)
        log_metrics(writer, csv_writer, args.epochs, "test", test_stats)
        if is_main_process():
            print(f"Test accuracy: {test_stats.accuracy * 100:.2f}%")
    finally:
        finalize_logs()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
