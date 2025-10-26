#!/usr/bin/env python3
"""Minimal PyTorch DDP benchmark for single-node GPU scaling."""
import argparse
import os
import time
from dataclasses import asdict, dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class BenchmarkConfig:
    batch_size: int
    steps_per_epoch: int
    epochs: int
    warmup_steps: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    backend: str
    device: str


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="DDP scaling benchmark with synthetic data")
    parser.add_argument("--batch-size", type=int, default=512, help="Per-GPU batch size")
    parser.add_argument("--steps-per-epoch", type=int, default=20, help="Number of training steps per epoch")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to run")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of initial steps to exclude from timing measurements",
    )
    parser.add_argument("--input-dim", type=int, default=4096, help="Input feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=2048, help="Hidden layer dimension")
    parser.add_argument("--output-dim", type=int, default=1024, help="Output dimension")
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="Process group backend (default: nccl, use gloo for CPU testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g., cuda:0, cuda, or cpu). Defaults to local GPU if available.",
    )
    args = parser.parse_args()
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        backend=args.backend,
        device=args.device if args.device is not None else "auto",
    )
    return config


def setup_distributed(backend: str) -> Tuple[int, int, int, bool]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world_size, local_rank, world_size > 1

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend=backend)
        return rank, world_size, local_rank, True

    return 0, 1, local_rank, False


def select_device(config: BenchmarkConfig, local_rank: int) -> torch.device:
    if config.device != "auto":
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def barrier_if_distributed(distributed: bool) -> None:
    if distributed:
        dist.barrier()


def build_model(config: BenchmarkConfig, device: torch.device) -> torch.nn.Module:
    model = torch.nn.Sequential(
        torch.nn.Linear(config.input_dim, config.hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(config.hidden_dim, config.hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(config.hidden_dim, config.output_dim),
    )
    return model.to(device)


def record_step_time(
    timings: List[float],
    step_start: float,
    warmup_steps: int,
    global_step: int,
) -> None:
    if global_step <= warmup_steps:
        return
    timings.append(time.perf_counter() - step_start)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(config: BenchmarkConfig) -> None:
    rank, world_size, local_rank, distributed = setup_distributed(config.backend)
    device = select_device(config, local_rank)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(42)

    model = build_model(config, device)
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    timings: List[float] = []
    global_step = 0
    barrier_if_distributed(distributed)

    total_steps = config.steps_per_epoch * config.epochs

    for _ in range(config.epochs):
        for _ in range(config.steps_per_epoch):
            global_step += 1
            inputs = torch.randn(config.batch_size, config.input_dim, device=device)
            targets = torch.randn(config.batch_size, config.output_dim, device=device)

            synchronize_device(device)
            step_start = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            synchronize_device(device)
            record_step_time(timings, step_start, config.warmup_steps, global_step)

    measured_steps = len(timings)
    local_total_time = sum(timings)

    time_tensor = torch.tensor(
        [local_total_time, float(measured_steps)],
        device=device,
        dtype=torch.float64,
    )
    if distributed:
        dist.all_reduce(time_tensor, op=dist.ReduceOp.SUM)

    total_time = time_tensor[0].item()
    total_measured_steps = time_tensor[1].item()

    if distributed:
        average_time = total_time / max(world_size, 1)
        average_steps = total_measured_steps / max(world_size, 1)
    else:
        average_time = total_time
        average_steps = total_measured_steps

    if average_steps > 0:
        avg_step_time = average_time / average_steps
        throughput = world_size * config.batch_size / avg_step_time
        total_samples = int(round(world_size * config.batch_size * average_steps))
    else:
        avg_step_time = float("nan")
        throughput = float("nan")
        total_samples = 0

    if rank == 0:
        print("=== DDP Scaling Benchmark ===")
        print(f"World size       : {world_size}")
        print(f"Device           : {device}")
        print(f"Config           : {asdict(config)}")
        print(f"Warmup steps     : {config.warmup_steps}")
        print(f"Measured steps   : {measured_steps} / {total_steps}")
        if average_steps > 0:
            print(f"Avg step time    : {avg_step_time * 1000:.2f} ms")
            print(f"Throughput       : {throughput:,.0f} samples/s")
            print(f"Measured samples : {total_samples:,}")
        else:
            print("Insufficient measured steps. Increase steps or reduce warmup.")

    barrier_if_distributed(distributed)
    if distributed:
        dist.destroy_process_group()


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
