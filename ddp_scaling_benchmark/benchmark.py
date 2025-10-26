#!/usr/bin/env python3
"""Minimal PyTorch DDP benchmark for single-node GPU scaling."""
import argparse
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

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
    num_layers: int
    output_dim: int
    backend: str
    device: str
    bucket_cap_mb: int
    broadcast_buffers: bool
    static_graph: bool


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="DDP scaling benchmark with synthetic data")
    parser.add_argument("--batch-size", type=int, default=256, help="Per-GPU batch size")
    parser.add_argument("--steps-per-epoch", type=int, default=20, help="Number of training steps per epoch")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to run")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of initial steps to exclude from timing measurements",
    )
    parser.add_argument("--input-dim", type=int, default=4096, help="Input feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Hidden layer dimension")
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of hidden layers (Linear + ReLU blocks) before the output layer",
    )
    parser.add_argument("--output-dim", type=int, default=2048, help="Output dimension")
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
    parser.add_argument(
        "--bucket-cap-mb",
        type=int,
        default=50,
        help="DDP gradient bucket size in megabytes. Increase to reduce all-reduce calls.",
    )
    parser.add_argument(
        "--broadcast-buffers",
        action="store_true",
        help="Enable broadcasting of model buffers. Disabled by default for the synthetic benchmark.",
    )
    parser.add_argument(
        "--static-graph",
        dest="static_graph",
        action="store_true",
        help="Set DDP static_graph=True when the model graph is static (default).",
    )
    parser.add_argument(
        "--no-static-graph",
        dest="static_graph",
        action="store_false",
        help="Disable DDP static graph optimizations.",
    )
    parser.set_defaults(static_graph=True)
    args = parser.parse_args()
    if args.num_layers < 0:
        parser.error("--num-layers must be non-negative")
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=args.output_dim,
        backend=args.backend,
        device=args.device if args.device is not None else "auto",
        bucket_cap_mb=args.bucket_cap_mb,
        broadcast_buffers=args.broadcast_buffers,
        static_graph=args.static_graph,
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
        device = torch.device(config.device)
        if device.type == "cuda" and device.index is None and torch.cuda.is_available():
            # Force the local rank mapping when the override omits an index.
            return torch.device(f"cuda:{local_rank}")
        return device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def log_rank_device_info(rank: int, local_rank: int, device: torch.device) -> None:
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "<not set>")
    if device.type == "cuda":
        current_device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(current_device)
        print(
            f"[Rank {rank}] CUDA_VISIBLE_DEVICES={visible}, LOCAL_RANK={local_rank}, "
            f"current_device=cuda:{current_device}, name={name}"
        )
    else:
        print(
            f"[Rank {rank}] CUDA_VISIBLE_DEVICES={visible}, LOCAL_RANK={local_rank}, "
            f"device={device}"
        )


def barrier_if_distributed(distributed: bool) -> None:
    if distributed:
        dist.barrier()


def build_model(config: BenchmarkConfig, device: torch.device) -> torch.nn.Module:
    layers: List[torch.nn.Module] = []
    in_features = config.input_dim

    for _ in range(config.num_layers):
        layers.append(torch.nn.Linear(in_features, config.hidden_dim))
        layers.append(torch.nn.ReLU())
        in_features = config.hidden_dim

    layers.append(torch.nn.Linear(in_features, config.output_dim))

    model = torch.nn.Sequential(*layers)
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
        cuda_index = device.index if device.index is not None else local_rank
        torch.cuda.set_device(cuda_index)
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

    log_rank_device_info(rank, local_rank, device)

    torch.manual_seed(42)

    model = build_model(config, device)
    if distributed:
        ddp_kwargs: Dict[str, object] = dict(
            bucket_cap_mb=config.bucket_cap_mb,
            broadcast_buffers=config.broadcast_buffers,
            gradient_as_bucket_view=True,
            static_graph=config.static_graph,
        )
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[device.index], output_device=device.index)
        model = DDP(model, **ddp_kwargs)

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
