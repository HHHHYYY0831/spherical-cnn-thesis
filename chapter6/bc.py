# chap6_phased_sweep_runpy_aligned.py
# Phased sweep aligned to the behaviour of provided run.py:
# - No dataset normalization (same as run.py)
# - test_loader shuffle=True (same as run.py)
# - Same original topology: S2Conv -> ReLU -> SO3Conv -> ReLU -> so3_integrate -> Linear
#
# Output structure:
#   results_root/
#     baseline/
#     phaseA_bandwidth/
#     phaseB_channels/
#     phaseC_interaction/
# each contains: results.csv + history_*.json
#
# Run:
#   python chap6_phased_sweep_runpy_aligned.py --data_path s2_mnist.gz
# Optional:
#   --epochs 10 (quick)
#   --include_optional
#   --device cuda / cpu
#   --phase baseline / phaseA_bandwidth / phaseB_channels / phaseC_interaction / all

# pylint: disable=E1101,R,C

import os
import json
import time
import csv
import argparse
from itertools import product

import numpy as np
import collections, collections.abc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import gzip
import pickle

from s2cnn import SO3Convolution, S2Convolution, so3_integrate
from s2cnn import so3_near_identity_grid, s2_near_identity_grid


# -------------------------
# Data (aligned to run.py)
# -------------------------
def load_data_runpy_aligned(path: str, batch_size: int):
    """
    Aligns with your run.py:
    - No normalization
    - train_loader shuffle=True
    - test_loader shuffle=True (as in run.py)
    """
    with gzip.open(path, "rb") as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(
        dataset["train"]["images"][:, None, :, :].astype(np.float32)
    )
    train_labels = torch.from_numpy(
        dataset["train"]["labels"].astype(np.int64)
    )

    train_dataset = data_utils.TensorDataset(train_data, train_labels)
    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,   # run.py
        drop_last=False
    )

    test_data = torch.from_numpy(
        dataset["test"]["images"][:, None, :, :].astype(np.float32)
    )
    test_labels = torch.from_numpy(
        dataset["test"]["labels"].astype(np.int64)
    )

    test_dataset = data_utils.TensorDataset(test_data, test_labels)
    test_loader = data_utils.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,   # run.py (yes)
        drop_last=False
    )

    return train_loader, test_loader, train_dataset, test_dataset


# -------------------------
# Model (topology fixed; only (b, f) change)
# -------------------------
class S2ConvNet_BWCh(nn.Module):
    """
    Exactly the same topology as run.py original:
      S2Convolution(nfeature_in=1, nfeature_out=f1, b_in=b_in, b_out=b_l1, grid=grid_s2)
      ReLU
      SO3Convolution(nfeature_in=f1, nfeature_out=f2, b_in=b_l1, b_out=b_l2, grid=grid_so3)
      ReLU
      so3_integrate
      Linear(f2 -> 10)
    """

    def __init__(self, b_in: int, b_l1: int, b_l2: int, f1: int, f2: int, f_output: int = 10):
        super().__init__()

        grid_s2 = s2_near_identity_grid()    # same default as run.py original
        grid_so3 = so3_near_identity_grid()  # same default as run.py original

        self.conv1 = S2Convolution(
            nfeature_in=1,
            nfeature_out=f1,
            b_in=b_in,
            b_out=b_l1,
            grid=grid_s2
        )

        self.conv2 = SO3Convolution(
            nfeature_in=f1,
            nfeature_out=f2,
            b_in=b_l1,
            b_out=b_l2,
            grid=grid_so3
        )

        self.out_layer = nn.Linear(f2, f_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = so3_integrate(x)
        x = self.out_layer(x)
        return x


# -------------------------
# Utils
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    # run.py doesn't fix seeds; we do for reproducible comparisons
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def eval_accuracy_runpy_style(model: nn.Module, test_loader, device) -> float:
    """
    run.py style:
      correct / total using torch.max
    """
    model.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).long().sum().item()
    return correct / max(total, 1)


def validate_cfg(cfg: dict) -> bool:
    # keep "reasonable" bandwidth schedule
    return (cfg["b_in"] >= cfg["b_l1"] >= cfg["b_l2"] >= 1)


def cfg_name(cfg: dict) -> str:
    return (
        f"bin{cfg['b_in']}_b1{cfg['b_l1']}_b2{cfg['b_l2']}"
        f"_f1{cfg['f1']}_f2{cfg['f2']}"
        f"_seed{cfg['seed']}"
    )


def append_csv_row(csv_path: str, row: dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# -------------------------
# One experiment (aligned loop style)
# -------------------------
def train_one_config(
    *,
    cfg: dict,
    train_loader,
    test_loader,
    train_dataset,
    device,
    epochs: int,
    lr: float,
    phase_dir: str
):
    name = cfg_name(cfg)
    history_path = os.path.join(phase_dir, f"history_{name}.json")
    results_csv = os.path.join(phase_dir, "results.csv")

    if os.path.exists(history_path):
        print(f"[SKIP] {name} (history exists)")
        return

    set_seed(cfg["seed"])

    model = S2ConvNet_BWCh(
        b_in=cfg["b_in"],
        b_l1=cfg["b_l1"],
        b_l2=cfg["b_l2"],
        f1=cfg["f1"],
        f2=cfg["f2"],
        f_output=10
    ).to(device)

    n_params = count_params(model)
    print(f"\n[{os.path.basename(phase_dir)}] #params {n_params} | {name}")

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    t0_total = time.perf_counter()

    history = {
        "name": name,
        "config": cfg,
        "n_params": n_params,
        "epochs": []
    }

    best_acc = -1.0
    best_epoch = -1

    # run.py prints per-iter progress; we keep it (but a bit safer)
    iters_per_epoch = max(1, len(train_dataset) // cfg["batch_size"])

    for epoch in range(epochs):
        model.train()

        t0_epoch = time.perf_counter()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # run.py style progress
            print(
                "\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}".format(
                    epoch + 1, epochs, i + 1, iters_per_epoch, loss.item()
                ),
                end=""
            )

        print("")  # newline like run.py

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.perf_counter() - t0_epoch
        avg_train_loss = running_loss / max(len(train_loader), 1)

        acc = eval_accuracy_runpy_style(model, test_loader, device)
        print("Test Accuracy: {0}".format(100.0 * acc))

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1

        history["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "test_acc": acc,
            "epoch_time_sec": epoch_time
        })

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated(device)
    else:
        peak_mem = None

    total_time = time.perf_counter() - t0_total
    avg_epoch_time = float(np.mean([e["epoch_time_sec"] for e in history["epochs"]])) if history["epochs"] else None
    final_acc = history["epochs"][-1]["test_acc"] if history["epochs"] else None

    history["summary"] = {
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "final_test_acc": final_acc,
        "total_time_sec": total_time,
        "avg_epoch_time_sec": avg_epoch_time,
        "peak_gpu_mem_bytes": peak_mem
    }

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    row = {
        "name": name,
        "seed": cfg["seed"],
        "b_in": cfg["b_in"],
        "b_l1": cfg["b_l1"],
        "b_l2": cfg["b_l2"],
        "f1": cfg["f1"],
        "f2": cfg["f2"],
        "n_params": n_params,
        "epochs": epochs,
        "lr": lr,
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "final_test_acc": final_acc,
        "total_time_sec": total_time,
        "avg_epoch_time_sec": avg_epoch_time if avg_epoch_time is not None else "",
        "peak_gpu_mem_bytes": peak_mem if peak_mem is not None else "",
        "history_json": os.path.basename(history_path)
    }
    append_csv_row(results_csv, row)


# -------------------------
# Phase design (as we planned)
# -------------------------
def build_phases(include_optional: bool):
    b_in = 30

    # Baseline anchor (run.py original):
    # (b_in, b_l1, b_l2, f1, f2) = (30,10,6,20,40)
    baseline = [
        (b_in, 10, 6, 20, 40),
    ]

    # Phase A: bandwidth sweep, channels fixed at baseline (f1=20,f2=40)
    phaseA = [
        (b_in,  2, 1, 20, 40),
        (b_in, 6, 4, 20, 40),
        (b_in, 18, 16, 20, 40),
        (b_in, 24, 22, 20, 40),
        (b_in, 26, 24, 20, 40),
    ]

    # Phase B: channel sweep, bandwidth fixed at baseline (b_l1=10,b_l2=6)
    phaseB = [
        (b_in, 10, 6, 10, 20),
        (b_in, 10, 6, 20, 20),
        (b_in, 10, 6, 30, 60),
        (b_in, 10, 6, 30, 80),
        (b_in, 10, 6, 40, 80)
    ]

    # Phase C: interaction contrasts
    phaseC = [
        (b_in,  6, 4, 30, 60),   # low bw + higher ch
        (b_in, 6, 4, 40, 80),
        (b_in, 24, 22, 10, 20),   # high bw + low ch
        (b_in, 26, 24, 10, 20),
        (b_in, 12, 8, 15, 30),   # moderate
    ]


    return {
        "baseline": baseline,
        "phaseA_bandwidth": phaseA,
        "phaseB_channels": phaseB,
        "phaseC_interaction": phaseC,
    }


def expand_with_seeds(base_list, seeds, batch_size):
    cfgs = []
    for (b_in, b1, b2, f1, f2), seed in product(base_list, seeds):
        cfg = {
            "b_in": b_in,
            "b_l1": b1,
            "b_l2": b2,
            "f1": f1,
            "f2": f2,
            "seed": seed,
            "batch_size": batch_size
        }
        if validate_cfg(cfg):
            cfgs.append(cfg)
    return cfgs


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="s2_mnist.gz")
    parser.add_argument("--results_root", type=str, default="bw_ch_sweep_results")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)

    parser.add_argument("--include_optional", action="store_true")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "baseline", "phaseA_bandwidth", "phaseB_channels", "phaseC_interaction"])

    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("DEVICE:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    ensure_dir(args.results_root)

    train_loader, test_loader, train_dataset, _ = load_data_runpy_aligned(
        args.data_path, args.batch_size
    )

    phases = build_phases(include_optional=args.include_optional)

    # Seeds:
    # baseline: 3 seeds as stability anchor; others: 1 seed quick coverage (same as before)
    baseline_seeds = [0]
    default_seeds = [0]

    for phase_name, base_cfgs in phases.items():
        if args.phase != "all" and phase_name != args.phase:
            continue

        phase_dir = os.path.join(args.results_root, phase_name)
        ensure_dir(phase_dir)

        seeds = baseline_seeds if phase_name == "baseline" else default_seeds
        cfgs = expand_with_seeds(base_cfgs, seeds, args.batch_size)

        print(f"\n========= Running phase: {phase_name} | configs={len(cfgs)} | seeds={seeds} =========")

        for cfg in cfgs:
            train_one_config(
                cfg=cfg,
                train_loader=train_loader,
                test_loader=test_loader,
                train_dataset=train_dataset,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                phase_dir=phase_dir
            )

    print("\nDone. Results in:", args.results_root)


if __name__ == "__main__":
    main()
