# pylint: disable=E1101,R,C
import argparse
import csv
import gzip
import os
import pickle
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils

from s2cnn import SO3Convolution, S2Convolution, so3_integrate
from s2cnn import so3_near_identity_grid, s2_near_identity_grid


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =========================================================
# Utils
# =========================================================
def cuda_sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def csv_write_header_if_needed(csv_path: str, fieldnames: List[str]):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def csv_append_row(csv_path: str, row: Dict, fieldnames: List[str]):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def safe_std_from_path(path: str) -> int:
    m = re.search(r"(train|test)_std_(\d+)", path)
    if m is None:
        raise ValueError(f"Cannot parse std from path: {path}")
    return int(m.group(2))


def list_test_files(test_root: str) -> List[Tuple[int, str]]:
    out = []
    for name in os.listdir(test_root):
        if name.startswith("test_std_"):
            std = int(name.split("_")[-1])
            fp = os.path.join(test_root, name, "s2_mnist_test.gz")
            if os.path.exists(fp):
                out.append((std, fp))
    out.sort(key=lambda x: x[0])
    return out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =========================================================
# Data loading
# =========================================================
def load_train_loader(train_path: str, batch_size: int):
    with gzip.open(train_path, "rb") as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(
        dataset["images"][:, None, :, :].astype(np.float32)
    )
    train_labels = torch.from_numpy(dataset["labels"].astype(np.int64))

    train_dataset = data_utils.TensorDataset(train_data, train_labels)
    train_loader = data_utils.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    return train_loader, train_dataset


def load_test_loader(test_path: str, batch_size: int):
    with gzip.open(test_path, "rb") as f:
        dataset = pickle.load(f)

    test_data = torch.from_numpy(
        dataset["images"][:, None, :, :].astype(np.float32)
    )
    test_labels = torch.from_numpy(dataset["labels"].astype(np.int64))

    test_dataset = data_utils.TensorDataset(test_data, test_labels)
    test_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return test_loader, test_dataset


# =========================================================
# Model: SO(3)-1 baseline
# =========================================================
class S2ConvNet_original(nn.Module):
    def __init__(self, act: str = "relu"):
        super().__init__()

        f1 = 20
        f2 = 40
        f_output = 10

        b_in = 30
        b_l1 = 10
        b_l2 = 6

        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()

        self.conv1 = S2Convolution(1, f1, b_in, b_l1, grid_s2)
        self.conv2 = SO3Convolution(f1, f2, b_l1, b_l2, grid_so3)

        self.act1 = self.get_activation(act)
        self.act2 = self.get_activation(act)

        self.out_layer = nn.Linear(f2, f_output)

    @staticmethod
    def get_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU(inplace=False)
        if name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=False)
        if name == "tanh":
            return nn.Tanh()
        if name == "sigmoid":
            return nn.Sigmoid()
        raise ValueError(f"Unknown activation: {name}")

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = so3_integrate(x)
        x = self.out_layer(x)
        return x


# =========================================================
# Eval
# =========================================================
def evaluate(model: nn.Module, loader, max_test_batches: int = 0) -> float:
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for j, (images, labels) in enumerate(loader, start=1):
            if max_test_batches > 0 and j > max_test_batches:
                break

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            pred = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (pred == labels).long().sum().item()

    return 100.0 * correct / max(1, total)


# =========================================================
# Train config
# =========================================================
@dataclass
class RunConfig:
    train_path: str
    train_std: int
    test_root: str
    results_dir: str

    epochs: int
    batch_size: int
    lr: float
    act: str

    max_train_batches: int
    max_test_batches: int

    save_model: bool


# =========================================================
# Main train + test-all
# =========================================================
def train_and_test_all(cfg: RunConfig):
    ensure_dir(cfg.results_dir)

    model = S2ConvNet_original(act=cfg.act).to(DEVICE)
    nparams = count_params(model)

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_loader, train_dataset = load_train_loader(cfg.train_path, cfg.batch_size)
    test_files = list_test_files(cfg.test_root)

    if len(test_files) == 0:
        raise RuntimeError(f"No test files found under: {cfg.test_root}")

    history_csv = os.path.join(
        cfg.results_dir,
        f"train_history_so3_1_trainstd_{cfg.train_std}_act_{cfg.act}.csv"
    )
    history_fields = [
        "train_std", "act", "epoch",
        "train_loss", "train_acc",
        "epoch_time_sec", "elapsed_total_sec",
        "params"
    ]
    csv_write_header_if_needed(history_csv, history_fields)

    eval_csv = os.path.join(
        cfg.results_dir,
        f"test_results_so3_1_trainstd_{cfg.train_std}_act_{cfg.act}.csv"
    )
    eval_fields = [
        "train_std", "test_std", "act",
        "epochs", "batch_size", "lr",
        "test_acc", "params"
    ]
    csv_write_header_if_needed(eval_csv, eval_fields)

    model_path = os.path.join(
        cfg.results_dir,
        f"so3_1_trainstd_{cfg.train_std}_act_{cfg.act}.pt"
    )

    print("=" * 90)
    print("Training SO(3)-1 baseline model")
    print(f"train_path   : {cfg.train_path}")
    print(f"test_root    : {cfg.test_root}")
    print(f"train_std    : {cfg.train_std}")
    print(f"results_dir  : {cfg.results_dir}")
    print(f"activation   : {cfg.act}")
    print(f"epochs       : {cfg.epochs}")
    print(f"batch_size   : {cfg.batch_size}")
    print(f"lr           : {cfg.lr}")
    print(f"device       : {DEVICE}")
    print(f"params       : {nparams}")
    print("=" * 90)

    cuda_sync()
    total_t0 = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        cuda_sync()
        epoch_t0 = time.perf_counter()

        running_loss = 0.0
        seen = 0
        correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(train_loader, start=1):
            if cfg.max_train_batches > 0 and i > cfg.max_train_batches:
                break

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            seen += bs

            pred = outputs.argmax(dim=1)
            total_train += bs
            correct_train += (pred == labels).long().sum().item()

        cuda_sync()
        epoch_time = time.perf_counter() - epoch_t0

        avg_loss = running_loss / max(1, seen)
        train_acc = 100.0 * correct_train / max(1, total_train)

        cuda_sync()
        elapsed_total = time.perf_counter() - total_t0

        print(
            f"Epoch {epoch:02d}/{cfg.epochs:02d} | "
            f"train_loss={avg_loss:.4f} | "
            f"train_acc={train_acc:.2f}% | "
            f"epoch_time={epoch_time:.2f}s"
        )

        csv_append_row(
            history_csv,
            {
                "train_std": cfg.train_std,
                "act": cfg.act,
                "epoch": epoch,
                "train_loss": float(f"{avg_loss:.8f}"),
                "train_acc": float(f"{train_acc:.8f}"),
                "epoch_time_sec": float(f"{epoch_time:.8f}"),
                "elapsed_total_sec": float(f"{elapsed_total:.8f}"),
                "params": nparams,
            },
            history_fields
        )

    if cfg.save_model:
        torch.save(model.state_dict(), model_path)
        print(f"\nSaved model to: {model_path}")

    print("\n" + "=" * 90)
    print("Evaluating on all test sets...")
    print("=" * 90)

    for test_std, test_path in test_files:
        test_loader, _ = load_test_loader(test_path, cfg.batch_size)
        test_acc = evaluate(model, test_loader, max_test_batches=cfg.max_test_batches)

        row = {
            "train_std": cfg.train_std,
            "test_std": test_std,
            "act": cfg.act,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "test_acc": float(f"{test_acc:.8f}"),
            "params": nparams,
        }

        print(f"[TEST] train_std={cfg.train_std:3d} -> test_std={test_std:3d} | acc={test_acc:.2f}%")

        csv_append_row(eval_csv, row, eval_fields)

    print("\n" + "=" * 90)
    print("Done.")
    print(f"Training history CSV : {history_csv}")
    print(f"Test results CSV     : {eval_csv}")
    if cfg.save_model:
        print(f"Model checkpoint     : {model_path}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_std",
        type=int,
        required=True,
        choices=[0, 25, 50, 75, 100],
        help="Training set noise std. Will map automatically to generated_data/train_std_X/s2_mnist_train.gz"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="generated_data",
        help="Root folder containing train_std_* and test_std_* folders"
    )
    parser.add_argument(
        "--test_root",
        type=str,
        default=None,
        help="Optional custom root containing test_std_* folders; default uses data_root"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Optional custom results directory; default becomes results_noise_so3_1_trainstd_X"
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--act", type=str, default="relu",
                        choices=["relu", "leaky_relu", "tanh", "sigmoid"])

    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_test_batches", type=int, default=0)

    parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()

    train_path = os.path.join(
        args.data_root,
        f"train_std_{args.train_std}",
        "s2_mnist_train.gz"
    )

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")

    test_root = args.test_root if args.test_root is not None else args.data_root

    results_dir = (
        args.results_dir
        if args.results_dir is not None
        else f"results_noise_so3_1_trainstd_{args.train_std}"
    )

    cfg = RunConfig(
        train_path=train_path,
        train_std=args.train_std,
        test_root=test_root,
        results_dir=results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        act=args.act,
        max_train_batches=args.max_train_batches,
        max_test_batches=args.max_test_batches,
        save_model=args.save_model,
    )

    train_and_test_all(cfg)


if __name__ == "__main__":
    main()