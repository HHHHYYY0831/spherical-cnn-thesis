# pylint: disable=E1101,R,C
import argparse
import gzip
import os
import pickle
import time
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils

from s2cnn import SO3Convolution, S2Convolution, so3_integrate
from s2cnn import so3_near_identity_grid, s2_near_identity_grid


MNIST_PATH = "s2_mnist.gz"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =========================================================
# Utils
# =========================================================
def cuda_sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_filename(s: str) -> str:
    return (
        str(s)
        .replace(" ", "")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
    )


def unique_path_if_exists(path: str) -> str:
    """Avoid overwriting by appending _run2/_run3..."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    k = 2
    while True:
        candidate = f"{base}_run{k}{ext}"
        if not os.path.exists(candidate):
            return candidate
        k += 1


def csv_write_header_if_needed(csv_path: str, fieldnames: List[str]):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            import csv
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()


def csv_append_row(csv_path: str, row: Dict, fieldnames: List[str]):
    with open(csv_path, "a", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)


def load_data(path: str, batch_size: int):
    with gzip.open(path, "rb") as f:
        dataset = pickle.load(f)

    train_data = torch.from_numpy(
        dataset["train"]["images"][:, None, :, :].astype(np.float32)
    )
    train_labels = torch.from_numpy(dataset["train"]["labels"].astype(np.int64))

    train_dataset = data_utils.TensorDataset(train_data, train_labels)
    train_loader = data_utils.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_data = torch.from_numpy(
        dataset["test"]["images"][:, None, :, :].astype(np.float32)
    )
    test_labels = torch.from_numpy(dataset["test"]["labels"].astype(np.int64))

    test_dataset = data_utils.TensorDataset(test_data, test_labels)
    test_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader, train_dataset, test_dataset


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_activation(name: str) -> nn.Module:
    """
    Keep only what you wanted:
    relu / leaky_relu / tanh / sigmoid
    """
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


# =========================================================
# Schedules (anchored endpoints)
# =========================================================
def geometric_int_schedule(start: int, end: int, intervals: int, mode: str) -> List[int]:
    """
    Return list length intervals+1, inclusive endpoints.
    Geometric interpolation + rounding, then enforce monotonicity.
    """
    assert intervals >= 1
    start = int(start)
    end = int(end)
    if start <= 0 or end <= 0:
        raise ValueError("start/end must be positive integers")

    ratio = (end / start) ** (1.0 / intervals)
    vals = [start]
    for i in range(1, intervals):
        v = int(round(start * (ratio ** i)))
        v = max(1, v)
        vals.append(v)
    vals.append(end)

    if mode == "decreasing":
        for i in range(1, len(vals)):
            if vals[i] > vals[i - 1]:
                vals[i] = vals[i - 1]
        vals[-1] = end
        for i in range(len(vals) - 1):
            if vals[i] < end:
                vals[i] = end
    elif mode == "increasing":
        for i in range(1, len(vals)):
            if vals[i] < vals[i - 1]:
                vals[i] = vals[i - 1]
        vals[-1] = end
        for i in range(len(vals) - 1):
            if vals[i] > end:
                vals[i] = end
    else:
        raise ValueError("mode must be 'decreasing' or 'increasing'")

    return [int(x) for x in vals]


def make_bandwidth_schedule(b0: int, b_end: int, L: int) -> List[int]:
    return geometric_int_schedule(b0, b_end, intervals=L, mode="decreasing")


def make_channel_schedule(f0: int, f_end: int, n_conv_layers: int) -> List[int]:
    return geometric_int_schedule(f0, f_end, intervals=n_conv_layers, mode="increasing")


# =========================================================
# Models
# =========================================================
class S2ConvNet_original(nn.Module):
    """
    Baseline: keep EXACT structure, but allow activation sweep (optional).
    """
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

        self.act1 = get_activation(act)
        self.act2 = get_activation(act)

        self.out_layer = nn.Linear(f2, f_output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = so3_integrate(x)
        x = self.out_layer(x)
        return x


class S2ConvNet_deep_var_anchored(nn.Module):
    """
    deep_var (anchored endpoints):
    - b0=30, b_end=b0//8
    - f0=1, f_end=64
    - only change SO3 depth L in {2..6}
    - FC head is FIXED (always expects 64)
    - activation is sweepable (optional)
    """
    def __init__(self, L: int, act: str = "relu", b0: int = 30, b_end: int = None, f0: int = 1, f_end: int = 64):
        super().__init__()

        if b_end is None:
            b_end = b0 // 8
        if L < 2 or L > 6:
            raise ValueError("L (num_so3) must be in {2,3,4,5,6}")
        if b_end < 1:
            raise ValueError("b_end too small")
        if f0 != 1:
            raise ValueError("Fix f0=1 for this ablation")
        if f_end != 64:
            raise ValueError("Fix f_end=64 for FC head")

        b_list = make_bandwidth_schedule(b0, b_end, L=L)               # length L+1
        f_list = make_channel_schedule(f0, f_end, n_conv_layers=1+L)   # length L+2

        self._b_list = b_list
        self._f_list = f_list
        self.num_so3 = L
        self.act_name = act

        act_fn = get_activation(act)

        # Keep grids fixed to reduce confounds
        grid_s2 = s2_near_identity_grid(n_alpha=6, max_beta=np.pi/16, n_beta=1)
        grid_so3 = so3_near_identity_grid(
            n_alpha=6, max_beta=np.pi/16, n_beta=1, max_gamma=2*np.pi, n_gamma=6
        )

        # S2Conv (b0 -> b0)
        self.s2 = S2Convolution(f_list[0], f_list[1], b0, b0, grid_s2)
        self.s2_act = act_fn

        # SO3 stack: L layers, (b_list[i-1] -> b_list[i])
        so3_layers: List[nn.Module] = []
        so3_acts: List[nn.Module] = []
        for i in range(1, L+1):
            cin = f_list[i]
            cout = f_list[i+1]
            bin_ = b_list[i-1]
            bout = b_list[i]
            so3_layers.append(SO3Convolution(cin, cout, bin_, bout, grid_so3))
            so3_acts.append(get_activation(act))  # new instance (safe)
        self.so3_layers = nn.ModuleList(so3_layers)
        self.so3_acts = nn.ModuleList(so3_acts)

        # Fixed FC head (same for all L), activation also sweepable if你想
        self.linear = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            get_activation(act),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            get_activation(act),
            nn.BatchNorm1d(32),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.s2(x)
        x = self.s2_act(x)
        for layer, act in zip(self.so3_layers, self.so3_acts):
            x = layer(x)
            x = act(x)
        x = so3_integrate(x)
        x = self.linear(x)
        return x


def describe_model(model: nn.Module):
    print("\n[MODEL ARCH CHECK]")
    if hasattr(model, "_b_list") and hasattr(model, "_f_list"):
        print(f"bandwidth schedule (SO3): {getattr(model, '_b_list')}")
        print(f"channel schedule (S2+SO3): {getattr(model, '_f_list')}")
        if hasattr(model, "act_name"):
            print(f"activation: {getattr(model, 'act_name')}")
    for name, m in model.named_modules():
        if isinstance(m, (S2Convolution, SO3Convolution)):
            cin = getattr(m, "nfeature_in", None)
            cout = getattr(m, "nfeature_out", None)
            b_in = getattr(m, "b_in", None)
            b_out = getattr(m, "b_out", None)
            grid = getattr(m, "grid", None)
            grid_len = len(grid) if grid is not None else None
            print(f"{name:28s} | {m.__class__.__name__:14s} Cin={cin} Cout={cout} b_in={b_in} b_out={b_out} grid_len={grid_len}")
    print("")


# =========================================================
# Training / Logging
# =========================================================
@dataclass
class RunConfig:
    network: str          # "original" or "deep_var"
    act: str              # activation name
    num_so3: int          # original=1, deep_var=L
    b0: int
    b_end: int
    f_end: int            # always 64 for deep_var
    epochs: int
    batch_size: int
    lr: float
    max_train_batches: int
    max_test_batches: int
    results_dir: str
    print_model: bool


def get_run_history_csv_path(cfg: RunConfig) -> str:
    layer_dir = os.path.join(cfg.results_dir, f"so3_{cfg.num_so3}")
    ensure_dir(layer_dir)

    if cfg.network == "original":
        fname = f"original_act-{cfg.act}_lr-{cfg.lr}_bs-{cfg.batch_size}_ep-{cfg.epochs}.csv"
    else:
        fname = (
            f"deep_var_act-{cfg.act}_L{cfg.num_so3}"
            f"_b0-{cfg.b0}_bend-{cfg.b_end}"
            f"_fend-{cfg.f_end}"
            f"_lr-{cfg.lr}_bs-{cfg.batch_size}_ep-{cfg.epochs}"
            ".csv"
        )

    fname = safe_filename(fname)
    return unique_path_if_exists(os.path.join(layer_dir, fname))


def summary_csv_path(results_dir: str) -> str:
    ensure_dir(results_dir)
    return os.path.join(results_dir, "summary.csv")


def append_summary(results_dir: str, row: Dict):
    out_csv = summary_csv_path(results_dir)
    fields = [
        "network", "act", "num_so3", "b0", "b_end", "f_end",
        "epochs", "batch_size", "lr",
        "params",
        "total_time_sec", "final_test_acc",
        "history_csv",
    ]
    csv_write_header_if_needed(out_csv, fields)
    csv_append_row(out_csv, row, fields)


def build_model(cfg: RunConfig) -> nn.Module:
    if cfg.network == "original":
        return S2ConvNet_original(act=cfg.act)
    if cfg.network == "deep_var":
        return S2ConvNet_deep_var_anchored(
            L=cfg.num_so3, act=cfg.act, b0=cfg.b0, b_end=cfg.b_end, f0=1, f_end=cfg.f_end
        )
    raise ValueError(f"Unknown network: {cfg.network}")


def train_one(cfg: RunConfig) -> Dict:
    train_loader, test_loader, train_dataset, _ = load_data(MNIST_PATH, cfg.batch_size)

    model = build_model(cfg).to(DEVICE)
    nparams = count_params(model)

    if cfg.print_model:
        describe_model(model)

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history_csv = get_run_history_csv_path(cfg)
    history_fields = [
        "network", "act", "num_so3", "b0", "b_end", "f_end", "batch_size", "lr", "params",
        "epoch",
        "train_loss", "train_acc",
        "test_acc",
        "epoch_time_sec",
        "elapsed_total_sec",
    ]
    csv_write_header_if_needed(history_csv, history_fields)

    cuda_sync()
    total_t0 = time.perf_counter()
    last_test_acc = None

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

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for j, (images, labels) in enumerate(test_loader, start=1):
                if cfg.max_test_batches > 0 and j > cfg.max_test_batches:
                    break
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                pred = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (pred == labels).long().sum().item()

        test_acc = 100.0 * correct / max(1, total)
        last_test_acc = test_acc

        cuda_sync()
        elapsed_total = time.perf_counter() - total_t0

        print(
            f"[{cfg.network}] act={cfg.act} so3={cfg.num_so3} "
            f"Epoch {epoch:02d}/{cfg.epochs:02d} "
            f"train_loss={avg_loss:.4f} train_acc={train_acc:.2f}% "
            f"test_acc={test_acc:.2f}% epoch_time={epoch_time:.2f}s"
        )

        csv_append_row(
            history_csv,
            {
                "network": cfg.network,
                "act": cfg.act,
                "num_so3": cfg.num_so3,
                "b0": cfg.b0,
                "b_end": cfg.b_end,
                "f_end": cfg.f_end,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "params": nparams,
                "epoch": epoch,
                "train_loss": float(f"{avg_loss:.8f}"),
                "train_acc": float(f"{train_acc:.8f}"),
                "test_acc": float(f"{test_acc:.8f}"),
                "epoch_time_sec": float(f"{epoch_time:.8f}"),
                "elapsed_total_sec": float(f"{elapsed_total:.8f}"),
            },
            history_fields,
        )

    cuda_sync()
    total_time = time.perf_counter() - total_t0

    return {
        "network": cfg.network,
        "act": cfg.act,
        "num_so3": cfg.num_so3,
        "b0": cfg.b0,
        "b_end": cfg.b_end,
        "f_end": cfg.f_end,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "params": nparams,
        "history_csv": history_csv,
        "total_time_sec": float(f"{total_time:.6f}"),
        "final_test_acc": float(f"{(last_test_acc if last_test_acc is not None else 0.0):.6f}"),
    }


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        default="sweep_depth",
        choices=["sweep_depth", "single"],
        help="sweep_depth: baseline(original) + deep_var(L=2..6); single: run one config",
    )

    # endpoints (fixed by protocol)
    parser.add_argument("--b0", type=int, default=30)
    parser.add_argument("--f_end", type=int, default=64)

    # training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)

    # results
    parser.add_argument("--results_dir", type=str, default="results_depth_anchored")

    # depth sweep
    parser.add_argument("--so3_list", type=int, nargs="+", default=[2, 3, 4, 5, 6])

    # activation sweep
    parser.add_argument("--sweep_activation", action="store_true",
                        help="if set: sweep act_list; otherwise only use --act (default relu)")
    parser.add_argument("--act_list", type=str, nargs="+", default=["relu", "leaky_relu", "tanh", "sigmoid"])
    parser.add_argument("--act", type=str, default="relu")

    # speed debug
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_test_batches", type=int, default=0)

    # print model structure
    parser.add_argument("--print_model", action="store_true")

    # single options
    parser.add_argument("--network", default="deep_var", choices=["original", "deep_var"])
    parser.add_argument("--num_so3", type=int, default=2)

    # whether baseline also sweeps activation (default: False -> baseline fixed relu)
    parser.add_argument("--baseline_sweep_activation", action="store_true",
                        help="also sweep activation for baseline original (optional)")

    args = parser.parse_args()
    ensure_dir(args.results_dir)

    b_end = args.b0 // 8

    if args.mode == "single":
        cfg = RunConfig(
            network=args.network,
            act=args.act,
            num_so3=(1 if args.network == "original" else args.num_so3),
            b0=args.b0,
            b_end=b_end,
            f_end=args.f_end,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_train_batches=args.max_train_batches,
            max_test_batches=args.max_test_batches,
            results_dir=args.results_dir,
            print_model=args.print_model,
        )
        res = train_one(cfg)
        append_summary(args.results_dir, res)
        print(f"[DONE] history -> {res['history_csv']}")
        print(f"[DONE] summary -> {summary_csv_path(args.results_dir)}")
        return

    # decide which activations to run
    acts = args.act_list if args.sweep_activation else [args.act]

    # 1) baseline original (so3=1)
    base_acts = acts if args.baseline_sweep_activation else ["relu"]
    for act in base_acts:
        cfg0 = RunConfig(
            network="original",
            act=act,
            num_so3=1,
            b0=args.b0,
            b_end=b_end,
            f_end=args.f_end,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_train_batches=args.max_train_batches,
            max_test_batches=args.max_test_batches,
            results_dir=args.results_dir,
            print_model=args.print_model,
        )
        print("\n" + "=" * 90)
        print(f"[SWEEP] baseline original (SO3=1) act={act}")
        print("=" * 90)
        res0 = train_one(cfg0)
        append_summary(args.results_dir, res0)
        print(f"[SAVED] {res0['history_csv']}")

    # 2) deep_var L=2..6 with anchored schedules
    for L in args.so3_list:
        for act in acts:
            cfg = RunConfig(
                network="deep_var",
                act=act,
                num_so3=L,
                b0=args.b0,
                b_end=b_end,
                f_end=args.f_end,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_train_batches=args.max_train_batches,
                max_test_batches=args.max_test_batches,
                results_dir=args.results_dir,
                print_model=args.print_model,
            )
            print("\n" + "=" * 90)
            print(f"[SWEEP] deep_var SO3={L} act={act} (b0={args.b0}, b_end={b_end}, f_end={args.f_end})")
            print("=" * 90)
            res = train_one(cfg)
            append_summary(args.results_dir, res)
            print(f"[SAVED] {res['history_csv']}")

    print(f"\n[SWEEP DONE] summary -> {summary_csv_path(args.results_dir)}")
    print(f"[SWEEP DONE] logs -> {os.path.abspath(args.results_dir)}/so3_*/*.csv")


if __name__ == "__main__":
    main()
