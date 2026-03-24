import csv
from pathlib import Path
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

ROOT = Path(".")

RESULT_DIRS = [
    ROOT / "results_noise_so3_6_trainstd_0",
    ROOT / "results_noise_so3_6_trainstd_25",
    ROOT / "results_noise_so3_6_trainstd_50",
    ROOT / "results_noise_so3_6_trainstd_75",
    ROOT / "results_noise_so3_6_trainstd_100",
]

OUTFILE = "noise_generalization_curves_so3_6.png"


def find_test_csv(result_dir: Path) -> Path:
    files = list(result_dir.glob("test_results_so3_6_trainstd_*_act_*.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No test result CSV found in {result_dir}")
    if len(files) > 1:
        print(f"Warning: multiple CSVs found in {result_dir}, using {files[0].name}")
    return files[0]


grouped = {}

for result_dir in RESULT_DIRS:
    csv_path = find_test_csv(result_dir)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_std = int(row["train_std"])
            test_std = int(row["test_std"])
            test_acc = float(row["test_acc"])

            if train_std not in grouped:
                grouped[train_std] = []

            grouped[train_std].append((test_std, test_acc))

plt.figure(figsize=(14, 6))

for train_std in sorted(grouped.keys()):
    points = sorted(grouped[train_std], key=lambda x: x[0])
    x = [p[0] for p in points]
    y = [p[1] for p in points]

    plt.plot(
        x, y,
        marker="o",
        linewidth=2,
        label=f"train noise std={train_std}"
    )

plt.xlabel("Test noise std")
plt.ylabel("Test accuracy (%)")
plt.title("SO(3)-6 model: generalization across test noise levels")
plt.xlim(0, 100)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(OUTFILE, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {OUTFILE}")
