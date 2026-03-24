import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ====== 全局字体设置（按你的习惯） ======
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

# ====== 路径设置 ======
ROOT = Path(".")

RESULT_DIRS = [
    ROOT / "results_noise_so3_1_trainstd_0",
    ROOT / "results_noise_so3_1_trainstd_25",
    ROOT / "results_noise_so3_1_trainstd_50",
    ROOT / "results_noise_so3_1_trainstd_75",
    ROOT / "results_noise_so3_1_trainstd_100",
]

OUTFILE = "noise_generalization_curves_so3_1.png"


def find_test_csv(result_dir: Path) -> Path:
    files = list(result_dir.glob("test_results_so3_1_trainstd_*_act_*.csv"))
    if len(files) == 0:
        raise FileNotFoundError(f"No test result CSV found in {result_dir}")
    if len(files) > 1:
        print(f"Warning: multiple CSVs found in {result_dir}, using {files[0].name}")
    return files[0]


# ====== 读取并合并所有结果 ======
dfs = []
for result_dir in RESULT_DIRS:
    csv_path = find_test_csv(result_dir)
    df = pd.read_csv(csv_path)
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

# 保证按数值排序
all_df["train_std"] = all_df["train_std"].astype(int)
all_df["test_std"] = all_df["test_std"].astype(int)
all_df["test_acc"] = all_df["test_acc"].astype(float)

# ====== 作图 ======
plt.figure(figsize=(14, 6))

for train_std in sorted(all_df["train_std"].unique()):
    sub = all_df[all_df["train_std"] == train_std].sort_values("test_std")
    plt.plot(
        sub["test_std"],
        sub["test_acc"],
        marker="o",
        linewidth=2,
        label=f"train noise std={train_std}"
    )

plt.xlabel("Test noise std")
plt.ylabel("Test accuracy (%)")
plt.title("SO(3)-1 model: generalization across test noise levels")
plt.xlim(0, 100)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(OUTFILE, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {OUTFILE}")