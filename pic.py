import pandas as pd
import matplotlib.pyplot as plt
import os

# 讀你的結果
df = pd.read_csv("results_all/raw_results_all.csv")

os.makedirs("figures", exist_ok=True)

metrics = ["MAPE", "MAE", "RMSE"]

for metric in metrics:
    lr = df[df["model"] == "LinearRegression"][metric]
    rf = df[df["model"] == "RandomForest"][metric]

    plt.figure(figsize=(7,5))

    plt.boxplot(
        [lr, rf],
        labels=["Linear Regression", "Random Forest"],
        showfliers=True,
        flierprops=dict(marker='o', markersize=2, alpha=0.3),  # ⭐ 小而透明
        medianprops=dict(linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2)
    )

    # ⭐ 核心：log scale
    plt.yscale("log")

    plt.title(f"{metric} Comparison (Log Scale)")
    plt.ylabel(metric)
    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"figures/{metric.lower()}_boxplot.png", dpi=300)
    plt.close()

    print(f"Saved figures/{metric.lower()}_boxplot.png")