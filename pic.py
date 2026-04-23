import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results_all/raw_results_all.csv")

os.makedirs("figures", exist_ok=True)

metrics = ["MAPE", "MAE", "RMSE"]

for metric in metrics:
    lr = df[df["model"] == "LinearRegression"][metric]
    rf = df[df["model"] == "RandomForest"][metric]

    plt.figure(figsize=(6,4))

    plt.boxplot(
        [lr, rf],
        labels=["Linear Regression", "Random Forest"]
    )

    plt.title(f"{metric} Comparison")
    plt.ylabel(metric)

    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"figures/{metric.lower()}_boxplot.png", dpi=300)
    plt.close()

    print(f"Saved figures/{metric.lower()}_boxplot.png")