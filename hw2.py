import os
import math
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error
)
from scipy.stats import wilcoxon

ROOT_DATA_DIR = "datasets"
TARGET_COL = "time"
N_RUNS = 30
TEST_SIZE = 0.30
RESULT_DIR = "results_all"

os.makedirs(RESULT_DIR, exist_ok=True)


def rmse_score(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def find_all_csv_files(root_dir):
    csv_files = []
    for current_root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(current_root, f))
    csv_files.sort()
    return csv_files


def load_dataset(path, target_col):
    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {path}")

    if df.isnull().sum().sum() > 0:
        raise ValueError(f"Missing values found in {path}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=False)

    return df, X, y


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "MAPE": mean_absolute_percentage_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": rmse_score(y_test, y_pred)
    }


def summarise(values):
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }


def safe_wilcoxon(a, b):
    try:
        stat, p = wilcoxon(a, b, alternative="greater")
        return float(stat), float(p)
    except ValueError:
        return None, None


def run_one_dataset(dataset_path):
    raw_df, X, y = load_dataset(dataset_path, TARGET_COL)

    rows = []
    for seed in range(N_RUNS):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed
        )

        lr = LinearRegression()
        lr_res = evaluate_model(lr, X_train, X_test, y_train, y_test)
        rows.append({
            "dataset": dataset_path,
            "run": seed + 1,
            "model": "LinearRegression",
            **lr_res
        })

        rf = RandomForestRegressor(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1
        )
        rf_res = evaluate_model(rf, X_train, X_test, y_train, y_test)
        rows.append({
            "dataset": dataset_path,
            "run": seed + 1,
            "model": "RandomForest",
            **rf_res
        })

    results_df = pd.DataFrame(rows)

    summary_rows = []
    for model_name in ["LinearRegression", "RandomForest"]:
        sub = results_df[results_df["model"] == model_name]
        for metric in ["MAPE", "MAE", "RMSE"]:
            s = summarise(sub[metric].values)
            summary_rows.append({
                "dataset": dataset_path,
                "model": model_name,
                "metric": metric,
                **s
            })

    summary_df = pd.DataFrame(summary_rows)

    stat_rows = []
    for metric in ["MAPE", "MAE", "RMSE"]:
        lr_vals = results_df[results_df["model"] == "LinearRegression"][metric].values
        rf_vals = results_df[results_df["model"] == "RandomForest"][metric].values
        stat, p = safe_wilcoxon(lr_vals, rf_vals)
        stat_rows.append({
            "dataset": dataset_path,
            "metric": metric,
            "lr_mean": float(np.mean(lr_vals)),
            "rf_mean": float(np.mean(rf_vals)),
            "wilcoxon_stat": stat,
            "p_value": p,
            "significant_at_0_05": (p is not None and p < 0.05)
        })

    stats_df = pd.DataFrame(stat_rows)
    return raw_df, results_df, summary_df, stats_df


def main():
    all_csvs = find_all_csv_files(ROOT_DATA_DIR)
    print(f"Found {len(all_csvs)} CSV datasets.")

    all_raw_results = []
    all_summary = []
    all_stats = []

    for i, dataset_path in enumerate(all_csvs, start=1):
        print(f"\n[{i}/{len(all_csvs)}] Processing: {dataset_path}")
        try:
            _, raw_results_df, summary_df, stats_df = run_one_dataset(dataset_path)
            all_raw_results.append(raw_results_df)
            all_summary.append(summary_df)
            all_stats.append(stats_df)
            print("Done.")
        except Exception as e:
            print(f"Skipped بسبب error: {e}")

    if not all_raw_results:
        print("No dataset processed successfully.")
        return

    raw_results_all = pd.concat(all_raw_results, ignore_index=True)
    summary_all = pd.concat(all_summary, ignore_index=True)
    stats_all = pd.concat(all_stats, ignore_index=True)

    raw_results_all.to_csv(os.path.join(RESULT_DIR, "raw_results_all.csv"), index=False)
    summary_all.to_csv(os.path.join(RESULT_DIR, "summary_all.csv"), index=False)
    stats_all.to_csv(os.path.join(RESULT_DIR, "wilcoxon_all.csv"), index=False)

    # 額外做一份「每個 dataset 哪個方法比較好」的簡表
    comparison_rows = []
    for dataset_name in stats_all["dataset"].unique():
        sub = stats_all[stats_all["dataset"] == dataset_name]
        better_count = int((sub["rf_mean"] < sub["lr_mean"]).sum())
        sig_count = int(sub["significant_at_0_05"].sum())
        comparison_rows.append({
            "dataset": dataset_name,
            "rf_better_metrics_count": better_count,
            "significant_metrics_count": sig_count
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(RESULT_DIR, "comparison_overview.csv"), index=False)

    print("\nSaved:")
    print("- results_all/raw_results_all.csv")
    print("- results_all/summary_all.csv")
    print("- results_all/wilcoxon_all.csv")
    print("- results_all/comparison_overview.csv")


if __name__ == "__main__":
    main()