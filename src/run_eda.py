"""Enhanced EDA script — saves plots and summary stats to reports/"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import load_data, preprocess

# === Output directory ===
OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
os.makedirs(OUT, exist_ok=True)


def run():
    print("=== Running Enhanced EDA ===")

    # === Load & preprocess data ===
    df = load_data()
    df = preprocess(df)

    # --- Basic info ---
    info_path = os.path.join(OUT, "info.txt")
    with open(info_path, "w") as f:
        df.info(buf=f)
    df.describe().to_csv(os.path.join(OUT, "describe.csv"))
    print("✅ Saved info.txt & describe.csv")

    # --- Missing data ---
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Data Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "missing_data_heatmap.png"))
    plt.close()

    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if not missing.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=missing.values, y=missing.index)
        plt.title("Missing Values per Column")
        plt.xlabel("Count")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "missing_data_bar.png"))
        plt.close()

    # --- Target distribution ---
    plt.figure(figsize=(6, 4))
    df["tsunami"].value_counts().sort_index().plot(kind="bar", color=["skyblue", "salmon"])
    plt.title("Tsunami Distribution")
    plt.xlabel("Tsunami (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "tsunami_distribution.png"))
    plt.close()

    # --- Correlation matrix ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "correlation_matrix.png"))
    plt.close()

    # --- Numeric distributions ---
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"dist_{col}.png"))
        plt.close()

    # --- Boxplots by target ---
    if "tsunami" in df.columns:
        for col in num_cols:
            if col != "tsunami":
                plt.figure(figsize=(6, 4))
                sns.boxplot(data=df, x="tsunami", y=col, hue="tsunami", palette="coolwarm", legend=False)
                plt.title(f"{col} vs Tsunami")
                plt.tight_layout()
                plt.savefig(os.path.join(OUT, f"boxplot_{col}.png"))
                plt.close()

    # --- Pairplot (sample to reduce file size) ---
    sample_df = df.sample(min(len(df), 500), random_state=42)
    sns.pairplot(sample_df[num_cols], corner=True, diag_kind="kde")
    plt.savefig(os.path.join(OUT, "pairplot_sample.png"))
    plt.close()

    print(f"✅ EDA complete — all plots saved in: {OUT}")


if __name__ == "__main__":
    run()
