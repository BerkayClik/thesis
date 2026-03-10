"""
Feature Selection for LunarCrush BTC 4-Hour Data.

Runs multiple feature selection methods to identify the optimal 4-feature
combination for Quaternion Neural Network models, targeting next-period
close price prediction.

Methods:
  - Filter:   Pearson correlation, Spearman correlation, Mutual Information,
               Variance Threshold
  - Embedded:  Random Forest importance, LASSO (L1) coefficients
  - Temporal:  Granger Causality, Lagged Cross-Correlation
  - Ensemble:  Aggregated ranking across all methods

Usage (from project root):
    python scripts/feature_selection.py
    python scripts/feature_selection.py --data data/cache/lunarcrush_btc_4hour_full.csv
    python scripts/feature_selection.py --top-k 4 --max-lag 6

Outputs:
    - Console: ranked feature tables per method + final recommendation
    - Figures: saved to scripts/feature_selection_outputs/
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── project root for imports ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.lunarcrush_api import LUNARCRUSH_ALL_COLUMNS

# ── defaults ──────────────────────────────────────────────────────────────
DEFAULT_CSV = PROJECT_ROOT / "data" / "cache" / "lunarcrush_btc_4hour_full.csv"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "feature_selection_outputs"
TARGET_COL = "close"
TRAIN_RATIO = 0.70
TOP_K = 4
MAX_GRANGER_LAG = 6   # max lag for Granger causality (in 4-hour steps)
MAX_XCORR_LAG = 12    # max lag for cross-correlation
RANDOM_STATE = 42


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

def load_and_prepare(csv_path: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Load CSV, handle missing values, create target (next-period close),
    return (features_df, target_series, feature_names).

    Uses only the training portion (first 70%) to avoid look-ahead bias.
    """
    df = pd.read_csv(csv_path, parse_dates=["Datetime"], index_col="Datetime")
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns from {csv_path}")
    print(f"Date range: {df.index.min()} -> {df.index.max()}")

    # ── missing values ────────────────────────────────────────────────
    missing = df.isnull().sum()
    if missing.any():
        print(f"\nMissing values:\n{missing[missing > 0]}")
        # Forward-fill then drop remaining NaN rows (start of series)
        df = df.ffill().dropna()
        print(f"After ffill + dropna: {len(df)} rows")

    # ── target: next-period close ─────────────────────────────────────
    target = df[TARGET_COL].shift(-1)  # predict next bar's close
    # Drop last row (no target available)
    df = df.iloc[:-1]
    target = target.iloc[:-1]

    # ── temporal split: use only training portion ─────────────────────
    train_end = int(len(df) * TRAIN_RATIO)
    df_train = df.iloc[:train_end].copy()
    target_train = target.iloc[:train_end].copy()

    print(f"Training set: {len(df_train)} rows "
          f"({df_train.index.min()} -> {df_train.index.max()})")

    feature_names = list(df_train.columns)

    return df_train, target_train, feature_names


# ═══════════════════════════════════════════════════════════════════════════
# 2. FILTER METHODS
# ═══════════════════════════════════════════════════════════════════════════

def pearson_correlation(df: pd.DataFrame, target: pd.Series) -> pd.Series:
    """Pearson (linear) correlation of each feature with target."""
    corrs = df.corrwith(target, method="pearson").abs()
    return corrs.sort_values(ascending=False).rename("pearson_abs")


def spearman_correlation(df: pd.DataFrame, target: pd.Series) -> pd.Series:
    """Spearman (rank) correlation -- captures monotonic non-linear relationships."""
    corrs = df.corrwith(target, method="spearman").abs()
    return corrs.sort_values(ascending=False).rename("spearman_abs")


def mutual_information(df: pd.DataFrame, target: pd.Series,
                       random_state: int = RANDOM_STATE) -> pd.Series:
    """Mutual Information regression -- captures arbitrary non-linear dependencies."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    mi = mutual_info_regression(
        X_scaled, target,
        n_neighbors=5,
        random_state=random_state,
    )
    return pd.Series(mi, index=df.columns, name="mutual_info").sort_values(ascending=False)


def variance_analysis(df: pd.DataFrame) -> pd.Series:
    """
    Variance of each feature (after z-score scaling).
    Low variance -> feature carries little information.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    variances = pd.Series(
        X_scaled.var(axis=0), index=df.columns, name="scaled_variance"
    ).sort_values(ascending=False)
    return variances


# ═══════════════════════════════════════════════════════════════════════════
# 3. EMBEDDED METHODS
# ═══════════════════════════════════════════════════════════════════════════

def random_forest_importance(df: pd.DataFrame, target: pd.Series,
                             random_state: int = RANDOM_STATE) -> pd.Series:
    """Feature importance from a Random Forest regressor."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_scaled, target)
    importances = pd.Series(
        rf.feature_importances_, index=df.columns, name="rf_importance"
    ).sort_values(ascending=False)
    return importances


def lasso_importance(df: pd.DataFrame, target: pd.Series,
                     random_state: int = RANDOM_STATE) -> pd.Series:
    """
    Absolute LASSO coefficients -- L1 regularisation naturally zeros out
    irrelevant features.  Uses LassoCV for automatic alpha selection.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    target_scaled = (target - target.mean()) / (target.std() + 1e-8)
    lasso = LassoCV(
        cv=5,
        random_state=random_state,
        max_iter=5000,
    )
    lasso.fit(X_scaled, target_scaled)
    coefs = pd.Series(
        np.abs(lasso.coef_), index=df.columns, name="lasso_abs_coef"
    ).sort_values(ascending=False)
    print(f"  LASSO best alpha = {lasso.alpha_:.6f}")
    return coefs


# ═══════════════════════════════════════════════════════════════════════════
# 4. TIME-SERIES METHODS
# ═══════════════════════════════════════════════════════════════════════════

def granger_causality(df: pd.DataFrame, target_name: str = TARGET_COL,
                      max_lag: int = MAX_GRANGER_LAG) -> pd.Series:
    """
    Test whether each feature Granger-causes the target.
    Returns min p-value across lags (lower -> more predictive).
    We invert to a score: score = 1 - min_pvalue.
    """
    results = {}
    target_series = df[target_name].values
    for col in df.columns:
        if col == target_name:
            results[col] = 1.0  # self-causality: assign max score
            continue
        try:
            test_data = np.column_stack([target_series, df[col].values])
            mask = ~np.isnan(test_data).any(axis=1)
            test_data = test_data[mask]
            gc = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
            min_p = min(gc[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1))
            results[col] = 1.0 - min_p
        except Exception as e:
            print(f"  Granger test failed for {col}: {e}")
            results[col] = 0.0
    return pd.Series(results, name="granger_score").sort_values(ascending=False)


def lagged_cross_correlation(df: pd.DataFrame, target: pd.Series,
                             max_lag: int = MAX_XCORR_LAG) -> pd.Series:
    """
    Max absolute cross-correlation between each feature and the target
    across lags 1..max_lag.  Measures temporal predictive power.
    """
    results = {}
    for col in df.columns:
        max_corr = 0.0
        for lag in range(1, max_lag + 1):
            feature_lagged = df[col].shift(lag).dropna()
            target_aligned = target.loc[feature_lagged.index]
            if len(feature_lagged) < 50:
                continue
            corr = np.abs(feature_lagged.corr(target_aligned))
            if corr > max_corr:
                max_corr = corr
        results[col] = max_corr
    return pd.Series(results, name="max_lagged_xcorr").sort_values(ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
# 5. ENSEMBLE RANKING
# ═══════════════════════════════════════════════════════════════════════════

def build_ensemble(method_results: dict[str, pd.Series],
                   feature_names: list[str]) -> pd.DataFrame:
    """
    Combine rankings from all methods into a single score.
    Each method's scores are min-max normalised to [0, 1], then averaged.
    """
    ranking_df = pd.DataFrame(index=feature_names)

    for name, scores in method_results.items():
        aligned = scores.reindex(feature_names).fillna(0.0)
        smin, smax = aligned.min(), aligned.max()
        if smax - smin > 1e-10:
            normalised = (aligned - smin) / (smax - smin)
        else:
            normalised = aligned * 0.0
        ranking_df[name] = normalised

    ranking_df["ensemble_score"] = ranking_df.mean(axis=1)
    ranking_df = ranking_df.sort_values("ensemble_score", ascending=False)
    ranking_df["rank"] = range(1, len(ranking_df) + 1)
    return ranking_df


# ═══════════════════════════════════════════════════════════════════════════
# 6. VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path):
    """Full feature-feature correlation heatmap (Pearson)."""
    corr = df.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix (Pearson)", fontsize=14, pad=15)
    plt.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'correlation_heatmap.png'}")


def plot_multi_method_comparison(ranking_df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart comparing normalised scores across methods."""
    method_cols = [c for c in ranking_df.columns if c not in ("ensemble_score", "rank")]
    plot_df = ranking_df[method_cols]

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_df.plot(kind="bar", ax=ax, width=0.85, edgecolor="white", linewidth=0.5)
    ax.set_title("Feature Importance -- Multi-Method Comparison (normalised)", fontsize=14)
    ax.set_ylabel("Normalised Score [0, 1]", fontsize=12)
    ax.set_xlabel("Feature", fontsize=12)
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(output_dir / "method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'method_comparison.png'}")


def plot_ensemble_ranking(ranking_df: pd.DataFrame, top_k: int, output_dir: Path):
    """Horizontal bar chart of ensemble scores with top-K highlighted."""
    fig, ax = plt.subplots(figsize=(10, 8))
    scores = ranking_df["ensemble_score"].sort_values(ascending=True)
    colors = ["#2ecc71" if feat in ranking_df.head(top_k).index else "#95a5a6"
              for feat in scores.index]
    scores.plot(kind="barh", ax=ax, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Ensemble Feature Ranking (top {top_k} highlighted)", fontsize=14)
    ax.set_xlabel("Ensemble Score (mean of normalised method scores)", fontsize=11)
    ax.axvline(x=scores.iloc[-top_k], color="red", linestyle="--", alpha=0.5,
               label=f"Top-{top_k} threshold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(output_dir / "ensemble_ranking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'ensemble_ranking.png'}")


def plot_feature_correlation_with_target(df: pd.DataFrame, target: pd.Series,
                                         output_dir: Path):
    """Bar chart: Pearson + Spearman correlation of each feature with target."""
    pearson = df.corrwith(target, method="pearson")
    spearman = df.corrwith(target, method="spearman")
    corr_df = pd.DataFrame({"Pearson": pearson, "Spearman": spearman})
    corr_df = corr_df.reindex(corr_df["Pearson"].abs().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(corr_df))
    width = 0.35
    ax.bar(x - width / 2, corr_df["Pearson"], width, label="Pearson", color="#3498db")
    ax.bar(x + width / 2, corr_df["Spearman"], width, label="Spearman", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(corr_df.index, rotation=45, ha="right")
    ax.set_ylabel("Correlation with next-period close")
    ax.set_title("Feature Correlation with Target (next-period close)", fontsize=14)
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(output_dir / "target_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'target_correlation.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Feature selection for LunarCrush BTC data")
    parser.add_argument("--data", type=str, default=str(DEFAULT_CSV),
                        help="Path to CSV file")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="Number of features to select (default: 4)")
    parser.add_argument("--max-lag", type=int, default=MAX_GRANGER_LAG,
                        help="Max lag for Granger causality test")
    args = parser.parse_args()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & prepare ─────────────────────────────────────────────
    print("=" * 70)
    print("PHASE 1: Data Loading & Preparation")
    print("=" * 70)
    df_train, target_train, feature_names = load_and_prepare(args.data)

    # ── 2. Filter methods ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: Filter Methods")
    print("=" * 70)

    print("\n[1] Pearson Correlation with target:")
    pearson = pearson_correlation(df_train, target_train)
    print(pearson.to_string())

    print("\n[2] Spearman Correlation with target:")
    spearman = spearman_correlation(df_train, target_train)
    print(spearman.to_string())

    print("\n[3] Mutual Information Regression:")
    mi = mutual_information(df_train, target_train)
    print(mi.to_string())

    print("\n[4] Scaled Variance:")
    var = variance_analysis(df_train)
    print(var.to_string())

    # ── 3. Embedded methods ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: Embedded Methods")
    print("=" * 70)

    print("\n[5] Random Forest Feature Importance:")
    rf_imp = random_forest_importance(df_train, target_train)
    print(rf_imp.to_string())

    print("\n[6] LASSO (L1) Absolute Coefficients:")
    lasso_imp = lasso_importance(df_train, target_train)
    print(lasso_imp.to_string())

    # ── 4. Time-series methods ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 4: Time-Series Methods")
    print("=" * 70)

    print(f"\n[7] Granger Causality (max lag = {args.max_lag}):")
    granger = granger_causality(df_train, TARGET_COL, max_lag=args.max_lag)
    print(granger.to_string())

    print(f"\n[8] Lagged Cross-Correlation (max lag = {MAX_XCORR_LAG}):")
    xcorr = lagged_cross_correlation(df_train, target_train, max_lag=MAX_XCORR_LAG)
    print(xcorr.to_string())

    # ── 5. Ensemble ranking ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 5: Ensemble Ranking")
    print("=" * 70)

    method_results = {
        "pearson": pearson,
        "spearman": spearman,
        "mutual_info": mi,
        "variance": var,
        "rf_importance": rf_imp,
        "lasso": lasso_imp,
        "granger": granger,
        "lagged_xcorr": xcorr,
    }

    ranking_df = build_ensemble(method_results, feature_names)

    print("\nFull Ensemble Ranking Table:")
    print(ranking_df.to_string(float_format="%.4f"))

    top_features = ranking_df.head(args.top_k)
    print(f"\n{'=' * 70}")
    print(f"RECOMMENDATION: Top {args.top_k} Features for QNN")
    print(f"{'=' * 70}")
    for i, (feat, row) in enumerate(top_features.iterrows(), 1):
        col_idx = LUNARCRUSH_ALL_COLUMNS.index(feat)
        print(f"  {i}. {feat:<22s} (column index {col_idx:>2d})  "
              f"ensemble = {row['ensemble_score']:.4f}")

    # Column indices for YAML config
    top_indices = [LUNARCRUSH_ALL_COLUMNS.index(f) for f in top_features.index]
    print(f"\nYAML config:  feature_cols: {sorted(top_indices)}")

    # Identify the target_col position within the selected features
    if TARGET_COL in top_features.index:
        sorted_feats = sorted(
            zip(top_indices, top_features.index), key=lambda x: x[0]
        )
        target_pos_sorted = [f for _, f in sorted_feats].index(TARGET_COL)
        print(f"              target_col: {target_pos_sorted}  "
              f"('{TARGET_COL}' position in sorted feature_cols)")
    else:
        print(f"\n  NOTE: '{TARGET_COL}' not in top-{args.top_k}. "
              "Consider including it manually as the prediction target.")

    # ── 6. Visualisations ─────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("PHASE 6: Generating Visualisations")
    print(f"{'=' * 70}")

    plot_correlation_heatmap(df_train, output_dir)
    plot_feature_correlation_with_target(df_train, target_train, output_dir)
    plot_multi_method_comparison(ranking_df, output_dir)
    plot_ensemble_ranking(ranking_df, args.top_k, output_dir)

    # ── Save ranking table to CSV ─────────────────────────────────────
    ranking_csv = output_dir / "feature_ranking.csv"
    ranking_df.to_csv(ranking_csv, float_format="%.6f")
    print(f"\n  Saved ranking table: {ranking_csv}")

    print(f"\nAll outputs saved to: {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
