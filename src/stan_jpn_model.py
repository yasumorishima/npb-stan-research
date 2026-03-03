"""
Step 7a: Japanese player (all NPB players) Stan model with K%/BB% features.

Marcel prediction serves as the prior mean. The Stan model tests whether
K%/BB% (skill-level stats) add predictive power beyond Marcel.

Model:
  Hitter:  actual_wOBA = Marcel_wOBA + delta_K * z_K + delta_BB * z_BB + noise
  Pitcher: actual_ERA  = Marcel_ERA  + delta_K * z_K + delta_BB * z_BB + noise

Training: 2018-2021 (using 2015-2020 history for Marcel)
Backtest: 2022-2025 (using 2019-2024 history for Marcel)

Output:
  data/model/jpn_hitter_predictions.csv  (year, player, team, marcel_woba, stan_woba, actual_woba, PA)
  data/model/jpn_pitcher_predictions.csv (year, player, team, marcel_era, stan_era, actual_era, IP)
  data/model/jpn_comparison.json         (MAE summary: Marcel vs Stan)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
RAW_DIR   = DATA_DIR / "raw"

# ── Cutoffs ─────────────────────────────────────────────────────────────────
MIN_PA  = 50     # minimum PA to include a hitter in training/test
MIN_IP  = 20     # minimum IP to include a pitcher in training/test

# ── Marcel parameters ────────────────────────────────────────────────────────
MARCEL_WEIGHTS   = {1: 5, 2: 4, 3: 3}   # years back: weight
REGRESS_PA_HIT   = 200                   # regression-to-mean PA (hitters)
REGRESS_IP_PITCH = 300                   # regression-to-mean IP (pitchers)

# ── Years ────────────────────────────────────────────────────────────────────
TRAIN_YEARS    = list(range(2018, 2022))  # 2018-2021
BACKTEST_YEARS = list(range(2022, 2026))  # 2022-2025


# ── IP format helper ─────────────────────────────────────────────────────────
def ip_to_decimal(ip: float) -> float:
    """Convert NPB IP (X.Y where Y = thirds) to decimal innings."""
    whole  = int(ip)
    thirds = round((ip - whole) * 10)
    return whole + thirds / 3.0


# ── Marcel helpers ────────────────────────────────────────────────────────────
def league_avg_woba(saber_df: pd.DataFrame, year: int) -> float:
    sub = saber_df[(saber_df["year"] == year - 1) & (saber_df["PA"] >= MIN_PA)]
    if len(sub) == 0:
        return 0.310
    return float(np.average(sub["wOBA"], weights=sub["PA"]))


def league_avg_era(pitchers_df: pd.DataFrame, year: int) -> float:
    sub = pitchers_df[pitchers_df["year"] == year - 1].copy()
    sub["IP_dec"] = sub["IP"].apply(ip_to_decimal)
    sub["ERA"]    = pd.to_numeric(sub["ERA"], errors="coerce")
    sub = sub[(sub["IP_dec"] >= MIN_IP) & sub["ERA"].notna()]
    if len(sub) == 0:
        return 3.80
    total_ip = sub["IP_dec"].sum()
    return float((sub["ERA"] * sub["IP_dec"]).sum() / total_ip) if total_ip > 0 else 3.80


def compute_marcel_woba(saber_df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Marcel wOBA projection for all players with sufficient history."""
    lg_avg = league_avg_woba(saber_df, target_year)
    rows = []
    for player, grp in saber_df.groupby("player"):
        w_total = woba_sum = 0.0
        for lag, w in MARCEL_WEIGHTS.items():
            yr = target_year - lag
            row = grp[grp["year"] == yr]
            if len(row) == 0:
                continue
            pa   = float(row.iloc[0]["PA"])
            woba = float(row.iloc[0]["wOBA"])
            if pa >= MIN_PA and not np.isnan(woba):
                w_total  += w * pa
                woba_sum += w * pa * woba
        if w_total == 0:
            continue
        # Most-recent year for team assignment
        recent = grp[grp["year"] == target_year - 1]
        if len(recent) == 0:
            continue
        team = recent.iloc[0]["team"]
        woba_raw  = woba_sum / w_total
        woba_proj = (woba_raw * w_total + lg_avg * REGRESS_PA_HIT) / (w_total + REGRESS_PA_HIT)
        rows.append({"player": player, "team": team, "year": target_year,
                     "marcel_woba": round(woba_proj, 5), "lg_avg_woba": round(lg_avg, 5)})
    return pd.DataFrame(rows)


def compute_marcel_era(pitchers_df: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Marcel ERA projection for all pitchers with sufficient history."""
    lg_avg = league_avg_era(pitchers_df, target_year)
    rows = []
    for player, grp in pitchers_df.groupby("player"):
        w_total = era_sum = 0.0
        for lag, w in MARCEL_WEIGHTS.items():
            yr = target_year - lag
            row = grp[grp["year"] == yr]
            if len(row) == 0:
                continue
            ip  = ip_to_decimal(float(row.iloc[0]["IP"]))
            era = pd.to_numeric(row.iloc[0]["ERA"], errors="coerce")
            if ip >= MIN_IP and not np.isnan(era):
                era = float(era)
                w_total += w * ip
                era_sum += w * ip * era
        if w_total == 0:
            continue
        recent = grp[grp["year"] == target_year - 1]
        if len(recent) == 0:
            continue
        team = recent.iloc[0]["team"]
        era_raw  = era_sum / w_total
        era_proj = (era_raw * w_total + lg_avg * REGRESS_IP_PITCH) / (w_total + REGRESS_IP_PITCH)
        rows.append({"player": player, "team": team, "year": target_year,
                     "marcel_era": round(era_proj, 4), "lg_avg_era": round(lg_avg, 4)})
    return pd.DataFrame(rows)


def add_kpct_bbpct_hitter(saber_df: pd.DataFrame, target_year: int,
                           marcel_df: pd.DataFrame) -> pd.DataFrame:
    """Add K%/BB%/BABIP features from the year before the target year.

    BABIP = (H - HR) / (AB - SO - HR + SF) captures luck in year t-1.
    High BABIP in t-1 → Marcel overestimates year t → expected delta_BABIP < 0.
    """
    cols = ["player", "PA", "SO", "BB", "AB", "H", "HR", "SF"]
    prev = saber_df[saber_df["year"] == target_year - 1][cols].copy()
    prev = prev[prev["PA"] >= MIN_PA]
    prev["K_pct"]  = prev["SO"] / prev["PA"]
    prev["BB_pct"] = prev["BB"] / prev["PA"]
    denom = (prev["AB"] - prev["SO"] - prev["HR"] + prev["SF"]).clip(lower=1)
    prev["BABIP"]  = (prev["H"] - prev["HR"]) / denom
    merged = marcel_df.merge(prev[["player", "K_pct", "BB_pct", "BABIP"]], on="player", how="inner")
    return merged


def add_kpct_bbpct_pitcher(pitchers_df: pd.DataFrame, target_year: int,
                            marcel_df: pd.DataFrame) -> pd.DataFrame:
    """Add K%/BB% features from the year before the target year."""
    prev = pitchers_df[pitchers_df["year"] == target_year - 1][
        ["player", "IP", "SO", "BB", "BF"]
    ].copy()
    prev["IP_dec"] = prev["IP"].apply(ip_to_decimal)
    prev = prev[prev["IP_dec"] >= MIN_IP]
    prev = prev[prev["BF"] > 0]
    prev["K_pct"]  = prev["SO"] / prev["BF"]
    prev["BB_pct"] = prev["BB"] / prev["BF"]
    merged = marcel_df.merge(prev[["player", "K_pct", "BB_pct"]], on="player", how="inner")
    return merged


def add_actual_woba(saber_df: pd.DataFrame, target_year: int,
                    df: pd.DataFrame) -> pd.DataFrame:
    actual = saber_df[saber_df["year"] == target_year][["player", "PA", "wOBA"]].copy()
    actual = actual.rename(columns={"PA": "actual_PA", "wOBA": "actual_woba"})
    actual = actual[actual["actual_PA"] >= MIN_PA]
    return df.merge(actual, on="player", how="inner")


def add_actual_era(pitchers_df: pd.DataFrame, target_year: int,
                   df: pd.DataFrame) -> pd.DataFrame:
    actual = pitchers_df[pitchers_df["year"] == target_year][["player", "IP", "ERA"]].copy()
    actual["actual_IP"]  = actual["IP"].apply(ip_to_decimal)
    actual["actual_era"] = pd.to_numeric(actual["ERA"], errors="coerce")
    actual = actual[(actual["actual_IP"] >= MIN_IP) & actual["actual_era"].notna()]
    return df.merge(actual[["player", "actual_IP", "actual_era"]], on="player", how="inner")


def build_dataset(saber_df, pitchers_df, years):
    """Build combined DataFrame for hitters and pitchers across given years."""
    hit_rows, pit_rows = [], []
    for yr in years:
        m_h = compute_marcel_woba(saber_df, yr)
        if len(m_h) == 0:
            continue
        m_h = add_kpct_bbpct_hitter(saber_df, yr, m_h)
        m_h = add_actual_woba(saber_df, yr, m_h)
        hit_rows.append(m_h)

        m_p = compute_marcel_era(pitchers_df, yr)
        if len(m_p) == 0:
            continue
        m_p = add_kpct_bbpct_pitcher(pitchers_df, yr, m_p)
        m_p = add_actual_era(pitchers_df, yr, m_p)
        pit_rows.append(m_p)

    hitters  = pd.concat(hit_rows,  ignore_index=True) if hit_rows  else pd.DataFrame()
    pitchers = pd.concat(pit_rows,  ignore_index=True) if pit_rows  else pd.DataFrame()
    return hitters, pitchers


def standardize_features(train_df, test_df, feat_cols):
    """Z-score features using training-set statistics. Returns scaled arrays."""
    means = train_df[feat_cols].mean()
    stds  = train_df[feat_cols].std().replace(0, 1)
    train_z = (train_df[feat_cols] - means) / stds
    test_z  = (test_df[feat_cols]  - means) / stds
    return train_z.values, test_z.values, means.to_dict(), stds.to_dict()


def run_stan_model(model_path: Path, stan_data: dict, draws=1000, warmup=500):
    """Run Stan model via cmdstanpy. Returns posterior means."""
    from cmdstanpy import CmdStanModel
    model = CmdStanModel(stan_file=str(model_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.sample(
            data=stan_data,
            chains=2,
            iter_sampling=draws,
            iter_warmup=warmup,
            show_progress=False,
            show_console=False,
        )
    return fit


def main(draws=1000, warmup=500):
    print("Loading raw data...")
    saber    = pd.read_csv(RAW_DIR / "npb_sabermetrics_2015_2025.csv",    encoding="utf-8-sig")
    pitchers = pd.read_csv(RAW_DIR / "npb_pitchers_2015_2025.csv",        encoding="utf-8-sig")
    saber    = saber.dropna(subset=["wOBA"])

    print(f"  Sabermetrics: {len(saber):,} player-years")
    print(f"  Pitchers:     {len(pitchers):,} player-years")

    # ── Build datasets ────────────────────────────────────────────────────────
    print("\nBuilding training data (2018-2021)...")
    train_h, train_p = build_dataset(saber, pitchers, TRAIN_YEARS)
    print(f"  Hitters train:  {len(train_h):3d}")
    print(f"  Pitchers train: {len(train_p):3d}")

    print("Building backtest data (2022-2025)...")
    test_h, test_p = build_dataset(saber, pitchers, BACKTEST_YEARS)
    print(f"  Hitters test:   {len(test_h):3d}")
    print(f"  Pitchers test:  {len(test_p):3d}")

    if len(train_h) == 0 or len(test_h) == 0:
        print("ERROR: insufficient data")
        return

    # ── Standardize ───────────────────────────────────────────────────────────
    feat_cols_h = ["K_pct", "BB_pct", "BABIP"]
    feat_cols_p = ["K_pct", "BB_pct"]

    train_z_h, test_z_h, h_means, h_stds = standardize_features(train_h, test_h, feat_cols_h)
    train_z_p, test_z_p, p_means, p_stds = standardize_features(train_p, test_p, feat_cols_p)

    # ── Stan: Hitters ─────────────────────────────────────────────────────────
    print("\nRunning Stan model — hitters...")
    stan_data_h = {
        "N":                  len(train_h),
        "marcel_woba":        train_h["marcel_woba"].tolist(),
        "z_K":                train_z_h[:, 0].tolist(),
        "z_BB":               train_z_h[:, 1].tolist(),
        "z_babip":            train_z_h[:, 2].tolist(),
        "actual_woba":        train_h["actual_woba"].tolist(),
        "N_pred":             len(test_h),
        "marcel_woba_pred":   test_h["marcel_woba"].tolist(),
        "z_K_pred":           test_z_h[:, 0].tolist(),
        "z_BB_pred":          test_z_h[:, 1].tolist(),
        "z_babip_pred":       test_z_h[:, 2].tolist(),
    }
    fit_h = run_stan_model(ROOT / "models" / "hitter_jpn.stan", stan_data_h, draws, warmup)

    # posterior means of test predictions
    stan_pred_h = fit_h.stan_variable("stan_pred").mean(axis=0)
    delta_K_h     = float(fit_h.stan_variable("delta_K").mean())
    delta_BB_h    = float(fit_h.stan_variable("delta_BB").mean())
    delta_BABIP_h = float(fit_h.stan_variable("delta_BABIP").mean())
    print(f"  delta_K={delta_K_h:+.4f}  delta_BB={delta_BB_h:+.4f}  delta_BABIP={delta_BABIP_h:+.4f}")

    test_h = test_h.copy()
    test_h["stan_woba"] = stan_pred_h

    mae_marcel_h = float((test_h["actual_woba"] - test_h["marcel_woba"]).abs().mean())
    mae_stan_h   = float((test_h["actual_woba"] - test_h["stan_woba"]).abs().mean())
    print(f"  Hitter wOBA MAE — Marcel: {mae_marcel_h:.4f}  Stan: {mae_stan_h:.4f}"
          f"  Δ: {mae_stan_h - mae_marcel_h:+.4f}")

    # ── Stan: Pitchers ────────────────────────────────────────────────────────
    print("\nRunning Stan model — pitchers...")
    stan_data_p = {
        "N":               len(train_p),
        "marcel_era":      train_p["marcel_era"].tolist(),
        "z_K":             train_z_p[:, 0].tolist(),
        "z_BB":            train_z_p[:, 1].tolist(),
        "actual_era":      train_p["actual_era"].tolist(),
        "N_pred":          len(test_p),
        "marcel_era_pred": test_p["marcel_era"].tolist(),
        "z_K_pred":        test_z_p[:, 0].tolist(),
        "z_BB_pred":       test_z_p[:, 1].tolist(),
    }
    fit_p = run_stan_model(ROOT / "models" / "pitcher_jpn.stan", stan_data_p, draws, warmup)

    stan_pred_p = fit_p.stan_variable("stan_pred").mean(axis=0)
    delta_K_p   = float(fit_p.stan_variable("delta_K").mean())
    delta_BB_p  = float(fit_p.stan_variable("delta_BB").mean())
    print(f"  delta_K={delta_K_p:+.4f}  delta_BB={delta_BB_p:+.4f}")

    test_p = test_p.copy()
    test_p["stan_era"] = stan_pred_p

    mae_marcel_p = float((test_p["actual_era"] - test_p["marcel_era"]).abs().mean())
    mae_stan_p   = float((test_p["actual_era"] - test_p["stan_era"]).abs().mean())
    print(f"  Pitcher ERA  MAE — Marcel: {mae_marcel_p:.4f}  Stan: {mae_stan_p:.4f}"
          f"  Δ: {mae_stan_p - mae_marcel_p:+.4f}")

    # ── Save predictions ──────────────────────────────────────────────────────
    cols_h = ["year", "player", "team", "marcel_woba", "stan_woba", "actual_woba", "actual_PA"]
    cols_p = ["year", "player", "team", "marcel_era", "stan_era", "actual_era", "actual_IP"]

    test_h[cols_h].to_csv(MODEL_DIR / "jpn_hitter_predictions.csv",
                          index=False, encoding="utf-8-sig")
    test_p[cols_p].to_csv(MODEL_DIR / "jpn_pitcher_predictions.csv",
                          index=False, encoding="utf-8-sig")

    comparison = {
        "hitter": {
            "mae_marcel": round(mae_marcel_h, 4),
            "mae_stan":   round(mae_stan_h, 4),
            "delta_mae":  round(mae_stan_h - mae_marcel_h, 4),
            "delta_K":     round(delta_K_h, 4),
            "delta_BB":    round(delta_BB_h, 4),
            "delta_BABIP": round(delta_BABIP_h, 4),
            "n_test":      len(test_h),
            "feature_means": h_means,
            "feature_stds":  h_stds,
        },
        "pitcher": {
            "mae_marcel": round(mae_marcel_p, 4),
            "mae_stan":   round(mae_stan_p, 4),
            "delta_mae":  round(mae_stan_p - mae_marcel_p, 4),
            "delta_K":    round(delta_K_p, 4),
            "delta_BB":   round(delta_BB_p, 4),
            "n_test":     len(test_p),
            "feature_means": p_means,
            "feature_stds":  p_stds,
        },
    }
    (MODEL_DIR / "jpn_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved -> {MODEL_DIR / 'jpn_hitter_predictions.csv'}")
    print(f"Saved -> {MODEL_DIR / 'jpn_pitcher_predictions.csv'}")
    print(f"Saved -> {MODEL_DIR / 'jpn_comparison.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws",  type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=500)
    args = parser.parse_args()
    main(args.draws, args.warmup)
