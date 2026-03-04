"""Step 11b: Statistical validation — age curve + recency weighting + FIP.

Five analyses:
  1. Player-level significance tests (n=1000+ hitters + pitchers)
     — 8-year LOO-CV with Ridge regression as Stan approximation
     — Now includes age_from_peak feature
  2. 2021 (COVID year) exclusion — 7-year results
  3. Foreign player Stan v1 LOO-CV (2015-2025)
  4. Recency weighting comparison (λ = 1.0, 0.9, 0.8)
  5. FIP vs ERA pitcher model comparison

Ridge alphas (= sigma^2 / tau^2, posterior mean approximation):
  Japanese hitter:  0.053^2 / 0.05^2 = 1.12  (4 features: K%, BB%, BABIP, age)
  Japanese pitcher: 1.10^2  / 0.5^2  = 4.84  (3 features: K%, BB%, age)
  Foreign hitter:   0.05^2  / 0.02^2 = 6.25  (3 features: wOBA, K%, BB%)
  Foreign pitcher:  1.0^2   / 0.5^2  = 4.0   (4 features: ERA, FIP, K%, BB%)

Output:
  data/projections/statistical_validation.json
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Allow importing stan_jpn_model from same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stan_jpn_model import (
    MIN_IP,
    MIN_PA,
    build_dataset,
    compute_fip_column,
    ip_to_decimal,
    load_birthday_df,
    standardize_features,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
RAW_DIR = DATA_DIR / "raw"
FOREIGN_DIR = DATA_DIR / "foreign"
OUT_DIR = DATA_DIR / "projections"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ridge alphas (sigma^2 / tau^2)
ALPHA_JPN_H = 0.053**2 / 0.05**2   # 1.1236
ALPHA_JPN_P = 1.10**2 / 0.5**2     # 4.84
ALPHA_FGN_H = 0.05**2 / 0.02**2    # 6.25
ALPHA_FGN_P = 1.0**2 / 0.5**2      # 4.0

# Japanese model LOO-CV years (Marcel needs 3 prior years → earliest = 2018)
JPN_YEARS = list(range(2018, 2026))

# Team-level constants
NPB_PYTH_EXP = 1.83
NPB_HIST_RS = 535.0
K_WOBA = 0.3256


# ── Team-level helpers (Step 12) ─────────────────────────────────────────────


def _load_raw_data():
    """Load raw sabermetrics and pitcher data for team-level fixes."""
    saber = pd.read_csv(RAW_DIR / "npb_sabermetrics_2015_2025.csv", encoding="utf-8-sig")
    pitchers_raw = pd.read_csv(RAW_DIR / "npb_pitchers_2015_2025.csv", encoding="utf-8-sig")
    pitchers_raw["IP_dec"] = pitchers_raw["IP"].apply(ip_to_decimal)
    pitchers_raw["ERA_num"] = pd.to_numeric(pitchers_raw["ERA"], errors="coerce")
    return saber, pitchers_raw


def _reassign_teams(h_df, p_df, saber, pitchers_raw):
    """Fix FA attribution: replace team with actual year-t team from raw data.

    For mid-season trades, the team with more PA/IP is used.
    """
    h_df = h_df.copy()
    p_df = p_df.copy()

    # Hitter lookup: (player, year) → team (keep team with max PA)
    h_lookup = (
        saber[saber["PA"] >= MIN_PA]
        .sort_values("PA", ascending=False)
        .drop_duplicates(subset=["player", "year"], keep="first")
        .set_index(["player", "year"])["team"]
    )
    for idx, row in h_df.iterrows():
        key = (row["player"], row["year"])
        if key in h_lookup.index:
            h_df.at[idx, "team"] = h_lookup[key]

    # Pitcher lookup: (player, year) → team (keep team with max IP)
    p_lookup = (
        pitchers_raw[pitchers_raw["IP_dec"] >= MIN_IP]
        .sort_values("IP_dec", ascending=False)
        .drop_duplicates(subset=["player", "year"], keep="first")
        .set_index(["player", "year"])["team"]
    )
    for idx, row in p_df.iterrows():
        key = (row["player"], row["year"])
        if key in p_lookup.index:
            p_df.at[idx, "team"] = p_lookup[key]

    return h_df, p_df


def _compute_league_averages(saber, pitchers_raw):
    """Compute PA-weighted wOBA and IP-weighted ERA per year for imputation."""
    lg_woba = {}
    for yr, grp in saber[saber["PA"] >= MIN_PA].groupby("year"):
        total_pa = grp["PA"].sum()
        if total_pa > 0:
            lg_woba[yr] = float((grp["wOBA"] * grp["PA"]).sum() / total_pa)

    lg_era = {}
    for yr, grp in pitchers_raw.groupby("year"):
        sub = grp[(grp["IP_dec"] >= MIN_IP) & grp["ERA_num"].notna()]
        total_ip = sub["IP_dec"].sum()
        if total_ip > 0:
            lg_era[yr] = float((sub["ERA_num"] * sub["IP_dec"]).sum() / total_ip)

    return lg_woba, lg_era


def _impute_missing_players(h_df, p_df, saber, pitchers_raw, lg_woba, lg_era):
    """Compute imputed RS/RA for missing players and coverage rates.

    Returns dict keyed by (year, team) with imputed_rs, imputed_ra,
    PA_cov, IP_cov values. Same imputation is added to both Marcel
    and Stan to keep the comparison fair.
    """
    result = {}

    for yr in sorted(h_df["year"].unique()):
        # Actual team PA totals (all players)
        yr_saber = saber[saber["year"] == yr]
        actual_pa = yr_saber.groupby("team")["PA"].sum()

        # Actual team IP totals (all pitchers)
        yr_pitch = pitchers_raw[pitchers_raw["year"] == yr]
        actual_ip = yr_pitch.groupby("team")["IP_dec"].sum()

        # Model PA/IP (after reassignment)
        yr_h = h_df[h_df["year"] == yr]
        model_pa = yr_h.groupby("team")["actual_PA"].sum()

        yr_p = p_df[p_df["year"] == yr]
        model_ip = yr_p.groupby("team")["actual_IP"].sum()

        teams = set(actual_pa.index) | set(actual_ip.index)
        woba_avg = lg_woba.get(yr, 0.310)
        era_avg = lg_era.get(yr, 3.80)

        for team in teams:
            act_pa = float(actual_pa.get(team, 0))
            mod_pa = float(model_pa.get(team, 0))
            act_ip = float(actual_ip.get(team, 0))
            mod_ip = float(model_ip.get(team, 0))

            missing_pa = max(act_pa - mod_pa, 0)
            missing_ip = max(act_ip - mod_ip, 0)

            imputed_rs = K_WOBA * woba_avg * missing_pa
            imputed_ra = era_avg * missing_ip / 9.0

            pa_cov = mod_pa / act_pa * 100 if act_pa > 0 else 0
            ip_cov = mod_ip / act_ip * 100 if act_ip > 0 else 0

            result[(yr, team)] = {
                "imputed_rs": imputed_rs,
                "imputed_ra": imputed_ra,
                "PA_cov": round(pa_cov, 1),
                "IP_cov": round(ip_cov, 1),
            }

    return result


def ridge_fit_predict(X_train, y_train, X_test, alpha):
    """Ridge regression: beta = (X'X + alpha*I)^{-1} X'y."""
    n_feat = X_train.shape[1]
    beta = np.linalg.solve(
        X_train.T @ X_train + alpha * np.eye(n_feat),
        X_train.T @ y_train,
    )
    return X_test @ beta, beta


def weighted_ridge_fit_predict(X_train, y_train, X_test, alpha, weights):
    """Weighted Ridge: beta = (X'WX + alpha*I)^{-1} X'Wy."""
    n_feat = X_train.shape[1]
    W = np.diag(weights)
    beta = np.linalg.solve(
        X_train.T @ W @ X_train + alpha * np.eye(n_feat),
        X_train.T @ W @ y_train,
    )
    return X_test @ beta, beta


# ── Analysis 1: Japanese Player LOO-CV ────────────────────────────────────────


def run_jpn_loocv(decay_lambda=1.0):
    """8-year LOO-CV for Japanese players with age feature and recency weighting.

    Args:
        decay_lambda: Exponential decay factor for recency weighting.
            w_i = lambda^|holdout_year - sample_year|
            1.0 = no decay (uniform), 0.9 = mild decay, 0.8 = strong decay.

    Returns:
        (h_df, p_df, p_fip_df, p_k9_df, p_fip_k9_df) — Player-level
        prediction DataFrames for hitters, pitchers (ERA 3feat),
        pitchers (FIP 3feat), pitchers (ERA 5feat+K/9+BB/9),
        pitchers (FIP 5feat+K/9+BB/9).
    """
    saber = pd.read_csv(RAW_DIR / "npb_sabermetrics_2015_2025.csv", encoding="utf-8-sig")
    pitchers = pd.read_csv(RAW_DIR / "npb_pitchers_2015_2025.csv", encoding="utf-8-sig")
    saber = saber.dropna(subset=["wOBA"])
    pitchers = compute_fip_column(pitchers)  # Add FIP column
    bday_df = load_birthday_df()

    feat_h = ["K_pct", "BB_pct", "BABIP", "age_from_peak"]
    feat_p = ["K_pct", "BB_pct", "age_from_peak"]
    feat_p_k9 = ["K_pct", "BB_pct", "K_per_9", "BB_per_9", "age_from_peak"]

    all_h, all_p, all_p_fip = [], [], []
    all_p_k9, all_p_fip_k9 = [], []

    for hold_yr in JPN_YEARS:
        train_years = [y for y in JPN_YEARS if y != hold_yr]
        train_h, train_p, train_p_fip = build_dataset(saber, pitchers, train_years, bday_df)
        test_h, test_p, test_p_fip = build_dataset(saber, pitchers, [hold_yr], bday_df)

        if len(test_h) == 0 or len(test_p) == 0:
            continue

        # Hitters
        train_z_h, test_z_h, _, _ = standardize_features(train_h, test_h, feat_h)
        y_h = (train_h["actual_woba"] - train_h["marcel_woba"]).values

        if decay_lambda < 1.0:
            w_h = np.array([decay_lambda ** abs(hold_yr - yr)
                            for yr in train_h["year"].values])
            delta_h, _ = weighted_ridge_fit_predict(
                train_z_h, y_h, test_z_h, ALPHA_JPN_H, w_h)
        else:
            delta_h, _ = ridge_fit_predict(train_z_h, y_h, test_z_h, ALPHA_JPN_H)

        stan_woba = test_h["marcel_woba"].values + delta_h

        for i, (_, row) in enumerate(test_h.iterrows()):
            all_h.append({
                "year": hold_yr, "player": row["player"], "team": row["team"],
                "actual": row["actual_woba"], "marcel": row["marcel_woba"],
                "stan": stan_woba[i], "actual_PA": row["actual_PA"],
            })

        # Pitchers (ERA)
        train_z_p, test_z_p, _, _ = standardize_features(train_p, test_p, feat_p)
        y_p = (train_p["actual_era"] - train_p["marcel_era"]).values

        if decay_lambda < 1.0:
            w_p = np.array([decay_lambda ** abs(hold_yr - yr)
                            for yr in train_p["year"].values])
            delta_p, _ = weighted_ridge_fit_predict(
                train_z_p, y_p, test_z_p, ALPHA_JPN_P, w_p)
        else:
            delta_p, _ = ridge_fit_predict(train_z_p, y_p, test_z_p, ALPHA_JPN_P)

        stan_era = test_p["marcel_era"].values + delta_p

        for i, (_, row) in enumerate(test_p.iterrows()):
            all_p.append({
                "year": hold_yr, "player": row["player"], "team": row["team"],
                "actual": row["actual_era"], "marcel": row["marcel_era"],
                "stan": stan_era[i], "actual_IP": row["actual_IP"],
            })

        # Pitchers (ERA + K/9 + BB/9)
        train_z_pk9, test_z_pk9, _, _ = standardize_features(
            train_p, test_p, feat_p_k9)

        if decay_lambda < 1.0:
            delta_pk9, _ = weighted_ridge_fit_predict(
                train_z_pk9, y_p, test_z_pk9, ALPHA_JPN_P, w_p)
        else:
            delta_pk9, _ = ridge_fit_predict(
                train_z_pk9, y_p, test_z_pk9, ALPHA_JPN_P)

        stan_era_k9 = test_p["marcel_era"].values + delta_pk9

        for i, (_, row) in enumerate(test_p.iterrows()):
            all_p_k9.append({
                "year": hold_yr, "player": row["player"], "team": row["team"],
                "actual": row["actual_era"], "marcel": row["marcel_era"],
                "stan": stan_era_k9[i], "actual_IP": row["actual_IP"],
            })

        # Pitchers (FIP)
        if len(test_p_fip) > 0 and len(train_p_fip) > 0:
            train_z_pf, test_z_pf, _, _ = standardize_features(
                train_p_fip, test_p_fip, feat_p)
            y_pf = (train_p_fip["actual_fip"] - train_p_fip["marcel_fip"]).values

            if decay_lambda < 1.0:
                w_pf = np.array([decay_lambda ** abs(hold_yr - yr)
                                 for yr in train_p_fip["year"].values])
                delta_pf, _ = weighted_ridge_fit_predict(
                    train_z_pf, y_pf, test_z_pf, ALPHA_JPN_P, w_pf)
            else:
                delta_pf, _ = ridge_fit_predict(
                    train_z_pf, y_pf, test_z_pf, ALPHA_JPN_P)

            stan_fip = test_p_fip["marcel_fip"].values + delta_pf

            for i, (_, row) in enumerate(test_p_fip.iterrows()):
                all_p_fip.append({
                    "year": hold_yr, "player": row["player"], "team": row["team"],
                    "actual": row["actual_fip"], "marcel": row["marcel_fip"],
                    "stan": stan_fip[i], "actual_IP": row["actual_IP"],
                })

            # FIP + K/9 + BB/9
            train_z_pfk9, test_z_pfk9, _, _ = standardize_features(
                train_p_fip, test_p_fip, feat_p_k9)

            if decay_lambda < 1.0:
                delta_pfk9, _ = weighted_ridge_fit_predict(
                    train_z_pfk9, y_pf, test_z_pfk9, ALPHA_JPN_P, w_pf)
            else:
                delta_pfk9, _ = ridge_fit_predict(
                    train_z_pfk9, y_pf, test_z_pfk9, ALPHA_JPN_P)

            stan_fip_k9 = test_p_fip["marcel_fip"].values + delta_pfk9

            for i, (_, row) in enumerate(test_p_fip.iterrows()):
                all_p_fip_k9.append({
                    "year": hold_yr, "player": row["player"], "team": row["team"],
                    "actual": row["actual_fip"], "marcel": row["marcel_fip"],
                    "stan": stan_fip_k9[i], "actual_IP": row["actual_IP"],
                })

    return (pd.DataFrame(all_h), pd.DataFrame(all_p), pd.DataFrame(all_p_fip),
            pd.DataFrame(all_p_k9), pd.DataFrame(all_p_fip_k9))


def _metric_test(df, label, metric_name):
    """Paired t-test, Wilcoxon, bootstrap on player-level absolute errors for one metric."""
    err_m = np.abs(df["actual"].values - df["marcel"].values)
    err_s = np.abs(df["actual"].values - df["stan"].values)
    diff = err_m - err_s   # positive → Stan better

    n = len(diff)
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else 1.0

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(err_m, err_s)
    # Wilcoxon signed-rank
    try:
        w_stat, w_p = stats.wilcoxon(diff)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    # Bootstrap P(Stan < Marcel)
    rng = np.random.default_rng(42)
    boot_means = [np.mean(diff[rng.integers(0, n, n)]) for _ in range(10000)]
    p_stan_better = float(np.mean([m > 0 for m in boot_means]))

    # Cohen's d
    d = mean_diff / sd_diff if sd_diff > 0 else 0.0

    stan_wins = int(np.sum(err_s < err_m))

    result = {
        "n": n,
        "mae_marcel": round(float(np.mean(err_m)), 5),
        "mae_stan": round(float(np.mean(err_s)), 5),
        "delta_mae": round(mean_diff, 5),
        "paired_t_stat": round(float(t_stat), 4),
        "paired_t_p": round(float(t_p), 6),
        "wilcoxon_p": round(float(w_p), 6),
        "bootstrap_p_stan_better": round(p_stan_better, 4),
        "cohens_d": round(float(d), 4),
        "stan_win_rate": f"{stan_wins}/{n} ({100 * stan_wins / n:.1f}%)",
    }

    print(f"\n  {label} — {metric_name} (n={n}):")
    print(f"    MAE Marcel={np.mean(err_m):.5f}  Stan={np.mean(err_s):.5f}  "
          f"Δ={mean_diff:+.5f}")
    print(f"    Paired t: t={t_stat:.3f}, p={t_p:.6f}")
    print(f"    Wilcoxon: p={w_p:.6f}")
    print(f"    Bootstrap P(Stan<Marcel): {p_stan_better:.4f}")
    print(f"    Cohen's d: {d:.4f}")
    print(f"    Stan wins: {stan_wins}/{n} ({100 * stan_wins / n:.1f}%)")

    return result


def player_level_tests(h_df, p_df, label="All years"):
    """Paired t-test, Wilcoxon, bootstrap on player-level absolute errors."""
    return {
        "hitter_woba": _metric_test(h_df, label, "hitter_woba"),
        "pitcher_era": _metric_test(p_df, label, "pitcher_era"),
    }


def team_level_mae(h_df, p_df, years=None, label="All years"):
    """Compute team-level Pythagorean MAE from player-level LOO-CV predictions.

    Step 12 fixes:
      1. FA reassignment — use actual year-t team from raw data
      2. Missing player imputation — fill uncovered PA/IP with league avg
      3. Coverage display — show PA_cov/IP_cov per team-year
    """
    actual = pd.read_csv(
        "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main"
        "/data/projections/pythagorean_2015_2025.csv",
        encoding="utf-8-sig",
    )

    if years is not None:
        h_df = h_df[h_df["year"].isin(years)]
        p_df = p_df[p_df["year"].isin(years)]
        actual = actual[actual["year"].isin(years)]

    # Step 12-1: Reassign teams using actual year-t data
    saber, pitchers_raw = _load_raw_data()
    h_df, p_df = _reassign_teams(h_df, p_df, saber, pitchers_raw)

    # Step 12-2: Compute league averages and imputation
    lg_woba, lg_era = _compute_league_averages(saber, pitchers_raw)
    impute = _impute_missing_players(h_df, p_df, saber, pitchers_raw, lg_woba, lg_era)

    # RS from hitters
    h = h_df.copy()
    h["rs_marcel"] = K_WOBA * h["marcel"] * h["actual_PA"]
    h["rs_stan"] = K_WOBA * h["stan"] * h["actual_PA"]
    rs = h.groupby(["year", "team"])[["rs_marcel", "rs_stan"]].sum().reset_index()

    # RA from pitchers
    p = p_df.copy()
    p["ra_marcel"] = p["marcel"] * p["actual_IP"] / 9.0
    p["ra_stan"] = p["stan"] * p["actual_IP"] / 9.0
    ra = p.groupby(["year", "team"])[["ra_marcel", "ra_stan"]].sum().reset_index()

    team = rs.merge(ra, on=["year", "team"], how="inner")
    merged = team.merge(
        actual[["year", "team", "G", "W"]], on=["year", "team"], how="inner"
    )

    # Step 12-2: Add imputed RS/RA (same amount for both models → fair comparison)
    merged["imputed_rs"] = merged.apply(
        lambda r: impute.get((r["year"], r["team"]), {}).get("imputed_rs", 0), axis=1)
    merged["imputed_ra"] = merged.apply(
        lambda r: impute.get((r["year"], r["team"]), {}).get("imputed_ra", 0), axis=1)
    merged["rs_marcel"] += merged["imputed_rs"]
    merged["rs_stan"] += merged["imputed_rs"]
    merged["ra_marcel"] += merged["imputed_ra"]
    merged["ra_stan"] += merged["imputed_ra"]

    # Step 12-3: Add coverage columns
    merged["PA_cov"] = merged.apply(
        lambda r: impute.get((r["year"], r["team"]), {}).get("PA_cov", 0), axis=1)
    merged["IP_cov"] = merged.apply(
        lambda r: impute.get((r["year"], r["team"]), {}).get("IP_cov", 0), axis=1)

    # Marcel-anchored scaling
    for yr in merged["year"].unique():
        mask = merged["year"] == yr
        for col_m, col_s in [
            ("rs_marcel", "rs_stan"),
            ("ra_marcel", "ra_stan"),
        ]:
            avg = merged.loc[mask, col_m].mean()
            if avg > 0:
                f = NPB_HIST_RS / avg
                merged.loc[mask, col_m] *= f
                merged.loc[mask, col_s] *= f

    # Pythagorean wins
    for model in ["marcel", "stan"]:
        rs_v = np.clip(merged[f"rs_{model}"].values, 1.0, None)
        ra_v = np.clip(merged[f"ra_{model}"].values, 1.0, None)
        wpct = rs_v ** NPB_PYTH_EXP / (rs_v ** NPB_PYTH_EXP + ra_v ** NPB_PYTH_EXP)
        merged[f"W_{model}"] = wpct * merged["G"].values

    merged["err_marcel"] = merged["W_marcel"] - merged["W"]
    merged["err_stan"] = merged["W_stan"] - merged["W"]

    mae_m = float(merged["err_marcel"].abs().mean())
    mae_s = float(merged["err_stan"].abs().mean())

    # Paired tests on team-level errors
    err_m_abs = merged["err_marcel"].abs().values
    err_s_abs = merged["err_stan"].abs().values
    diff = err_m_abs - err_s_abs
    t_stat, t_p = stats.ttest_rel(err_m_abs, err_s_abs)

    # Bootstrap
    rng = np.random.default_rng(42)
    n = len(diff)
    boot_means = [np.mean(diff[rng.integers(0, n, n)]) for _ in range(10000)]
    p_stan_better = float(np.mean([m > 0 for m in boot_means]))

    print(f"\n  {label} — Team MAE (n={len(merged)}):")
    print(f"    Marcel={mae_m:.3f}  Stan={mae_s:.3f}  Δ={mae_s - mae_m:+.3f}")
    print(f"    Paired t: p={t_p:.6f}  Bootstrap P(Stan<Marcel): {p_stan_better:.4f}")

    yearly = {}
    for yr, grp in merged.groupby("year"):
        m = float(grp["err_marcel"].abs().mean())
        s = float(grp["err_stan"].abs().mean())
        pa_c = float(grp["PA_cov"].mean())
        ip_c = float(grp["IP_cov"].mean())
        yearly[int(yr)] = {
            "marcel": round(m, 3), "stan": round(s, 3),
            "delta": round(s - m, 3), "n": len(grp),
            "avg_PA_cov": round(pa_c, 1), "avg_IP_cov": round(ip_c, 1),
        }
        print(f"    {yr}: Marcel={m:.3f}  Stan={s:.3f}  Δ={s - m:+.3f}  "
              f"PA_cov={pa_c:.1f}%  IP_cov={ip_c:.1f}%")

    summary = {
        "mae_marcel": round(mae_m, 3),
        "mae_stan": round(mae_s, 3),
        "delta_mae": round(mae_s - mae_m, 3),
        "paired_t_p": round(float(t_p), 6),
        "bootstrap_p_stan_better": round(p_stan_better, 4),
        "n_team_years": len(merged),
        "yearly": yearly,
    }
    return summary, merged


# ── Analysis 8: Team prediction anomaly diagnosis ────────────────────────────


def diagnose_team_anomalies(h_df, p_df):
    """Diagnose team prediction anomalies by checking player coverage rates.

    Compares actual team PA/IP totals (all players) vs model PA/IP
    (only players with Marcel predictions + age data).
    Low coverage → distorted RS/RA → bad win predictions.
    """
    saber = pd.read_csv(RAW_DIR / "npb_sabermetrics_2015_2025.csv", encoding="utf-8-sig")
    pitchers_raw = pd.read_csv(RAW_DIR / "npb_pitchers_2015_2025.csv", encoding="utf-8-sig")
    actual = pd.read_csv(
        "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main"
        "/data/projections/pythagorean_2015_2025.csv",
        encoding="utf-8-sig",
    )

    rows = []
    for yr in sorted(h_df["year"].unique()):
        # Actual team PA (all players, no filter)
        yr_saber = saber[saber["year"] == yr]
        actual_pa_team = yr_saber.groupby("team")["PA"].sum()

        # Model PA (only players in LOO-CV results)
        yr_h = h_df[h_df["year"] == yr]
        model_pa_team = yr_h.groupby("team")["actual_PA"].sum()

        # Actual team IP (all pitchers, no filter)
        yr_pitch = pitchers_raw[pitchers_raw["year"] == yr].copy()
        yr_pitch["IP_dec"] = yr_pitch["IP"].apply(ip_to_decimal)
        actual_ip_team = yr_pitch.groupby("team")["IP_dec"].sum()

        # Model IP (only pitchers in LOO-CV results)
        yr_p = p_df[p_df["year"] == yr]
        model_ip_team = yr_p.groupby("team")["actual_IP"].sum()

        # Raw RS/RA from model (before scaling)
        rs_raw = (yr_h.assign(rs=K_WOBA * yr_h["marcel"] * yr_h["actual_PA"])
                  .groupby("team")["rs"].sum())
        ra_raw = (yr_p.assign(ra=yr_p["marcel"] * yr_p["actual_IP"] / 9.0)
                  .groupby("team")["ra"].sum())

        # Actual RS/RA from pythagorean CSV
        yr_act = actual[actual["year"] == yr].set_index("team")

        teams = sorted(
            set(actual_pa_team.index) & set(actual_ip_team.index)
            & set(yr_act.index)
        )
        for team in teams:
            act_pa = int(actual_pa_team.get(team, 0))
            mod_pa = int(model_pa_team.get(team, 0))
            act_ip = float(actual_ip_team.get(team, 0))
            mod_ip = float(model_ip_team.get(team, 0))
            pa_cov = mod_pa / act_pa * 100 if act_pa > 0 else 0
            ip_cov = mod_ip / act_ip * 100 if act_ip > 0 else 0

            act_rs = float(yr_act.loc[team, "RS"]) if "RS" in yr_act.columns else 0
            act_ra = float(yr_act.loc[team, "RA"]) if "RA" in yr_act.columns else 0
            raw_rs = float(rs_raw.get(team, 0))
            raw_ra = float(ra_raw.get(team, 0))

            rows.append({
                "year": yr, "team": team,
                "actual_PA": act_pa, "model_PA": mod_pa,
                "PA_cov": round(pa_cov, 1),
                "actual_IP": round(act_ip, 1), "model_IP": round(mod_ip, 1),
                "IP_cov": round(ip_cov, 1),
                "actual_RS": act_rs, "model_RS_raw": round(raw_rs, 1),
                "actual_RA": act_ra, "model_RA_raw": round(raw_ra, 1),
                "actual_W": int(yr_act.loc[team, "W"]),
                "n_hitters": len(yr_h[yr_h["team"] == team]),
                "n_pitchers": len(yr_p[yr_p["team"] == team]),
            })

    cov_df = pd.DataFrame(rows)

    # ── Print per-year average coverage ──
    print("\n  Per-year average coverage:")
    print(f"  {'Year':>4}  {'PA%':>5}  {'IP%':>5}  {'PA_range':>16}  {'IP_range':>16}  "
          f"{'nH':>3}  {'nP':>3}")
    for yr in sorted(cov_df["year"].unique()):
        sub = cov_df[cov_df["year"] == yr]
        print(f"  {yr:>4}  {sub['PA_cov'].mean():>5.1f}  {sub['IP_cov'].mean():>5.1f}  "
              f"{sub['PA_cov'].min():>5.1f}-{sub['PA_cov'].max():>5.1f}%  "
              f"{sub['IP_cov'].min():>5.1f}-{sub['IP_cov'].max():>5.1f}%  "
              f"{sub['n_hitters'].mean():>3.0f}  {sub['n_pitchers'].mean():>3.0f}")

    # ── Worst PA coverage teams ──
    print("\n  Top 10 worst PA coverage:")
    print(f"  {'Year':>4}  {'Team':<6}  {'ActPA':>5}  {'ModPA':>5}  {'PA%':>5}  "
          f"{'ActIP':>6}  {'ModIP':>6}  {'IP%':>5}  {'W':>3}  "
          f"{'ActRS':>5}  {'RawRS':>5}  {'ActRA':>5}  {'RawRA':>5}")
    worst = cov_df.nsmallest(10, "PA_cov")
    for _, r in worst.iterrows():
        print(f"  {int(r['year']):>4}  {r['team']:<6}  "
              f"{r['actual_PA']:>5}  {r['model_PA']:>5}  {r['PA_cov']:>5.1f}  "
              f"{r['actual_IP']:>6.1f}  {r['model_IP']:>6.1f}  {r['IP_cov']:>5.1f}  "
              f"{r['actual_W']:>3}  "
              f"{r['actual_RS']:>5.0f}  {r['model_RS_raw']:>5.1f}  "
              f"{r['actual_RA']:>5.0f}  {r['model_RA_raw']:>5.1f}")

    # ── Missing hitters for worst 5 teams ──
    print("\n  Missing hitters in worst-coverage teams:")
    for _, r in worst.head(5).iterrows():
        yr, team = int(r["year"]), r["team"]
        yr_team = saber[(saber["year"] == yr) & (saber["team"] == team)]
        yr_team = yr_team[yr_team["PA"] >= 50]  # significant contributors
        model_players = set(h_df[(h_df["year"] == yr) & (h_df["team"] == team)]["player"])
        missing = yr_team[~yr_team["player"].isin(model_players)]
        missing = missing.sort_values("PA", ascending=False)
        if len(missing) > 0:
            total_pa = int(missing["PA"].sum())
            print(f"\n  {yr} {team}: {len(missing)} missing, {total_pa} PA gap")
            for _, m in missing.head(8).iterrows():
                woba = m["wOBA"] if pd.notna(m["wOBA"]) else 0
                print(f"    {m['player']}: {int(m['PA'])} PA, wOBA={woba:.3f}")

    # ── Missing pitchers for worst IP coverage ──
    print("\n  Missing pitchers in worst IP-coverage teams:")
    worst_ip = cov_df.nsmallest(5, "IP_cov")
    for _, r in worst_ip.iterrows():
        yr, team = int(r["year"]), r["team"]
        yr_team = pitchers_raw[(pitchers_raw["year"] == yr)
                               & (pitchers_raw["team"] == team)].copy()
        yr_team["IP_dec"] = yr_team["IP"].apply(ip_to_decimal)
        yr_team = yr_team[yr_team["IP_dec"] >= 20]
        model_players = set(p_df[(p_df["year"] == yr) & (p_df["team"] == team)]["player"])
        missing = yr_team[~yr_team["player"].isin(model_players)]
        missing = missing.sort_values("IP_dec", ascending=False)
        if len(missing) > 0:
            total_ip = round(missing["IP_dec"].sum(), 1)
            print(f"\n  {yr} {team}: {len(missing)} missing, {total_ip} IP gap")
            for _, m in missing.head(8).iterrows():
                era = pd.to_numeric(m["ERA"], errors="coerce")
                era_s = f"{era:.2f}" if pd.notna(era) else "N/A"
                print(f"    {m['player']}: {m['IP_dec']:.1f} IP, ERA={era_s}")

    # ── Coverage vs prediction error correlation ──
    # Merge with team_detail to check if low coverage → bigger errors
    print("\n  Coverage vs prediction error (PA_cov → |err_Marcel|):")
    # Sort by PA coverage and show correlation
    corr_pa = cov_df[["PA_cov", "actual_W"]].copy()
    # We don't have err_Marcel directly here, but we can flag teams
    low = cov_df[cov_df["PA_cov"] < 70]
    high = cov_df[cov_df["PA_cov"] >= 70]
    print(f"    Low coverage (<70%):  {len(low)} team-years, "
          f"avg PA_cov={low['PA_cov'].mean():.1f}%")
    print(f"    High coverage (≥70%): {len(high)} team-years, "
          f"avg PA_cov={high['PA_cov'].mean():.1f}%")

    # Save coverage CSV
    csv_path = OUT_DIR / "team_coverage_diagnosis.csv"
    cov_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved -> {csv_path}")

    return cov_df


# ── Analysis 3: Foreign Player LOO-CV ─────────────────────────────────────────


def run_foreign_loocv():
    """LOO-CV for foreign player Stan v1 model (baseline=league avg vs Stan v1)."""
    # Load foreign player data
    hitter_pairs, pitcher_pairs = [], []
    with open(FOREIGN_DIR / "player_conversion_details.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yr = int(row["npb_first_year"])
            # Hitters
            if (row.get("prev_wOBA") and row.get("npb_first_wOBA")
                    and row.get("wOBA_ratio")):
                try:
                    d = {
                        "year": yr,
                        "name": row.get("english_name", ""),
                        "prev_wOBA": float(row["prev_wOBA"]),
                        "actual": float(row["npb_first_wOBA"]),
                    }
                    for col in ["prev_K_pct", "prev_BB_pct"]:
                        try:
                            d[col] = float(row[col])
                        except (ValueError, KeyError, TypeError):
                            d[col] = None
                    hitter_pairs.append(d)
                except (ValueError, KeyError):
                    pass
            # Pitchers
            if (row.get("prev_ERA") and row.get("npb_first_ERA")
                    and row.get("ERA_ratio")):
                try:
                    d = {
                        "year": yr,
                        "name": row.get("english_name", ""),
                        "prev_ERA": float(row["prev_ERA"]),
                        "actual": float(row["npb_first_ERA"]),
                    }
                    for col in ["prev_FIP", "prev_K_pct", "prev_BB_pct"]:
                        try:
                            d[col] = float(row[col])
                        except (ValueError, KeyError, TypeError):
                            d[col] = None
                    pitcher_pairs.append(d)
                except (ValueError, KeyError):
                    pass

    h_df = pd.DataFrame(hitter_pairs)
    p_df = pd.DataFrame(pitcher_pairs)

    # NPB league averages per year
    saber = pd.read_csv(RAW_DIR / "npb_sabermetrics_2015_2025.csv", encoding="utf-8-sig")
    pitchers_raw = pd.read_csv(RAW_DIR / "npb_pitchers_2015_2025.csv", encoding="utf-8-sig")

    lg_woba = {}
    for yr, grp in saber[saber["PA"] >= 100].groupby("year"):
        lg_woba[yr] = float(grp["wOBA"].mean())

    lg_era = {}
    for yr, grp in pitchers_raw.groupby("year"):
        grp = grp.copy()
        grp["IP_dec"] = grp["IP"].apply(ip_to_decimal)
        grp["ERA"] = pd.to_numeric(grp["ERA"], errors="coerce")
        grp = grp[(grp["IP_dec"] >= 30) & grp["ERA"].notna()]
        if len(grp) > 0:
            lg_era[yr] = float(grp["ERA"].mean())

    h_df["lg_avg"] = h_df["year"].map(lg_woba)
    p_df["lg_avg"] = p_df["year"].map(lg_era)
    h_df = h_df.dropna(subset=["lg_avg"])
    p_df = p_df.dropna(subset=["lg_avg"])

    print(f"  Foreign data: {len(h_df)} hitters, {len(p_df)} pitchers")

    # ── Hitter LOO-CV ──
    h_results = []
    for hold_yr in sorted(h_df["year"].unique()):
        train = h_df[h_df["year"] != hold_yr].copy()
        test = h_df[h_df["year"] == hold_yr].copy()
        if len(test) == 0 or len(train) < 5:
            continue

        # Fill missing K%/BB% with training mean
        for col in ["prev_K_pct", "prev_BB_pct"]:
            tmean = train[col].mean()
            train[col] = train[col].fillna(tmean)
            test[col] = test[col].fillna(tmean)

        feat_cols = ["prev_wOBA", "prev_K_pct", "prev_BB_pct"]
        means = train[feat_cols].mean()
        stds = train[feat_cols].std().replace(0, 1)

        X_train = ((train[feat_cols] - means) / stds).values
        X_test = ((test[feat_cols] - means) / stds).values
        y_train = (train["actual"] - train["lg_avg"]).values

        delta, _ = ridge_fit_predict(X_train, y_train, X_test, ALPHA_FGN_H)
        stan_pred = test["lg_avg"].values + delta

        for i, (_, row) in enumerate(test.iterrows()):
            h_results.append({
                "year": int(hold_yr), "name": row["name"],
                "actual": row["actual"], "baseline": row["lg_avg"],
                "stan": float(stan_pred[i]),
            })

    # ── Pitcher LOO-CV ──
    p_results = []
    for hold_yr in sorted(p_df["year"].unique()):
        train = p_df[p_df["year"] != hold_yr].copy()
        test = p_df[p_df["year"] == hold_yr].copy()
        if len(test) == 0 or len(train) < 5:
            continue

        for col in ["prev_FIP", "prev_K_pct", "prev_BB_pct"]:
            tmean = train[col].mean()
            train[col] = train[col].fillna(tmean)
            test[col] = test[col].fillna(tmean)

        feat_cols = ["prev_ERA", "prev_FIP", "prev_K_pct", "prev_BB_pct"]
        means = train[feat_cols].mean()
        stds = train[feat_cols].std().replace(0, 1)

        X_train = ((train[feat_cols] - means) / stds).values
        X_test = ((test[feat_cols] - means) / stds).values
        y_train = (train["actual"] - train["lg_avg"]).values

        delta, _ = ridge_fit_predict(X_train, y_train, X_test, ALPHA_FGN_P)
        stan_pred = test["lg_avg"].values + delta

        for i, (_, row) in enumerate(test.iterrows()):
            p_results.append({
                "year": int(hold_yr), "name": row["name"],
                "actual": row["actual"], "baseline": row["lg_avg"],
                "stan": float(stan_pred[i]),
            })

    fgn_h = pd.DataFrame(h_results)
    fgn_p = pd.DataFrame(p_results)

    # Summarize
    results = {}
    for name, df in [("hitter", fgn_h), ("pitcher", fgn_p)]:
        err_base = np.abs(df["actual"].values - df["baseline"].values)
        err_stan = np.abs(df["actual"].values - df["stan"].values)
        diff = err_base - err_stan  # positive → Stan better

        n = len(diff)
        mae_base = float(np.mean(err_base))
        mae_stan = float(np.mean(err_stan))

        t_stat, t_p = stats.ttest_rel(err_base, err_stan)
        try:
            w_stat, w_p = stats.wilcoxon(diff)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")

        rng = np.random.default_rng(42)
        boot_means = [np.mean(diff[rng.integers(0, n, n)]) for _ in range(10000)]
        p_stan_better = float(np.mean([m > 0 for m in boot_means]))

        stan_wins = int(np.sum(err_stan < err_base))

        yearly = {}
        for yr in sorted(df["year"].unique()):
            sub = df[df["year"] == yr]
            eb = float(np.abs(sub["actual"].values - sub["baseline"].values).mean())
            es = float(np.abs(sub["actual"].values - sub["stan"].values).mean())
            yearly[int(yr)] = {
                "baseline": round(eb, 4), "stan": round(es, 4),
                "delta": round(es - eb, 4), "n": len(sub),
            }

        results[name] = {
            "n": n,
            "mae_baseline": round(mae_base, 4),
            "mae_stan": round(mae_stan, 4),
            "delta_mae": round(mae_stan - mae_base, 4),
            "paired_t_p": round(float(t_p), 6),
            "wilcoxon_p": round(float(w_p), 6),
            "bootstrap_p_stan_better": round(p_stan_better, 4),
            "stan_win_rate": f"{stan_wins}/{n} ({100 * stan_wins / n:.1f}%)",
            "yearly": yearly,
        }

        print(f"\n  Foreign {name} LOO-CV (n={n}):")
        print(f"    MAE Baseline={mae_base:.4f}  Stan={mae_stan:.4f}  "
              f"Δ={mae_stan - mae_base:+.4f}")
        print(f"    Paired t: p={t_p:.6f}  Wilcoxon: p={w_p:.6f}")
        print(f"    Bootstrap P(Stan<Baseline): {p_stan_better:.4f}")
        print(f"    Stan wins: {stan_wins}/{n} ({100 * stan_wins / n:.1f}%)")
        for yr in sorted(yearly):
            y = yearly[yr]
            print(f"    {yr}: Base={y['baseline']:.4f}  Stan={y['stan']:.4f}  "
                  f"Δ={y['delta']:+.4f}  n={y['n']}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("Step 11: Statistical Validation — Age Curve + Recency Weighting")
    print("=" * 70)

    # ── 1. Japanese LOO-CV (λ=1.0, uniform weighting) ──
    print("\n[1] Japanese Player 8-year LOO-CV (2018-2025) — λ=1.0 (uniform)")
    h_df, p_df, p_fip_df, p_k9_df, p_fip_k9_df = run_jpn_loocv(decay_lambda=1.0)
    print(f"  Collected: {len(h_df)} hitter-years, {len(p_df)} pitcher-years"
          f", {len(p_fip_df)} pitcher-FIP-years"
          f", {len(p_k9_df)} pitcher-ERA+K/9-years"
          f", {len(p_fip_k9_df)} pitcher-FIP+K/9-years")

    # 1a. Player-level significance tests
    print("\n[1a] Player-level significance tests — ALL years")
    player_all = player_level_tests(h_df, p_df, "All 8 years (λ=1.0)")

    # 1b. Team-level MAE
    print("\n[1b] Team-level MAE — ALL years")
    team_all, team_detail = team_level_mae(h_df, p_df, label="All 8 years (λ=1.0)")

    # ── 2. 2021 (COVID) exclusion ──
    print("\n[2] 2021 exclusion — 7-year results")
    h_no21 = h_df[h_df["year"] != 2021]
    p_no21 = p_df[p_df["year"] != 2021]
    years_no21 = [y for y in JPN_YEARS if y != 2021]

    player_no21 = player_level_tests(h_no21, p_no21, "Excluding 2021 (λ=1.0)")
    team_no21, _ = team_level_mae(h_no21, p_no21, years=years_no21,
                                  label="Excluding 2021 (λ=1.0)")

    # ── 3. Foreign player LOO-CV ──
    print("\n[3] Foreign Player Stan v1 LOO-CV (2015-2025)")
    foreign = run_foreign_loocv()

    # ── 4. Recency weighting comparison ──
    print("\n[4] Recency Weighting Comparison")
    recency_results = {}
    for lam in [1.0, 0.9, 0.8]:
        print(f"\n  --- λ = {lam} ---")
        h_lam, p_lam, *_ = run_jpn_loocv(decay_lambda=lam)
        res = player_level_tests(h_lam, p_lam, f"λ={lam}")
        recency_results[str(lam)] = {
            "n_hitters": len(h_lam),
            "n_pitchers": len(p_lam),
            "player_level": res,
        }

    # ── 5. FIP vs ERA comparison ──
    pitcher_fip_results = {}
    if len(p_fip_df) > 0:
        print("\n[5] FIP — Pitcher LOO-CV (2018-2025)")
        pitcher_fip_results = _metric_test(
            p_fip_df, "All 8 years (λ=1.0)", "pitcher_fip")

        # Side-by-side ERA vs FIP summary
        era_res = player_all["pitcher_era"]
        print("\n  ── ERA vs FIP comparison ──")
        print(f"    ERA:  MAE Marcel={era_res['mae_marcel']:.5f}  "
              f"Stan={era_res['mae_stan']:.5f}  "
              f"p={era_res['paired_t_p']:.6f}")
        print(f"    FIP:  MAE Marcel={pitcher_fip_results['mae_marcel']:.5f}  "
              f"Stan={pitcher_fip_results['mae_stan']:.5f}  "
              f"p={pitcher_fip_results['paired_t_p']:.6f}")
    else:
        print("\n[5] FIP — SKIPPED (no FIP data)")

    # ── 6. K/9 + BB/9 feature addition ──
    pitcher_k9_results = {}
    pitcher_fip_k9_results = {}
    if len(p_k9_df) > 0:
        print("\n[6] K/9 + BB/9 — Pitcher LOO-CV (2018-2025)")
        pitcher_k9_results = _metric_test(
            p_k9_df, "All 8 years (λ=1.0)", "pitcher_era+K/9+BB/9")

        if len(p_fip_k9_df) > 0:
            pitcher_fip_k9_results = _metric_test(
                p_fip_k9_df, "All 8 years (λ=1.0)", "pitcher_fip+K/9+BB/9")

        # Side-by-side comparison (all 4 variants)
        era_res = player_all["pitcher_era"]
        print("\n  ── All pitcher model comparison ──")
        print(f"    ERA  (3feat):  p={era_res['paired_t_p']:.6f}  "
              f"MAE Δ={era_res['delta_mae']:+.5f}")
        if pitcher_fip_results:
            print(f"    FIP  (3feat):  p={pitcher_fip_results['paired_t_p']:.6f}  "
                  f"MAE Δ={pitcher_fip_results['delta_mae']:+.5f}")
        print(f"    ERA  (5feat):  p={pitcher_k9_results['paired_t_p']:.6f}  "
              f"MAE Δ={pitcher_k9_results['delta_mae']:+.5f}")
        if pitcher_fip_k9_results:
            print(f"    FIP  (5feat):  p={pitcher_fip_k9_results['paired_t_p']:.6f}  "
                  f"MAE Δ={pitcher_fip_k9_results['delta_mae']:+.5f}")
    else:
        print("\n[6] K/9 + BB/9 — SKIPPED (no data)")

    # ── 7. Team-level detail table (2018-2025) ──
    if len(team_detail) > 0:
        print("\n[7] Team-level detail table (2018-2025)")
        td = team_detail[["year", "team", "W", "W_marcel", "W_stan",
                           "PA_cov", "IP_cov"]].copy()
        td["W_marcel"] = td["W_marcel"].round(1)
        td["W_stan"] = td["W_stan"].round(1)
        td["err_M"] = (td["W_marcel"] - td["W"]).round(1)
        td["err_S"] = (td["W_stan"] - td["W"]).round(1)
        td = td.sort_values(["year", "team"]).reset_index(drop=True)
        td = td.rename(columns={"W": "actual_W"})

        print(f"\n  {'Year':>4}  {'Team':<8}  {'Actual':>6}  {'Marcel':>6}  "
              f"{'Stan':>6}  {'err_M':>6}  {'err_S':>6}  "
              f"{'PA%':>5}  {'IP%':>5}")
        print("  " + "-" * 72)
        for _, row in td.iterrows():
            print(f"  {int(row['year']):>4}  {row['team']:<8}  "
                  f"{int(row['actual_W']):>6}  {row['W_marcel']:>6.1f}  "
                  f"{row['W_stan']:>6.1f}  {row['err_M']:>+6.1f}  "
                  f"{row['err_S']:>+6.1f}  "
                  f"{row['PA_cov']:>5.1f}  {row['IP_cov']:>5.1f}")

        # Per-year MAE summary
        print(f"\n  {'Year':>4}  {'MAE_M':>6}  {'MAE_S':>6}  {'Δ':>6}  "
              f"{'avgPA%':>6}  {'avgIP%':>6}")
        for yr in sorted(td["year"].unique()):
            sub = td[td["year"] == yr]
            mae_m = sub["err_M"].abs().mean()
            mae_s = sub["err_S"].abs().mean()
            pa_c = sub["PA_cov"].mean()
            ip_c = sub["IP_cov"].mean()
            print(f"  {int(yr):>4}  {mae_m:>6.1f}  {mae_s:>6.1f}  "
                  f"{mae_s - mae_m:>+6.1f}  {pa_c:>6.1f}  {ip_c:>6.1f}")

        # Save CSV
        csv_path = OUT_DIR / "team_detail_2018_2025.csv"
        td.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n  Saved -> {csv_path}")

    # ── 8. Team prediction anomaly diagnosis ──
    print("\n[8] Team Prediction Anomaly Diagnosis")
    print("=" * 70)
    coverage_df = diagnose_team_anomalies(h_df, p_df)

    # ── Save results ──
    output = {
        "step": "11b_fip_pitcher_model",
        "japanese_loocv": {
            "years": JPN_YEARS,
            "n_hitters": len(h_df),
            "n_pitchers": len(p_df),
            "features_hitter": ["K_pct", "BB_pct", "BABIP", "age_from_peak"],
            "features_pitcher": ["K_pct", "BB_pct", "age_from_peak"],
            "player_level_all": player_all,
            "team_level_all": team_all,
            "player_level_no2021": player_no21,
            "team_level_no2021": team_no21,
        },
        "pitcher_fip": {
            "n": len(p_fip_df),
            "features": ["K_pct", "BB_pct", "age_from_peak"],
            "player_level": pitcher_fip_results,
        },
        "pitcher_era_k9": {
            "n": len(p_k9_df),
            "features": ["K_pct", "BB_pct", "K_per_9", "BB_per_9", "age_from_peak"],
            "player_level": pitcher_k9_results,
        },
        "pitcher_fip_k9": {
            "n": len(p_fip_k9_df),
            "features": ["K_pct", "BB_pct", "K_per_9", "BB_per_9", "age_from_peak"],
            "player_level": pitcher_fip_k9_results,
        },
        "foreign_loocv": foreign,
        "recency_weighting": recency_results,
        "ridge_alphas": {
            "jpn_hitter": round(ALPHA_JPN_H, 3),
            "jpn_pitcher": round(ALPHA_JPN_P, 3),
            "fgn_hitter": round(ALPHA_FGN_H, 3),
            "fgn_pitcher": round(ALPHA_FGN_P, 3),
        },
    }

    out_path = OUT_DIR / "statistical_validation.json"
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
