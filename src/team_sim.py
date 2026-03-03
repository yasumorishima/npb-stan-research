"""
Step 6: Team standings simulation with posterior uncertainty propagation.

Algorithm:
  1. Load Marcel 2026 hitter/pitcher projections from npb-prediction (GitHub raw)
  2. For each player: add Gaussian noise (Marcel MAE as sigma) -> N_SIM draws of OPS/ERA
  3. Aggregate to team RS (via K_HIT calibration) and RA (via ERA x IP/9)
  4. Apply Pythagorean expectation -> simulated win totals
  5. Rank teams within each league -> P(pennant), P(CS), 80% CI for wins

Output:
  data/projections/team_sim_2026.json
  data/projections/team_sim_2026.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "projections"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Simulation settings ────────────────────────────────────────────────────────
N_SIM = 10_000
NPB_GAMES = 143
NPB_PYTH_EXP = 1.83          # Pythagorean exponent for NPB
NPB_TARGET_IP = 1_287        # 143 games x 9 innings = total IP per team season

# ── Uncertainty parameters ─────────────────────────────────────────────────────
# sigma from Marcel backtest (npb-prediction 2025 backtest):
SIGMA_OPS = 0.063            # Marcel OPS MAE -> per-player sigma
SIGMA_ERA = 0.78             # Marcel ERA MAE -> per-player sigma

# ── RS calibration ─────────────────────────────────────────────────────────────
# Calibrated from: NPB historical avg RS=535, avg team weighted OPS*PA=4039
# RS ≈ K_HIT * sum_i(OPS_i * PA_i)
K_HIT = 0.1326

# ── League structure ───────────────────────────────────────────────────────────
LEAGUES: dict[str, list[str]] = {
    "CL": ["\u962a\u795e", "\u5e83\u5cf6", "DeNA", "\u5de8\u4eba", "\u4e2d\u65e5", "\u30e4\u30af\u30eb\u30c8"],
    "PL": ["\u30bd\u30d5\u30c8\u30d0\u30f3\u30af", "\u65e5\u672c\u30cf\u30e0", "\u697d\u5929",
           "\u30aa\u30ea\u30c3\u30af\u30b9", "\u30ed\u30c3\u30c6", "\u897f\u6b66"],
}
# 阪神, 広島, DeNA, 巨人, 中日, ヤクルト
# ソフトバンク, 日本ハム, 楽天, オリックス, ロッテ, 西武

CS_SPOTS = 3  # top-3 qualify for Climax Series

# ── Data source: npb-prediction (public repo) ──────────────────────────────────
NPBP_BASE = (
    "https://raw.githubusercontent.com/yasumorishima/npb-prediction/main/data/projections"
)


def load_marcel() -> tuple[pd.DataFrame, pd.DataFrame]:
    hitters = pd.read_csv(f"{NPBP_BASE}/marcel_hitters_2026.csv", encoding="utf-8-sig")
    pitchers = pd.read_csv(f"{NPBP_BASE}/marcel_pitchers_2026.csv", encoding="utf-8-sig")
    return hitters, pitchers


def normalize_pitcher_ip(pitchers: pd.DataFrame) -> pd.DataFrame:
    """Scale each team's total projected IP to NPB_TARGET_IP (= 143 * 9 innings)."""
    pitchers = pitchers.copy()
    for team, grp in pitchers.groupby("team"):
        total_ip = grp["IP"].sum()
        if total_ip > 0:
            pitchers.loc[pitchers["team"] == team, "IP"] *= NPB_TARGET_IP / total_ip
    return pitchers


def simulate(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
    n_sim: int = N_SIM,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Run Monte Carlo simulation. Returns dict: team -> (n_sim,) win totals."""
    rng = np.random.default_rng(seed)

    # Hitters: sample OPS draws (n_players x n_sim)
    ops = hitters["OPS"].values[:, None]   # (P, 1)
    pa  = hitters["PA"].values[:, None]    # (P, 1)
    ops_sim = np.clip(ops + rng.normal(0, SIGMA_OPS, size=(len(hitters), n_sim)), 0.250, 1.200)

    # Pitchers: sample ERA draws (n_pitchers x n_sim)
    era = pitchers["ERA"].values[:, None]  # (Q, 1)
    ip  = pitchers["IP"].values[:, None]   # (Q, 1)
    era_sim = np.clip(era + rng.normal(0, SIGMA_ERA, size=(len(pitchers), n_sim)), 0.50, 12.0)

    # Aggregate per team
    h_teams = hitters["team"].values
    p_teams = pitchers["team"].values
    all_teams = sorted(set(hitters["team"].unique()) | set(pitchers["team"].unique()))

    wins_sim: dict[str, np.ndarray] = {}
    for team in all_teams:
        h_mask = h_teams == team
        p_mask = p_teams == team
        if not h_mask.any() or not p_mask.any():
            continue

        rs = K_HIT * (ops_sim[h_mask] * pa[h_mask]).sum(axis=0)
        ra = (era_sim[p_mask] * ip[p_mask]).sum(axis=0) / 9.0

        rs_exp = np.power(np.clip(rs, 1.0, None), NPB_PYTH_EXP)
        ra_exp = np.power(np.clip(ra, 1.0, None), NPB_PYTH_EXP)
        wpct   = rs_exp / (rs_exp + ra_exp)
        wins_sim[team] = wpct * NPB_GAMES

    return wins_sim


def compute_probabilities(wins_sim: dict[str, np.ndarray]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for lg_name, lg_teams in LEAGUES.items():
        lg_sim = {t: wins_sim[t] for t in lg_teams if t in wins_sim}
        if not lg_sim:
            continue

        teams = list(lg_sim.keys())
        win_matrix = np.stack([lg_sim[t] for t in teams], axis=1)   # (n_sim, T)
        # Double argsort trick: ranks[i, j] = rank of team j in simulation i (1=best)
        ranks = (-win_matrix).argsort(axis=1).argsort(axis=1) + 1

        for i, team in enumerate(teams):
            w = win_matrix[:, i]
            r = ranks[:, i]
            results[team] = {
                "league":      lg_name,
                "p_pennant":   float((r == 1).mean()),
                "p_cs":        float((r <= CS_SPOTS).mean()),
                "median_wins": float(np.median(w)),
                "mean_wins":   float(w.mean()),
                "wins_80ci":   [float(np.percentile(w, 10)), float(np.percentile(w, 90))],
            }
    return results


def main(n_sim: int = N_SIM) -> None:
    print("Loading Marcel 2026 projections from npb-prediction...")
    hitters, pitchers = load_marcel()
    print(f"  Hitters : {len(hitters):3d} players / {hitters['team'].nunique()} teams")
    print(f"  Pitchers: {len(pitchers):3d} players / {pitchers['team'].nunique()} teams")

    pitchers = normalize_pitcher_ip(pitchers)

    print(f"Running {n_sim:,} simulations...")
    wins_sim = simulate(hitters, pitchers, n_sim)

    results = compute_probabilities(wins_sim)

    # Print results table
    for lg in ["CL", "PL"]:
        ranked = sorted(
            [(t, v) for t, v in results.items() if v["league"] == lg],
            key=lambda x: -x[1]["median_wins"],
        )
        print(f"\n-- {lg} -----------------------------------------------")
        print(f"{'Team':14s}  {'P(Pennant)':>10s}  {'P(CS)':>7s}  {'Median W':>8s}  80% CI")
        for t, v in ranked:
            lo, hi = v["wins_80ci"]
            print(
                f"{t:14s}  {v['p_pennant']:9.1%}  {v['p_cs']:6.1%}"
                f"  {v['median_wins']:7.1f}  [{lo:.1f}, {hi:.1f}]"
            )

    # Save JSON
    json_path = OUT_DIR / "team_sim_2026.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved -> {json_path}")

    # Save CSV
    rows = []
    for team, v in results.items():
        lo, hi = v["wins_80ci"]
        rows.append({
            "team":         team,
            "league":       v["league"],
            "p_pennant":    round(v["p_pennant"], 4),
            "p_cs":         round(v["p_cs"], 4),
            "median_wins":  round(v["median_wins"], 1),
            "wins_80ci_lo": round(lo, 1),
            "wins_80ci_hi": round(hi, 1),
        })
    (
        pd.DataFrame(rows)
        .sort_values(["league", "median_wins"], ascending=[True, False])
        .to_csv(OUT_DIR / "team_sim_2026.csv", index=False, encoding="utf-8-sig")
    )
    print(f"Saved -> {OUT_DIR / 'team_sim_2026.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sim", type=int, default=N_SIM)
    args = parser.parse_args()
    main(args.n_sim)
