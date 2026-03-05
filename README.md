# npb-bayes-projection

NPB (Nippon Professional Baseball) player performance projection using Bayesian (Stan) methods.

Successor to [npb-prediction](https://github.com/yasumorishima/npb-prediction)'s Marcel method. Adds **Bayesian corrections** to improve accuracy for both Japanese players and foreign first-year players.

## Results

### Marcel vs Stan Comparison (Team-Level, 2022-2025)

| Model | Team Win MAE | vs Marcel |
|---|---|---|
| Marcel (baseline) | 10.565 wins | — |
| Stan (K%/BB% only) | 10.531 wins | -0.034 |
| **Stan (+BABIP)** | **10.502 wins** | **-0.063 ✓** |

Stan beats Marcel in 3 of 4 backtest years (2022, 2023, 2024).

### Player-Level Accuracy (2022-2025 backtest)

| Model | Marcel MAE | Stan MAE | Improvement |
|---|---|---|---|
| Japanese hitters (wOBA) | 0.0378 | 0.0378 | — (tied) |
| Japanese pitchers (ERA) | 0.9368 | **0.9330** | **-0.4%** |
| Foreign first-year hitters (wOBA) | 0.0337 | **0.0325** | **-3.8%** |
| Foreign first-year pitchers (ERA) | 0.749 | **0.736** | **-1.7%** |

### Key Findings

- **Japanese pitchers**: K% provides signal beyond Marcel ERA (`delta_K = -0.133`). ERA is influenced by luck (BABIP); K% is more stable year-to-year.
- **Japanese hitters**: K%/BB% don't add beyond Marcel wOBA (wOBA already incorporates these). BABIP (luck component) does add signal (`delta_BABIP = -0.006`): high BABIP in t-1 → Marcel overestimates year t.
- **Foreign first-year players**: K%/BB% from previous league add significant signal for both hitters and pitchers.

### Team Backtest (Marcel baseline + PF correction, 2018-2025)

Park factor (PF_5yr) correction is applied before Pythagorean win% calculation to remove stadium bias embedded in Marcel projections.

| Metric | No PF | With PF | Δ |
|---|---|---|---|
| MAE | 6.41 wins | **6.41 wins** | ±0.00 |
| Bias | +2.69 wins | +2.70 wins | +0.01 |
| 80% CI coverage | 86.5% | **87.5%** | **+1.0%** |

PF correction is incorporated into the pipeline. MAE is unchanged; CI coverage improves marginally (+1.0%). Effect expected to grow as Vantelin Dome and Rakuten Mobile Park undergo fan-friendly renovations in 2026.

---

## Approach

### Japanese Players (all NPB players)

```
Hitter:  actual_wOBA = Marcel_wOBA + δ_K·z_K + δ_BB·z_BB + δ_BABIP·z_BABIP + noise
Pitcher: actual_ERA  = Marcel_ERA  + δ_K·z_K + δ_BB·z_BB + noise
```

- Marcel 3-year weighted average (weights 5:4:3) as prior mean
- K%/BB%/BABIP z-scored on training-set statistics
- Training: 2018-2021 | Backtest: 2022-2025

### Foreign First-Year Players

```
Hitter:  npb_wOBA = lg_avg + β_woba·z_woba + β_K·z_K + β_BB·z_BB + noise
Pitcher: npb_ERA  = lg_avg + β_era·z_era + β_fip·z_fip + β_K·z_K + β_BB·z_BB + noise
```

- League conversion factors (MLB→NPB, KBO→NPB, etc.) as prior
- K%/BB%/FIP from previous league as features
- Training: 2015-2019 | Backtest: 2020-2025

### Team Projection

```
team_RS = K_WOBA × Σ_player(Stan_wOBA × actual_PA)
team_RA = Σ_player(Stan_ERA × actual_IP / 9)
W = Pythagorean(RS, RA, exp=1.83)
```

Monte Carlo simulation (N=10,000) propagates player uncertainty to team win distribution.

---

## League-to-NPB Conversion Factors

| League | wOBA ratio [95%CI] (n) | ERA ratio [95%CI] (n) |
|---|---|---|
| **MLB** | 1.235 [1.14, 1.32] (56) | 0.579 [0.51, 0.67] (74) |
| **AAA** | 1.271 [1.01, 1.47] (9) | 0.462 [0.25, 0.69] (6) |

*Interpretation*: MLB hitters need ~24% higher wOBA to match NPB; MLB pitchers ERA inflates ~42% in NPB.

---

## Blog Articles

- [I Added Park Factor Correction to My NPB Bayesian Model — Backtest & 2026 Forecast (DEV.to)](https://dev.to/yasumorishima/i-added-park-factor-correction-to-my-npb-bayesian-prediction-model-backtest-validation-2026-forecast)
- [NPBベイズ順位予測にパークファクター補正を追加した（Zenn）](https://zenn.dev/shogaku/articles/npb-bayes-park-factors)

## Data Sources

- **NPB stats**: [baseball-data.com](https://baseball-data.com/) + [npb.jp](https://npb.jp/)
- **MLB/MiLB stats**: [FanGraphs](https://www.fangraphs.com/) via [pybaseball](https://github.com/jldbc/pybaseball)

## Project Structure

```
npb-bayes-projection/
├── data/
│   ├── raw/              # NPB CSVs (hitters, pitchers, sabermetrics 2015-2025)
│   ├── foreign/          # Foreign player data & conversion factors
│   ├── model/            # Stan model outputs (predictions, comparison JSON)
│   └── projections/      # Team simulation outputs
├── models/
│   ├── hitter_jpn.stan   # Japanese hitter Stan model (Marcel + K%/BB%/BABIP)
│   ├── pitcher_jpn.stan  # Japanese pitcher Stan model (Marcel + K%/BB%)
│   ├── hitter.stan       # Foreign first-year hitter Stan model
│   └── pitcher.stan      # Foreign first-year pitcher Stan model
├── src/
│   ├── stan_jpn_model.py    # Japanese player model runner
│   ├── stan_model.py        # Foreign player model runner
│   ├── team_compare.py      # Marcel vs Stan team-level MAE comparison
│   ├── team_sim.py          # Monte Carlo team standings simulation
│   ├── build_conversion_factors.py
│   └── identify_foreign_players.py
└── .github/workflows/
    └── build_factors.yml    # All pipeline steps via workflow_dispatch
```

## Running via GitHub Actions

```bash
gh workflow run build_factors.yml -f step=run_jpn_model   # Japanese player Stan model
gh workflow run build_factors.yml -f step=run_stan_model  # Foreign player Stan model
gh workflow run build_factors.yml -f step=team_compare    # Marcel vs Stan comparison
gh workflow run build_factors.yml -f step=team_sim        # Team standings simulation
```

## License

MIT
