# npb-bayes-projection

NPB (Nippon Professional Baseball) player performance projection using Bayesian methods.

Successor to [npb-prediction](https://github.com/yasumorishima/npb-prediction)'s Marcel method, addressing the "new foreign player" problem: when a player has zero NPB history, Marcel defaults to league average. This project builds **league-to-NPB conversion factors** to provide better priors for foreign player projections.

## Project Status

**Step 1: League-to-NPB Conversion Factors** (in progress)

## Approach

### Step 1: Conversion Factors
1. Identify foreign players in NPB (2015-2025) using katakana name detection + profile data
2. Map each player to their previous league (MLB, KBO, CPBL, etc.) and fetch pre-NPB stats
3. Calculate conversion factors: `NPB_first_year_metric / prev_league_metric`
4. Aggregate by origin league with bootstrap 95% CI

### Future Steps
- Step 2: PyMC Bayesian model with conversion-factor-informed priors
- Step 3: Full team projection with uncertainty quantification

## Data Sources

- **NPB stats**: [baseball-data.com](https://baseball-data.com/) + [npb.jp](https://npb.jp/)
- **MLB stats**: [FanGraphs](https://www.fangraphs.com/) via [pybaseball](https://github.com/jldbc/pybaseball)

## Usage

```bash
# Step 1a: Identify foreign players (generates candidate CSV)
python src/identify_foreign_players.py

# Step 1b: After manually curating the master CSV and collecting prev-league stats:
python src/build_conversion_factors.py
```

## Project Structure

```
npb-bayes-projection/
├── data/
│   ├── raw/              # NPB CSVs (from npb-prediction)
│   ├── foreign/          # Foreign player data & conversion factors
│   └── projections/      # Future Bayesian projection outputs
├── src/
│   ├── identify_foreign_players.py   # Extract foreign players from NPB data
│   └── build_conversion_factors.py   # Compute conversion factors
├── notebooks/            # Exploration notebooks
├── tests/
├── .github/workflows/
│   └── build_factors.yml  # CI: run conversion factor pipeline
├── requirements.txt
└── README.md
```

## License

MIT
