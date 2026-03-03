// Japanese hitter year-ahead wOBA prediction
//
// Model: actual_wOBA = Marcel_wOBA + delta_K * z_K + delta_BB * z_BB + noise
//
// If delta_K = delta_BB = 0, reduces to pure Marcel (baseline).
// K% and BB% are z-scored using the training-set mean/sd.
// Positive delta_BB: high walk rate -> higher wOBA
// Negative delta_K:  high strikeout rate -> lower wOBA

data {
  int<lower=1> N;               // training observations
  vector[N] marcel_woba;        // Marcel-projected wOBA (prior mean)
  vector[N] z_K;                // z-scored K%  (SO / PA)
  vector[N] z_BB;               // z-scored BB% (BB / PA)
  vector[N] actual_woba;        // actual next-year wOBA (target)

  int<lower=0> N_pred;          // test observations
  vector[N_pred] marcel_woba_pred;
  vector[N_pred] z_K_pred;
  vector[N_pred] z_BB_pred;
}

parameters {
  real delta_K;                 // K% correction weight
  real delta_BB;                // BB% correction weight
  real<lower=0> sigma;          // residual std
}

model {
  // Regularized priors: corrections start near zero
  // Prior scale matches typical effect size: 1 z-score of K%/BB% ≈ ±0.010 wOBA
  delta_K  ~ normal(0, 0.05);
  delta_BB ~ normal(0, 0.05);
  sigma    ~ exponential(1);    // wOBA residual scale; exponential(1) avoids sigma→0

  actual_woba ~ normal(
    marcel_woba + delta_K * z_K + delta_BB * z_BB,
    sigma
  );
}

generated quantities {
  // Point predictions for test set (posterior mean)
  vector[N_pred] stan_pred;
  vector[N] log_lik;

  for (i in 1:N_pred) {
    stan_pred[i] = marcel_woba_pred[i]
                   + delta_K  * z_K_pred[i]
                   + delta_BB * z_BB_pred[i];
  }
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(
      actual_woba[n] |
      marcel_woba[n] + delta_K * z_K[n] + delta_BB * z_BB[n],
      sigma
    );
  }
}
