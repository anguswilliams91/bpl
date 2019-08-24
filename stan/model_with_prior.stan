#include stan/functions.stan

data {
    int<lower=1> nteam;
    int<lower=1> nmatch;
    int<lower=1> nfeat;
    int home_team[nmatch];
    int away_team[nmatch];
    int home_goals[nmatch];
    int away_goals[nmatch];
    matrix[nteam, nfeat] X;

    // prior parameters for tau and rho
    real rho_prior_mean;
    real<lower=0> rho_prior_sigma;
    real<lower=0> tau_prior_alpha;
    real<lower=0> tau_prior_beta;
}
parameters {
    vector[nteam] log_a_tilde;
    vector[nteam] log_b_tilde;
    real<lower=0> gamma;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    real beta_b_0;
    vector[nfeat] beta_a;
    vector[nfeat] beta_b;
    real<lower=0, upper=1> u;
    real<lower=-0.5, upper=0.5> tau;
}
transformed parameters {
    real<lower=-1, upper=1> rho = 2 * u - 1;
    vector[nteam] a = exp(X * beta_a + sigma_a * log_a_tilde);
    vector[nteam] b = exp(beta_b_0 + X * beta_b + sigma_b * log_b_tilde);
}
model {
    vector[nmatch] home_rate = a[home_team] .* b[away_team] * gamma;
    vector[nmatch] away_rate = a[away_team] .* b[home_team];
    tau ~ normal(rho_prior_mean, rho_prior_sigma);
    u ~ beta(tau_prior_alpha, tau_prior_beta);
    beta_a ~ normal(0, 1);
    beta_b ~ normal(0, 1);
    beta_b_0 ~ normal(0, 1);
    sigma_a ~ normal(0, 1);
    sigma_b ~ normal(0, 1);
    gamma ~ lognormal(0, 1);
    log_a_tilde ~ normal(0, 1);
    log_b_tilde ~ normal(rho * log_a_tilde, sqrt(1 - square(rho)));
    home_goals ~ poisson(home_rate);
    away_goals ~ poisson(away_rate);
    target += correlation_term(home_goals, away_goals, home_rate, away_rate, tau);
}