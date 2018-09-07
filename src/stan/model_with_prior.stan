data {
    int<lower=1> nteam;
    int<lower=1> nmatch;
    int<lower=1> nfeat;
    int home_team[nmatch];
    int away_team[nmatch];
    int home_goals[nmatch];
    int away_goals[nmatch];
    matrix[nteam, nfeat] X;
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
}
transformed parameters {
    vector[nteam] mu_a = X * beta_a;
    vector[nteam] mu_b = beta_b_0 + X * beta_b;
    vector[nteam] a = exp(mu_a + sigma_a * log_a_tilde);
    vector[nteam] b = exp(mu_b + sigma_b * log_b_tilde);
    vector[nmatch] home_rate = a[home_team] .* b[away_team] * gamma;
    vector[nmatch] away_rate = a[away_team] .* b[home_team];
}
model {
    beta_a ~ normal(0, 1);
    beta_b ~ normal(0, 1);
    beta_b_0 ~ normal(0, 1);
    sigma_a ~ normal(0, 1);
    sigma_b ~ normal(0, 1);
    log_a_tilde ~ normal(0, 1);
    log_b_tilde ~ normal(0, 1);
    gamma ~ lognormal(0, 1);
    home_goals ~ poisson(home_rate);
    away_goals ~ poisson(away_rate);
}