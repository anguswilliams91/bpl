data {
    int<lower=1> nteam;
    int<lower=1> nmatch;
    int home_team[nmatch];
    int away_team[nmatch];
    int home_goals[nmatch];
    int away_goals[nmatch];
}
parameters {
    vector[nteam] log_a_tilde;
    vector[nteam] log_b_tilde;
    real<lower=0> gamma;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    real mu_b;
}
transformed parameters {
    vector[nteam] a = exp(sigma_a * log_a_tilde);
    vector[nteam] b = exp(mu_b + sigma_b * log_b_tilde);
    vector[nmatch] home_rate = a[home_team] .* b[away_team] * gamma;
    vector[nmatch] away_rate = a[away_team] .* b[home_team];
}
model {
    gamma ~ lognormal(0, 1);
    sigma_a ~ normal(0, 1);
    sigma_b ~ normal(0, 1);
    mu_b ~ normal(0, 1);
    log_a_tilde ~ normal(0, 1);
    log_b_tilde ~ normal(0, 1);
    home_goals ~ poisson(home_rate);
    away_goals ~ poisson(away_rate);
}