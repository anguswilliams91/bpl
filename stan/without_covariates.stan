#include stan/functions.stan

data {
    int<lower=1> nteam;
    int<lower=1> nmatch;
    int home_team[nmatch];
    int away_team[nmatch];
    int home_goals[nmatch];
    int away_goals[nmatch];

    // prior parameters for tau and rho
    real rho_prior_mean;
    real<lower=0> rho_prior_sigma;
    real<lower=0> tau_prior_alpha;
    real<lower=0> tau_prior_beta;
}
parameters {
    vector[nteam] log_a_tilde;
    vector[nteam] log_b_tilde;
    vector[nteam] log_gamma_tilde;
    real<lower=0> sigma_a;
    real<lower=0> sigma_b;
    real<lower=0> sigma_log_gamma;
    real mu_log_gamma;
    real mu_b;
    real<lower=0, upper=1> u;
    real<lower=-0.5, upper=0.5> tau;
}
transformed parameters {
    real<lower=-1, upper=1> rho = 2 * u - 1;
    vector[nteam] log_a = sigma_a * log_a_tilde;
    vector[nteam] log_b = mu_b + sigma_b * log_b_tilde;
    vector[nteam] log_gamma = mu_log_gamma + sigma_log_gamma * log_gamma_tilde;
}
model {
    vector[nmatch] home_rate = log_a[home_team] + log_b[away_team] + log_gamma[home_team];
    vector[nmatch] away_rate = log_a[away_team] + log_b[home_team];
    tau ~ normal(rho_prior_mean, rho_prior_sigma);
    u ~ beta(tau_prior_alpha, tau_prior_beta);
    sigma_a ~ std_normal();
    sigma_b ~ std_normal();
    sigma_log_gamma ~ std_normal();
    mu_b ~ std_normal();
    mu_log_gamma ~ std_normal();
    log_a_tilde ~ std_normal();
    log_b_tilde ~ normal(rho * log_a_tilde, sqrt(1 - square(rho)));
    log_gamma_tilde ~ std_normal();
    home_goals ~ poisson_log(home_rate);
    away_goals ~ poisson_log(away_rate);
    target += correlation_term(home_goals, away_goals, exp(home_rate), exp(away_rate), tau);
}