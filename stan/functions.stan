functions {
    real correlation_term(
        int[] home_goals, 
        int[] away_goals, 
        vector home_rate, 
        vector away_rate, 
        real tau) {
            // evaluate the correlation term introduced in Dixon & Coles (1997)
            real accum = 0.0;
            int n = size(home_goals);
            for (i in 1:n) {
                if ((home_goals[i] == 0) && (away_goals[i] == 0))
                    accum += log(1.0 - tau * home_rate[i] * away_rate[i]);
                else if ((home_goals[i] == 1) && (away_goals[i] == 0))
                    accum += log(1.0 + tau * away_rate[i]);
                else if ((home_goals[i] == 0) && (away_goals[i] == 1))
                    accum += log(1.0 + tau * home_rate[i]);
                else if ((home_goals[i] == 1) && (away_goals[i] == 1))
                    accum += log(1.0 - tau);
                else
                    continue;
            }
            return accum;
        } 
}