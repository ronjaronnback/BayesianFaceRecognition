data {
  int<lower = 1> N_trials;
  array[4] int<lower = 0, upper = N_trials> ans;
}
parameters {
  real<lower = 0, upper = 1> p;
  real<lower = 0, upper = 1> q;
  real<lower = 0, upper = 1> r;
  
}
transformed parameters {
  simplex[4] theta;
  theta[1] = 1 - p; # NR
  theta[2] = p*q; #C
  theta[3] = p * (1 - q) * (1 - r); #RNN
  theta[4] = p * (1 - q) * r; #TT

}
model {
  target += beta_lpdf(p | 2, 2);
  target += beta_lpdf(q | 2, 2);
  target += beta_lpdf(r | 2, 2);
 
  target += multinomial_lpmf(ans | theta);
}
generated quantities{
    array[4] int pred_ans;
  pred_ans = multinomial_rng(theta, N_trials);
}

