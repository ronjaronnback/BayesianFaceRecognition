//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// The input data is a vector 'y' of length 'N'.

data {
  int<lower = 1> N_obs;
  array[N_obs] int<lower = 1, upper = 5> w_ans;
  array[N_obs] real complexity;
  int<lower = 1> N_subj;
  array[N_obs] int<lower = 1, upper = N_subj> subj;
}
parameters {
  //real<lower = 0, upper = 1> p; // taken over by hierarchical
  //real<lower = 0, upper = 1> q; // taken over by complexity
  real<lower = 0, upper = 1> r;
  // p parameter stuff ("Recognised (p) or not (p-1)")
  real alpha_p;
  real<lower = 0> tau_u_p;
  vector[N_subj] u_p;
  // q parameter stuff ("Named (q) or Not (1-q)")
  real alpha_q;
  real beta_q;
}
transformed parameters {
  array[N_obs] simplex[4] theta;
  for (n in 1:N_obs){
    // alpha_a is the "average" of all subjects, u_a the individual divergences
    real p = inv_logit(alpha_p + u_p[subj[n]]); // adds individual differences
    real q = inv_logit(alpha_q + complexity[n] * beta_q); // adds complexity

    // Not Recognised
    theta[n,1] = 1 - p;
    // Correct
    theta[n,2] = p*q;
    // Recognised Not Named
    theta[n,3] = p * (1 - q) * (1 - r);
    // Recognised Not Named, but Tip of The Tongue
    theta[n,4] = p * (1 - q) * r;
  }
}
model {
  target += beta_lpdf(r | 2, 2); // no item or subject variation, so wide beta 
  target += normal_lpdf(alpha_p | 0, 1.5); //
  target += normal_lpdf(alpha_q | 0, 1.5); //
  target += normal_lpdf(beta_q | 0, 1); //
  target += normal_lpdf(u_p | 0, tau_u_p); //
  target += normal_lpdf(tau_u_p | 0, 1) - normal_lccdf(0 | 0, 1); // truncated
  for(n in 1:N_obs)
    target +=  categorical_lpmf(w_ans[n] | theta[n]);
}
generated quantities{
  array[N_obs] int<lower = 1, upper = 4> pred_w_ans;
  for(n in 1:N_obs)
    pred_w_ans[n] = categorical_rng(theta[n]);
}

