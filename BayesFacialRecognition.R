
# Loading packages --------------------------------------------------------

library(tidyr)
library(dplyr)
options(mc.cores = parallel::detectCores())
library(bcogsci)
library(extraDistr)
library(bayesplot)
library(ggplot2)
library(rstan)
library(posterior)
rstan_options(auto_write = FALSE)


# Load Data ---------------------------------------------------------------

# just for ronja's laptop shenanigans
#setwd("/Users/ronjaronnback/Documents/GitHub/BayesianFaceRecognition")
#data <-read.csv("data/outcome_data.csv")
data <-read.csv("outcome_data.csv")



# Find probabilities for each branch --------------------------------------
value_counts <- table(data$outcome)
value_counts 

p_true <- .58
q_true <- .66
r_true <- .23

# Declaring Functions ---------------------------------------------------------

NotRecognised <-function(p,q,r) # NR
  1 - p

RecognisedAndNamed <-function(p,q,r) #C
  p * q

RecognisedNotNamed <- function(p,q,r) #RNN
  p * (1 - q) * (1 - r)

TipOfTongue <- function(p,q,r) #TT
  p * (1 - q) * r


# Creating a Dataframe -------------------------------------------------

theta_NR <- NotRecognised(p_true, q_true, r_true)
theta_C <- RecognisedAndNamed(p_true, q_true, r_true)
theta_RNN <- RecognisedNotNamed(p_true, q_true, r_true)
theta_TT <- TipOfTongue(p_true, q_true, r_true)

# generate data vector of probabilities
Theta <- tibble(theta_NR,
    theta_C,
    theta_RNN,
    theta_TT)

# generate values of a multinomial distribution of responses given Theta
N_trials <- 200
(ans <- rmultinom(1, N_trials, c(Theta)))


# ------------------------------------------------------------------------------
# BASE Stan Model --------------------------------------------------------------

data_face <-  list(N_trials = N_trials,
                   ans = c(ans)) 

fit_face <- stan("Facial.stan", data = data_face)


# Model Inspection --------------------------------------------------------
print(fit_face, pars = c("p", "q", "r"))


as.data.frame(fit_face) %>%
  select(c("p","q","r")) %>%
  mcmc_recover_hist(true = c(p_true, q_true, r_true)) +
  coord_cartesian(xlim = c(0, 1))

print(fit_face, pars = c("theta")) # nice & close to the derived "true" values!


# ------------------------------------------------------------------------------
# HIERARCHICAL Stan Model --------------------------------------------------

N_item <- 20 # number trials per subject
N_subj <- 176 # number of subjects
N_obs <- N_item * N_subj 

# make tibble of our real data
(exp <- tibble(subj = data$participant,
               item = data$trial_number,
               complexity = data$TotalPageViews,
               w_ans = data$outcome)) 



# Prior predictive check --------------------------------------------------
# make list of our experiment data
exp_list_h <-  list(onlyprior = 1,
                    N_obs = nrow(exp),
                    w_ans = exp$w_ans,
                    N_subj = max(exp$subj),
                    subj = exp$subj,
                    complexity = exp$complexity)

mpt_hierarch <- stan("HierarchicalFacial.stan", data = exp_list_h,
                     control = list(adapt_delta = 0.9))


print(mpt_hierarch,
      pars = c("r", "tau_u_p", "alpha_p", "alpha_q", "beta_q", "beta_p"
              ))

as.data.frame(mpt_hierarch) %>%
  select(c( "alpha_p","alpha_q",  "r")) %>%
  mcmc_recover_hist(true = c( # might have to use what's below, ubt that's just the simulated data from the book so idk
    
    qlogis(p_true),
    qlogis(q_true),
    
    r_true)
  )  
#Plot generated outcomes
ypred_w_ans <- rstan::extract(mpt_hierarch)$pred_w_ans
ypred_w_ans_samples <- ypred_w_ans[1:50, ]
categories <- table(ypred_w_ans_samples)

barplot_result <- barplot(categories, main = "Outcomes - Prior Predictive Check", xlab = "Categories", ylab = "Frequency")
text(x = barplot_result, y = categories + 1, labels = c("NR", "C", "RNN", "ToT"), pos = 1)

#plot outcomes sampled from dataset
y_outcomes_data <- data$outcome[1:50]
categories_data <- table(y_outcomes_data)


barplot_result <- barplot(categories, main = "Outcomes - From Data", xlab = "Categories", ylab = "Frequency")
text(x = barplot_result, y = categories + 1, labels = c("NR", "C", "RNN", "ToT"), pos = 1)





# Train model on data -----------------------------------------------------

# make list of our experiment data
exp_list_h <-  list(onlyprior = 0,
                    N_obs = nrow(exp),
                    w_ans = exp$w_ans,
                    N_subj = max(exp$subj),
                    subj = exp$subj,
                    complexity = exp$complexity)

mpt_hierarch_posterior <- stan("HierarchicalFacial.stan", data = exp_list_h,
                     control = list(adapt_delta = 0.9))
print(mpt_hierarch,
      pars = c("r", "tau_u_p", "alpha_p", "alpha_q", "beta_q", "beta_p"
      ))

print(mpt_hierarch,
      pars = c("r", "tau_u_p", "alpha_p", "alpha_q", "beta_q"))



# plot results
as.data.frame(mpt_hierarch) %>%
  select(c( "alpha_p","alpha_q",  "r")) %>%
  mcmc_recover_hist(true = c( # might have to use what's below, ubt that's just the simulated data from the book so idk
                             
                              qlogis(p_true),
                              qlogis(q_true),
                              
                             r_true)
                    )  



# redefine p_true probability as function of individual variance
tau_u_p <- 1.1 # assume std of 1.1 for alphas
u_p <- rnorm(N_subj, 0, tau_u_p)
p_true <- plogis(plogis(p_true) + u_p[exp$subj])#works with book so far

# redefine q_true probability as function of fame complexity interept & slope
alpha_q <- .6
beta_q <- .2
q_true <- plogis(alpha_q + exp$complexity * beta_q)

# continue with the probabilities
theta_NR_hierarch  <- NotRecognised(p_true, q_true, r_true)
theta_C_hierarch   <- RecognisedAndNamed(p_true, q_true, r_true)
theta_RNN_hierarch <- RecognisedNotNamed(p_true, q_true, r_true)
theta_TT_hierarch  <- TipOfTongue(p_true, q_true, r_true)

theta_hierarch <- matrix(
  c(theta_NR_hierarch,
    theta_C_hierarch,
    theta_RNN_hierarch,
    theta_TT_hierarch),
  ncol = 4)
dim(theta_hierarch)







# ------------------------------------------------------------------------------
# ADDED COMPLEXITY Stan Model --------------------------------------------------

N_obs <- 50

theta_NR_v <- rep(NotRecognised(p_true, q_true, r_true), N_obs)
theta_C_v <- RecognisedAndNamed(p_true, q_true, r_true)
theta_RNN_v <- RecognisedNotNamed(p_true, q_true, r_true)
theta_RNN_v <- TipOfTongue(p_true, q_true, r_true)

theta_item <- matrix(c(theta_NR_v,
                       theta_C_v,
                       theta_RNN_v,
                       theta_RNN_v),
                     ncol = 4)
dim(theta_item)

