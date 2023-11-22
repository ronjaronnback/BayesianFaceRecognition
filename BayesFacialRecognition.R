
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
data <-read.csv("data/outcome_data.csv")



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

# make list of our experiment data
exp_list_h <-  list(N_obs = nrow(exp),
                    w_ans = exp$w_ans,
                    N_subj = max(exp$subj),
                    subj = exp$subj,
                    complexity = exp$complexity)
# get STAN model
mpt_hierarch <- stan("HierarchicalFacial.stan", data = exp_list_h)

print(mpt_hierarch,
      pars = c("r", "tau_u_p", "alpha_p", "beta_p", "alpha_q", "beta_q"))

# see if we converged:
traceplot(mpt_hierarch)


# SCRIBBLES AND SCRABBLES ------------------------------------------------------

# plot results -- NEED TO HAVE "TRUE" VALUES to do what the book does in Chap 18.2.4, BUT FROM WHERE?
tau_u_p_true <- 1.1
u_p <- rnorm(N_subj, 0, tau_u_p_true)
p_true2 <- plogis(plogis(p_true) + u_p[exp$subj])#works with book so far

alpha_p_true <- 1
beta_p_true <- 1


alpha_q_true <- 1
beta_q_true <- 1

as.data.frame(mpt_hierarch) %>%
  select(c("tau_u_p", "alpha_p", "beta_p", "alpha_q", "beta_q", "r")) %>%
  mcmc_recover_hist(true = c(tau_u_p,
                             qlogis(p_true), # why q-logis? --> transforms probabilities to quantiles
                             alpha_q, 
                             beta_q, 
                             r_true)
                    )

# redefine p_true probability as function of individual variance
tau_u_p <- 1.1 # assume std of 1.1 for alphas
u_p <- rnorm(N_subj, 0, tau_u_p)
p_true <- plogis(plogis(p_true) + u_p[exp$subj])#works with book so far

# redefine q_true probability as function of fame complexity intercept & slope
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
# TEST POSTERIOR PREDICTIVE CHECK 
# ------------------------------------------------------------------------------

# The argument of the matrix `drop` needs to be set to FALSE,
# otherwise R will simplify the matrix into a vector.
# The two commas in the line below are not a mistake!
draws_par <- as.matrix(mpt_hierarch)[1:500, ,drop = FALSE]

# get generative model
gen_model <- rstan::get_stanmodel(mpt_hierarch)
gen_mix_data <- rstan::gqs(gen_model,
                           data = exp_list_h,
                           draws = draws_par)
# get preds from model
outcome_pred <- extract(gen_mix_data)$pred_w_ans

ppc_stat(exp_list_h$w_ans, # true
         yrep = outcome_pred, # pred
         stat = mean) 


# CURSED - WE HAVE CATEG OUTCOMES, DON'T DO PPC_DENS_OVERLAY
ppc_dens_overlay(y = exp_list_h$w_ans, yrep = outcome_pred[1:100,]) +
  coord_cartesian(xlim = c(1, 5)) 


# ------------------------------------------------------------------------------

