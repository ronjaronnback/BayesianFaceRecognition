
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
library(tictoc)
rstan_options(auto_write = FALSE)

# define min-max scaling function for later
min_max_scale <- function(x){(x-min(x))/(max(x) - min(x))}

# Load Data ---------------------------------------------------------------

# just for ronja's laptop shenanigans
setwd("/Users/ronjaronnback/Documents/GitHub/BayesianFaceRecognition")

#data <-read.csv("data/outcome_data.csv")
#data <-read.csv("outcome_data.csv")

#setwd("C:/Users/louis/Documents/University/Master (TiU)/Year 2/Courses/BayesModels/Group_assign/BayesianFaceRecognition")
data <-read.csv("data/outcome_data.csv")

# center and min-max scale data
data$C_TotalPageViews <- min_max_scale(data$TotalPageViews) - 0.5

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

TipOfTongue <- function(p,q,r) #TT
  p * (1 - q) * r

RecognisedNotNamed <- function(p,q,r) #RNN
  p * (1 - q) * (1 - r)

# ==============================================================================
# BASE Stan Model 
# ==============================================================================

# Creating a Dataframe -------------------------------------------------

theta_NR <- NotRecognised(p_true, q_true, r_true)
theta_C <- RecognisedAndNamed(p_true, q_true, r_true)
theta_TT <- TipOfTongue(p_true, q_true, r_true)
theta_RNN <- RecognisedNotNamed(p_true, q_true, r_true)

# generate data vector of probabilities
Theta <- tibble(theta_NR,
    theta_C,
    theta_TT,
    theta_RNN)

# generate values of a multinomial distribution of responses given Theta
N_trials <- 200
(ans <- rmultinom(1, N_trials, c(Theta)))

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

# ==============================================================================
# HIERARCHICAL Stan Model 
# ==============================================================================

N_item <- 20 # number trials per subject
N_subj <- 176 # number of subjects
N_obs <- N_item * N_subj 

# HIERARCHICAL Stan Model WITH SIMULATED DATA ----------------------------------

subj <- rep(1:N_subj, each = N_item)
trial_number <- rep(1:N_item, time = N_subj)
# make approximate distribution for complexity (fameousness) and center it
complexity <- min_max_scale(rep(rlnorm(N_item, meanlog = 0, sdlog = 1), 
                                      times = N_subj)) - 0.5

# define simulated true parameters
p_true <- 0.58
q_true <- 0.66
r_true <- 0.23

tau_u <- c(1.1,1.1)
u <- matrix(nrow = N_subj,ncol = 2)
u[,1:2] <- c(rnorm(N_subj, 0, tau_u[1]), rnorm(N_subj, 0, tau_u[2]))

alpha_p <- 0.5
beta_p <- 1
p_true <- plogis(alpha_p + u[subj,1] + complexity * beta_p)

alpha_q <- 0
beta_q <- 1
q_true <- plogis(alpha_q + u[subj,2] + complexity * beta_q)

# generate data vector of probabilities
theta_h <- matrix(
  c(NotRecognised(p_true, q_true, r_true),
    RecognisedAndNamed(p_true, q_true, r_true),
    TipOfTongue(p_true, q_true, r_true),
    RecognisedNotNamed(p_true, q_true, r_true)),
  ncol = 4)

# generate values of a categorical distribution of responses given Theta
ans <- rcat(N_obs,theta_h)

# make tibble of our real data
sim_exp <- tibble(subj = subj,
                   item = trial_number,
                   complexity = complexity,
                   w_ans = ans)

# make list of our experiment data
sim_exp_list <-  list(onlyprior = 0,
                      N_obs = nrow(sim_exp),
                      w_ans = sim_exp$w_ans,
                      N_subj = max(sim_exp$subj),
                      subj = sim_exp$subj,
                      complexity = sim_exp$complexity)
# get STAN model
mpt_hierarch_sim <- stan("HFR_Improved.stan", data = sim_exp_list)

print(mpt_hierarch_sim,
      pars = c("r", "tau_u", "alpha_p", "beta_p", "alpha_q", "beta_q"))
# OUT:
#          mean se_mean   sd  2.5%  25%  50%  75% 97.5% n_eff Rhat
#r        0.22       0 0.01  0.19  0.21 0.22 0.22  0.24  6073    1
# tau_u[1] 1.13       0 0.08  0.98  1.08 1.13 1.19  1.30  1167    1
# tau_u[2] 1.02       0 0.09  0.85  0.95 1.02 1.08  1.21  1294    1
# alpha_p  0.59       0 0.11  0.39  0.52 0.59 0.67  0.81  1109    1
# beta_p   1.21       0 0.17  0.89  1.09 1.21 1.33  1.54  5333    1
# alpha_q  0.02       0 0.11 -0.20 -0.05 0.02 0.09  0.23  1705    1
# beta_q   0.86       0 0.19  0.48  0.73 0.86 1.00  1.23  5250    1
# parameter recovery pretty good!

# see if we converged: Looks good!
traceplot(mpt_hierarch_sim) +
  ggtitle("Traceplot of Hierarchical Model on Simulated Data") + 
  theme(plot.title = element_text(size = 12))

# posterior predictive checks for simulated data
as.data.frame(mpt_hierarch_sim) %>%
  select(r, alpha_p, beta_p, alpha_q, beta_q) %>%
  mcmc_recover_hist(true = c(r_true, alpha_p, beta_p, alpha_q, beta_q)) +
  ggtitle("Posterior Predictive Distributions of the Hierarchical Model \non Simulated Data") + 
  theme(plot.title = element_text(size = 12))

remove(mpt_hierarch_sim)
remove(sim_exp)
remove(sim_exp_list)

# Prior predictive check with REAL DATA ----------------------------------------

# make tibble of our real data
(exp <- tibble(subj = data$participant,
               item = data$trial_number,
               complexity = data$C_TotalPageViews,
               w_ans = data$outcome))

# make list of our experiment data
exp_list_h <-  list(onlyprior = 1,
                    N_obs = nrow(exp),
                    w_ans = exp$w_ans,
                    N_subj = max(exp$subj),
                    subj = exp$subj,
                    complexity = exp$complexity)

mpt_hierarch_prior <- stan("HFR_Improved.stan", data = exp_list_h,
                     control = list(adapt_delta = 0.9))


print(mpt_hierarch_prior,
      pars = c("r", "tau_u", "alpha_p", "beta_p", "alpha_q", "beta_q"))

#          mean se_mean   sd  2.5%   25%   50%  75% 97.5% n_eff Rhat


# see if we converged:
traceplot(mpt_hierarch_prior) +
  ggtitle("Traceplot of Prior Hierarchical Model on Actual Data")


# HIERARCHICAL Stan Model WITH REAL DATA ---------------------------------------

# make tibble of our real data
(exp <- tibble(subj = data$participant,
               item = data$trial_number,
               complexity = data$C_TotalPageViews,
               w_ans = data$outcome)) 

# make list of our experiment data
exp_list_h <-  list(onlyprior = 0,
                    N_obs = nrow(exp),
                    w_ans = exp$w_ans,
                    N_subj = max(exp$subj),
                    subj = exp$subj,
                    complexity = exp$complexity)
# get STAN model
mpt_hierarch <- stan("HFR_Improved.stan", data = exp_list_h)

print(mpt_hierarch,
      pars = c("r", "tau_u", "alpha_p", "beta_p", "alpha_q", "beta_q"))
# OUT:
#          mean se_mean   sd 2.5%  25%  50%  75% 97.5% n_eff Rhat
# r        0.23    0.00 0.02 0.20 0.22 0.23 0.24  0.27  5385    1
# tau_u[1] 0.82    0.00 0.07 0.69 0.77 0.81 0.86  0.95  1600    1
# tau_u[2] 0.94    0.00 0.09 0.76 0.87 0.93 0.99  1.13  1548    1
# alpha_p  2.51    0.00 0.13 2.26 2.42 2.51 2.59  2.77  1566    1
# beta_p   6.90    0.01 0.31 6.31 6.69 6.90 7.12  7.50  2401    1
# alpha_q  1.88    0.00 0.13 1.64 1.80 1.88 1.97  2.14  1736    1
# beta_q   5.02    0.01 0.32 4.40 4.81 5.02 5.24  5.66  2599    1

# see if we converged:
traceplot(mpt_hierarch) +
  ggtitle("Traceplot of Hierarchical Model on Actual Data") + 
  theme(plot.title = element_text(size = 12))


# TEST POSTERIOR PREDICTIVE CHECK ----------------------------------------------

# another attempt at posterior predictive checks here 
as.data.frame(mpt_hierarch) %>%
  select(r, alpha_p, beta_p, alpha_q, beta_q, tau_u_p) %>%
  mcmc_recover_hist(true = c(r_true, alpha_p_true, beta_p_true, 
                             alpha_q_true, beta_q_true, tau_u_p_true))

# bar plot as posterior predictive check
gen_data <- rstan::extract(mpt_hierarch)$pred_w_ans
ppc_bars(exp$w_ans, gen_data) +
  ggtitle ("Hierarchical model") 

# same but grouped by subject, doesn't really look like anything now with this many subjects
ppc_bars_grouped(exp$w_ans, 
                 gen_data, group = exp$subj) +
  ggtitle ("By-subject plot for the hierarchical model")




# ==============================================================================
# POSTERIOR CHECKS FOR HIERARCHICAL 2
# ==============================================================================

N_item <- 20 # number trials per subject
N_subj <- 176 # number of subjects
N_obs <- N_item * N_subj 
# HIERARCHICAL Stan Model WITH SIMULATED DATA ----------------------------------
subj <- rep(1:N_subj, each = N_item)
trial_number <- rep(1:N_item, time = N_subj)
# make approximate distribution for complexity (fameousness) and center it
complexity <- min_max_scale(rep(rlnorm(N_item, meanlog = 0, sdlog = 1), 
                                times = N_subj)) - 0.5
# define simulated true parameters
p_true <- 0.58
q_true <- 0.66
r_true <- 0.23

# tau_u_p <- 1.1
# u_p <- rnorm(N_subj, 0, tau_u_p)
# p_true_u <- qlogis(p_true) + u_p[subj]
# alpha_p <- 0.5
# beta_p <- 1
# p_true <- plogis(alpha_p + p_true_u + complexity*beta_p)
# 
# tau_u_q <- 1.1
# u_q <- rnorm(N_subj, 0, tau_u_q)
# q_true_u <- qlogis(q_true) + u_q[subj]
# alpha_q <- 0.5
# beta_q <- 1
# q_true <- plogis(alpha_q + q_true_u + complexity * beta_q)

# tau_u_p <- 1.1
# u_p <- rnorm(N_subj, 0, tau_u_p)
# #p_true_u <- qlogis(p_true) + u_p[subj]
# alpha_p <- 0.5
# beta_p <- 1
# p_true <- plogis(qlogis(alpha_p) + u_p[subj] + complexity*beta_p)
# 
# tau_u_q <- 1.1
# u_q <- rnorm(N_subj, 0, tau_u_q)
# #q_true_u <- qlogis(q_true) + u_q[subj]
# alpha_q <- 0.5
# beta_q <- 1
# q_true <- plogis(qlogis(alpha_q)  + u_q[subj] + complexity * beta_q)

tau_u_p <- 1.1
u_p <- rnorm(N_subj, 0, tau_u_p)
alpha_p <- 0.5
beta_p <- 1
p_true <- plogis(alpha_p + u_p[subj] + complexity*beta_p)

tau_u_q <- 1.1
u_q <- rnorm(N_subj, 0, tau_u_q)
alpha_q <- 0.5
beta_q <- 1
q_true <- plogis(alpha_q  + u_q[subj] + complexity * beta_q)


# generate data vector of probabilities
theta_h <- matrix(
  c(NotRecognised(p_true, q_true, r_true),
    RecognisedAndNamed(p_true, q_true, r_true),
    TipOfTongue(p_true, q_true, r_true),
    RecognisedNotNamed(p_true, q_true, r_true)),
  ncol = 4)
# generate values of a categorical distribution of responses given Theta
ans <- rcat(N_obs,theta_h)
# make tibble of our real data
sim_exp <- tibble(subj = subj,
                   item = trial_number,
                   complexity = complexity,
                   w_ans = ans)
# make list of our experiment data
sim_exp_list <-  list(onlyprior = 0,
                      N_obs = nrow(sim_exp),
                      w_ans = sim_exp$w_ans,
                      N_subj = max(sim_exp$subj),
                      subj = sim_exp$subj,
                      complexity = sim_exp$complexity)
# get STAN model
mpt_hierarch_sim <- stan("HFR.stan", data = sim_exp_list)
print(mpt_hierarch_sim,
      pars = c("r", "tau_u", "alpha_p", "beta_p", "alpha_q", "beta_q"))

# posterior predictive checks for simulated data
as.data.frame(mpt_hierarch_sim) %>%
  select(r, alpha_p, beta_p, alpha_q, beta_q) %>%
  mcmc_recover_hist(true = c(r_true, alpha_p, beta_p, alpha_q, beta_q))

mcmc_hist(mpt_hierarch_sim, pars = c("r", "alpha_p", "beta_p", "alpha_q", "beta_q"))

# bar plot as posterior predictive check of general model output
gen_data <- rstan::extract(mpt_hierarch_sim)$pred_w_ans
ppc_bars(sim_exp$w_ans, gen_data) +
  ggtitle ("Hierarchical model") 

# get posterior predictions for these 12 subjects only
temp <- head(unique(sim_exp$subj), 12)
exp_subset <- sim_exp[sim_exp$subj%in%temp,]
ppc_bars_grouped(exp_subset$w_ans, 
                 gen_data[,1:240], group = exp_subset$subj) +
  ggtitle ("By-subject plot for the hierarchical model")

# get ppd for fame tiers
fame_tiers <- cut(data$C_TotalPageViews, breaks = 4, 
                  labels = c("D-Tier", "C-Tier", "B-Tier", "A-Tier"), include.lowest = TRUE)
# want frequency, not counts!
ppc_bars_grouped(sim_exp$w_ans, gen_data, group = fame_tiers, freq = FALSE) + 
  ggtitle ("Posterior predictive distributions for different tiers\nof fame")

# get ppd for specific celebrities
MJNAorNOT <- factor(ifelse(data$name=="Michael Jackson", "Michael Jackson", 
                           ifelse(data$name=="Bob Dylan", "Bob Dylan", 
                                  ifelse(data$name=="Steve Jobs", "Steve Jobs", 
                                         "Other Celebrities"))))

# want frequency, not counts!
ppc_bars_grouped(sim_exp$w_ans, gen_data, group = MJNAorNOT, freq = FALSE) + 
  ggtitle ("Plot for the hierarchical model for Michael Jackson, Bob Dylan or\nother celebrities")


# ==============================================================================
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
                              
                             r_true)) )


