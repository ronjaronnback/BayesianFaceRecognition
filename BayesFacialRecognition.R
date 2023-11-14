
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


data <-read.csv("outcome_data.csv")



# Find probabilities for each branch --------------------------------------
value_counts <- table(data$outcome)
value_counts



p_true <- .58
q_true <- .66
r_true <- .23

value_counts <- table(data$participant)
#participant 4 and 114 only had 19 answers , not 20 so they were dropped
newdata <- subset(data, participant != c(4, 114))

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


Theta <- tibble(theta_NR,
    theta_C,
    theta_RNN,
    theta_TT)

N_trials <- 200
(ans <- rmultinom(1, N_trials, c(Theta)))


# Stan Model ---------------------------------------------------------

data_face <-  list(N_trials = N_trials,
                   ans = c(ans)) 

fit_face <- stan("Facial.stan", data = data_face)  




# Model Inspection --------------------------------------------------------
print(fit_face, pars = c("p", "q", "r"))


as.data.frame(fit_face) %>%
  select(c("p","q","r")) %>%
  mcmc_recover_hist(true = c(p_true, q_true, r_true)) +
  coord_cartesian(xlim = c(0, 1))

print(fit_face, pars = c("theta"))
