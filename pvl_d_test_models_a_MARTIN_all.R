################################################################################
# Helen Steingroever, Last updated July 2014
# Fit Pvl-Delta model in Stan
################################################################################

################################################################################
rm(list = ls())
setwd("D:/Dropbox/Dokumenty/!Amster/Helen PVL-D mixture")
#setwd("~/Dropbox/2014/Berlin")
library(rstan) 
set_cppo('fast')  # for best running speed

################################################################################
# Load data 
# Data of group 1
# load("Data/Thorsten_2_1.RData")
load("Data/control_group_data.RData")  # Gescheidt data - KONTROLNI skupina

# Data preparation
n_s_1 <- nrow(wi)  # Number of subjects
n_t <- ncol(wi)  # Number of trials
net <- (wi + lo) / 100   # Net gains of each subject on each trial
choice <- ind
choice1 <- choice
index <- rep(1, n_s_1)
rm(wi, lo, choice)

# Data of group 2
# load("Data/Thorsten_2_2.RData")
load("Data/experimental_group_data.RData")  # Gescheidt data - EXPERIMENTALNI skupina

n_s_2 <- nrow(wi)
index <- c(index, rep(2, n_s_2))
n_s <- n_s_1 + n_s_2
net <- rbind(net, (wi + lo) / 100)
choice <- ind
choice <- rbind(choice1, choice)
#p2 <- c(.5, .5)
pvl_d_dat <- list(choice = choice, n_s = n_s, n_s_1=n_s_1, n_s_2=n_s_2,
                  n_t = n_t, net = net, index=index)  
################################################################################    
# setwd("FitPvlD")
 
# myinits <- list(  # for "OPTIMIZED" model
#     list(mu_pr=matrix(rep(0,8),4,2), std=matrix(rep(1,8),4,2), 
#          ind_pr=matrix(rep(0,4*44),4,44), mix=c(.5,.5,.5,.5)))

# mypars <- c( "z","mu_A_1", "mu_A_2", "mu_w_1", "mu_w_2", "mu_a_1", "mu_a_2", 
#              "mu_c_1", "mu_c_2", "A_ind", "w_ind", "a_ind", "c_ind")
mypars <- c("z", "mu", "std")  # for optimized model

# Fit the model
start = Sys.time()
pvl_d_fit <- stan(file = 'pvl_d_a_Michael_mix_all_optimized.stan', data = pvl_d_dat, 
                  verbose = FALSE, 
                  warmup = 5, iter = 10, thin = 1, chains = 1, 
                  init = "random", pars = mypars)
# pvl_d_fit <- stan(file = 'pvl_d_a_Michael_mix_all.stan', data = pvl_d_dat, 
#                   pars=mypars, init="random",
#                   warmup = 10, iter = 20, thin = 1, chains = 1)

pvl_d_fit <- stan(fit = pvl_d_fit, data = pvl_d_dat, 
                  pars=mypars, init="random",
                  warmup = 100, iter = 300, thin = 1, chains = 4)
end = Sys.time()                  
end - start        
print(pvl_d_fit, digits=2)      
save(pvl_d_fit, file="pvl_d_all_big_data.rdata")
############# Testing Z proportions ##################################
z1 <- extract(pvl_d_fit)$z[, 1]
z2 <- extract(pvl_d_fit)$z[, 2]
z3 <- extract(pvl_d_fit)$z[, 3]
z4 <- extract(pvl_d_fit)$z[, 4]

z <- paste(z1,z2,z3,z4,sep="")


# Testing if the results match the "one z" solution
# proportion of z[1]
sum(z=="1111")/(sum(z=="0111")+sum(z=="1111"))
# proportion of z[2] 
sum(z=="1111")/(sum(z=="1011")+sum(z=="1111"))
# proportion of z[3] 
sum(z=="1111")/(sum(z=="1101")+sum(z=="1111"))
# proportion of z[4] 
sum(z=="1111")/(sum(z=="1110")+sum(z=="1111"))

table(z)


############### T-test for individual parameters (A_ind means) ##############

data <- summary(pvl_d_fit)
x <- data$summary[213:256,1]
g <- rep(c("con", "exp"), each=22)

t.test(x ~ g)  # p-value = 0.000686




