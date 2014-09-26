rm(list = ls())
setwd("D:/Dropbox/Dokumenty/IOWA GAMBLING TASK/Model Code/A! muj a Helen PVL-D MIXTURE")
library(rstan) 

################################################################################
# Load data 
# Data of group 1
load("Data/Gescheidt_cont.Rdata")
# load("Data/simulated_1.RData")
choice <- choice[1:2,]
wi <- wi[1:2,]
lo <- lo[1:2,]


# Data preparation
n_s_1 <- nrow(wi)  # Number of subjects
n_t <- ncol(wi)  # Number of trials
net1 <- wi + lo    # Net gains of each subject on each trial
choice1 <- choice
index <- rep(1, n_s_1)
rm(wi, lo, choice)

# Data of group 2
load("Data/Gescheidt_exp.Rdata")
# load("Data/simulated_2.RData")
choice <- choice[1:2,]
wi <- wi[1:2,]
lo <- lo[1:2,]


n_s_2 <- nrow(wi)
index <- c(index, rep(2, n_s_2))
n_s <- n_s_1 + n_s_2
net2 <- wi + lo
net <- rbind(net1, net2)
choice <- rbind(choice1, choice)

# SPECIFY prior mixing proportion (higher value favours different group dist.)
mix <- c(.5, .5, .5, .5)

pvl_d_dat <- list(choice = choice, n_s = n_s, n_s_1=n_s_1, n_s_2=n_s_2,
                  n_t = n_t, net = net, index=index, mix=mix)  
################################################################################    

mypars <- c("z", "mu_A", "mu_w", "mu_a", "mu_c", "std")  # for optimized model

# Fit the model
pvl_d_fit <- stan(file = "Models/4-z/pvl_d_mix_all_optimized_3_(1-std).stan",
                  data = pvl_d_dat, init = "random", pars = mypars, 
                  warmup = 1, iter = 1, thin = 1, chains = 1)

start = Sys.time()

################################################################################    
chains <- 700

BF1 <- c()
BF2 <- c()
BF3 <- c()
BF4 <- c()
lp <- c()

start = Sys.time()
for (i in 1:chains) {
  samples <- stan(fit = pvl_d_fit,
                    data = pvl_d_dat, init="random", pars=mypars, 
                    warmup = 300, iter = 800, thin = 1, chains = 1)
  
  
  ############# Testing Z proportions ##################################
  z1 <- extract(samples)$z[, 1]
  z2 <- extract(samples)$z[, 2]
  z3 <- extract(samples)$z[, 3]
  z4 <- extract(samples)$z[, 4]
  lp[i] <- mean(extract(samples)$lp__)
  
  z <- paste(z1, z2, z3, z4, sep="")
  
  
  # BF for parameter A
  BF1[i] <- sum(z=="1000")/sum(z=="0000")
  # BF for parameter w
  BF2[i] <- sum(z=="0100")/sum(z=="0000")
  # BF for parameter a
  BF3[i] <- sum(z=="0010")/sum(z=="0000")
  # BF for parameter c
  BF4[i] <- sum(z=="0001")/sum(z=="0000")

}
end = Sys.time()                  
end - start  

res <- cbind(BF1, BF2, BF3, BF4, lp)
results <- rbind(results, res)

mean(results[, "BF1"]); median(results[, "BF1"])
mean(results[, "BF2"]); median(results[, "BF2"])
mean(results[, "BF3"]); median(results[, "BF3"])
mean(results[, "BF4"]); median(results[, "BF4"])

windows(14,7)
layout(matrix(1:4, 2, 2, byrow=TRUE))
hist(results[, "BF1"], breaks=chains)
hist(results[, "BF2"], breaks=chains)
hist(results[, "BF3"], breaks=chains)
hist(results[, "BF4"], breaks=chains)

# windows(14,7)
# layout(matrix(1:4, 2, 2, byrow=TRUE))
# hist(log(results[, "BF1"]), breaks=chains)
# hist(log(results[, "BF2"]), breaks=chains)
# hist(log(results[, "BF3"]), breaks=chains)
# hist(log(results[, "BF4"]), breaks=chains)


# lpResults <- results[20:885, ]

windows(14,7)
layout(matrix(1:4, 2, 2, byrow=TRUE))
plot(lpResults[, "BF1"], lpResults[, 5])
plot(lpResults[, "BF2"], lpResults[, 5])
plot(lpResults[, "BF3"], lpResults[, 5])
plot(lpResults[, "BF4"], lpResults[, 5])

# windows(14,7)
# layout(matrix(1:4, 2, 2, byrow=TRUE))
# plot(log(lpResults[, "BF1"]), lpResults[, 5])
# plot(log(lpResults[, "BF2"]), lpResults[, 5])
# plot(log(lpResults[, "BF3"]), lpResults[, 5])
# plot(log(lpResults[, "BF4"]), lpResults[, 5])
