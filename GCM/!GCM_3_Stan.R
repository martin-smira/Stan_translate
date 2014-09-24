# clears workspace: 
rm(list=ls()) 

library(rstan)

model <- "
// Generalized Context Model With Two Latent Groups
data { 
  int nstim;
  int nsubj;
  int n;
  int a[nstim];
  int y[nsubj,nstim];
  matrix[nstim,nstim] d1;
  matrix[nstim,nstim] d2;
}
transformed data {
  real b;
  
  b <- .5;
}
parameters {
  vector<lower=0>[nsubj] c;
  vector<lower=0,upper=1>[nsubj] w;
  vector<lower=0>[2] cpredg;
  vector<lower=0,upper=1>[2] wpredg;  
  real<lower=0,upper=1> phic;
  real<lower=0,upper=1> phig;
  real<lower=0,upper=1> muctmp;
  real<lower=0,upper=1> muwtmp;
  real<lower=0,upper=1> delta;
  real<lower=0,upper=1> sigmactmp;
  real<lower=0,upper=1> sigmawtmp;
} 
transformed parameters {
  matrix<lower=0,upper=1>[nsubj,nstim] r; 
  real<lower=0,upper=5> muc;
  vector<lower=0,upper=1>[2] muw;
  real<lower=.01,upper=3> sigmac;
  real<lower=.01,upper=1> sigmaw;
  vector[2] lp_parts_c[nsubj];
  vector[2] lp_parts_g[nsubj];
  
  // Mean Generalization
  muc <- 5 * muctmp;
  // Mean Attention
  muw[1] <- muwtmp;
  muw[2] <- fmin(1, delta + muw[1]);
  // Standard Deviation Generalization
  sigmac <- fmax(.01, 3 * sigmactmp);
  // Standard Deviation Attention
  sigmaw <- fmax(.01, sigmawtmp);
  
  for (i in 1:nstim) {
    vector[nstim] numerator;
    vector[nstim] denominator;
    for (k in 1:nsubj) {
      for (j in 1:nstim) {
        real s;
        // Similarities
        s <- exp(-c[k] * (w[k] * d1[i,j] + (1 - w[k]) * d2[i,j])); 
        
        // Base Decision Probabilities
        numerator[j] <- (a[j] == 1) * b * s;
        denominator[j] <- (a[j] == 1) * b * s + (a[j] == 2) * (1 - b) * s;
      }
      r[k,i] <- sum(numerator) / sum(denominator);
    } 
  }  
  for (k in 1:nsubj) { 
    lp_parts_g[k,1] <- log1m(phig) + normal_log(w[k], muw[1], sigmaw);
    lp_parts_g[k,2] <- log(phig) + normal_log(w[k], muw[2], sigmaw);
  }
  for (k in 1:nsubj) {
    lp_parts_c[k,1] <- log1m(phic) + binomial_log(y[k], n, r[k]);
    lp_parts_c[k,2] <- log(phic) + binomial_log(y[k], n, .5);
  }
}
model {
  // Subject Parameters
  for (k in 1:nsubj)
    c[k] ~ normal(muc, sigmac)T[0,];
    
  // Predicted Group Parameters
  for (g in 1:2) {
    wpredg[g] ~ normal(muw[g], sigmaw)T[0,1];
    cpredg[g] ~ normal(muc, sigmac)T[0,];
  }
  
  // Decision Data
  for (k in 1:nsubj) 
    increment_log_prob(log_sum_exp(lp_parts_g[k]));
    
  for (k in 1:nsubj)   
    increment_log_prob(log_sum_exp(lp_parts_c[k]));
}
generated quantities {
  matrix[nstim,nsubj] predy;
  matrix[3,nstim] predyg;
  int<lower=0,upper=3> z[nsubj];

  for (i in 1:nstim) {
    matrix[2,nstim] numeratorpredg;
    matrix[2,nstim] denominatorpredg;
    vector[3] rpredg;
    for (j in 1:nstim) { 
      for (g in 1:2) {
        real spredg;
        spredg <- exp(-cpredg[g] * (wpredg[g] * d1[i,j]
                                    + (1 - wpredg[g]) * d2[i,j]));
        numeratorpredg[g,j]   <- (a[j] == 1) * b * spredg;
        denominatorpredg[g,j] <- (a[j] == 1) * b * spredg
                                  + (a[j] == 2) * (1 - b) * spredg;
      }      
    }
    for (g in 1:2)
      rpredg[g] <- sum(numeratorpredg[g]) / sum(denominatorpredg[g]); 
      
    rpredg[3] <- 0.5;
    // Subjects
    for (k in 1:nsubj)
      predy[i,k] <- binomial_rng(n, r[k,i]);
    // Groups
    for (g in 1:3)
      predyg[g,i] <- binomial_rng(n, rpredg[g]);
  }
  
  for (k in 1:nsubj) {
    vector[2] prob_c;
    vector[2] prob_g;
    int zc;
    int zg;
    
    prob_c <- softmax(lp_parts_c[k]);
    prob_g <- softmax(lp_parts_g[k]);
    zc <- bernoulli_rng(prob_c[2]);
    zg <- bernoulli_rng(prob_g[2]);
    z[k] <- (zc == 0) * (zg + 1) + 3 * (zc == 1);
  }
}"

load("KruschkeData.Rdata")

y <- t(y)  # Transpose matrix (for simpler Stan implementation)        

# To be passed on to Stan
data <- list(y=y, nstim=nstim, nsubj=nsubj, n=n, a=a, d1=d1, d2=d2) 

myinits <- list(
  list(c=rep(2.5, nsubj), w=rep(.5, nsubj), cpredg=c(2.5, 2.5),
       wpredg=c(.5, .5), phic=.5, phig=.5, muctmp=.5, muwtmp=.25, delta=.5,
       sigmactmp=.5, sigmawtmp=.5),
  list(c=rep(2.5, nsubj), w=rep(.5, nsubj), cpredg=c(2.5, 2.5),
       wpredg=c(.5, .5), phic=.5, phig=.5, muctmp=.5, muwtmp=.25, delta=.5,
       sigmactmp=.5, sigmawtmp=.5))

# Parameters to be monitored
parameters <- c("delta", "phic", "phig", "c", "w", "muc", "muw", "sigmac",
                "sigmaw", "predy", "wpredg", "cpredg", "predyg", "z")  

# The following command calls Stan with specific options.
# For a detailed description type "?stan".
samples <- stan(model_code=model,   
                data=data, 
  #              init=myinits,  # If not specified, gives random inits
                pars=parameters,
                iter=1500, 
                chains=2, 
                thin=1,
                warmup=500,  # Stands for burn-in; Default = iter/2
                # seed=123  # Setting seed; Default is random seed
)
# Now the values for the monitored parameters are in the "samples" object, 
# ready for inspection.

# print(samples)

z <- extract(samples)$z
pr <- extract(samples)$predyg

############# PLot ################
z1 <- c()
z2 <- c()
z3 <- c()
for (i in 1:40) {
  z1[i] <- sum(z[,i] == 1) / length(z[,1])
  z2[i] <- sum(z[,i] == 2) / length(z[,1])
  z3[i] <- sum(z[,i] == 3) / length(z[,1])
}

ord1 <- order(z1, decreasing=TRUE)
ord2 <- order(z2, decreasing=TRUE)
ord3 <- order(z3, decreasing=TRUE)
ord1 <- ord1[z1[ord1] > .5]
ord2 <- ord2[z2[ord2] > .5]
ord3 <- ord3[z3[ord3] > .5]

ord <- c(ord3, ord2, ord1)

windows()
plot(z1[ord], ylim=c(0, 1), type="b", pch=0)
lines(z2[ord], type="b", pch=1)
lines(z3[ord], type="b", pch=2)

# Heatmaps for predyg
for (p in 1:3) {
  dat <- matrix(NA, 9, 8)
  for (i in 1:8) 
    for (j in 0:8)
      dat[j+1,i]  <- table(pr[,p,i])[as.character(j)]
  dat[is.na(dat)] <- 0
  windows()
  heatmap(dat, Rowv=NA, Colv=NA, scale="column")
}