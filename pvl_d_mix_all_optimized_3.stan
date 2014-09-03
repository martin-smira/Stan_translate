# PVL-Delta in Stan

data {
  int<lower=1> n_s;                       // Total # subjects
  int<lower=0> n_s_1;                     // # subjects in group 1
  int<lower=0> n_s_2;                     // # subjects in group 2    
  int<lower=1> n_t;                       // # trials
  int<lower=0,upper=4> choice[n_s, n_t];  // # subj. x # trials matrix with 
                                          //   choices
  real<lower=-35,upper=2> net[n_s, n_t];  // Net amount of wins + losses    
                                          // (# subj. x # trials matrix) 
  int<lower=0,upper=2> index[n_s];        // Group index          
}

parameters {
  // Group-level mean parameters
  vector[2] mu_pr[4];
  // Group-level standard deviation  
  vector<lower=.01,upper=1.5>[2] std[4];
  // Individual-level paramters    
  vector[n_s] ind_pr[4];
  vector<lower=0,upper=1>[4] mix;
}

transformed parameters {
  vector[2] lp_parts[4];
  // Individual-level paramters    
  vector<lower=0,upper=1>[n_s] A_ind; 
  vector<lower=0,upper=5>[n_s] w_ind; 
  vector<lower=0,upper=1>[n_s] a_ind;   
  vector<lower=0,upper=5>[n_s] c_ind;   
  
  for (s in 1:n_s) {
    A_ind[s] <- Phi_approx(ind_pr[1,s]);
    w_ind[s] <- Phi_approx(ind_pr[2,s]) * 5;
    a_ind[s] <- Phi_approx(ind_pr[3,s]);    
    c_ind[s] <- Phi_approx(ind_pr[4,s]) * 5;  
  }
  
  for (i in 1:4) {
    lp_parts[i,1] <- log(mix[i]) + normal_log(tail(ind_pr[i], n_s_2),
                                               mu_pr[i,1], std[i,1]);
    lp_parts[i,2] <- log1m(mix[i]) + normal_log(tail(ind_pr[i], n_s_2),
                                                 mu_pr[i,2], std[i,2]);
  }
}
model {
  vector[4] p;
  vector[4] Ev;
  real theta;
  real v;

  // Prior on the group-level mean parameters
  // probit scale [-Inf, Inf] 
  for (i in 1:4)
    mu_pr[i] ~ normal(0, 1);
    
  // Individual-level paramters
  // Group 1  
  for (i in 1:4)  
    for (s in 1:n_s_1)
      ind_pr[i,s] ~ normal(mu_pr[i,1], std[i,1]);      
  // Group 2 
  for (i in 1:4)  
    increment_log_prob(log_sum_exp(lp_parts[i]));
  
  for (s in 1:n_s) {  // loop over subjects
    theta <- 3 ^ c_ind[s] - 1;     
   
   // Trial 1
    for (d in 1:4) { // loop over decks
      p[d] <- .25;
      Ev[d] <- 0;
    }  
    choice[s,1] ~ categorical(p);

    // Remaining trials
    for (t in 1:(n_t - 1)) { 
      if (net[s,t] >= 0)
        v <- (net[s,t]) ^ A_ind[s];
      else
        v <- -1 * w_ind[s] * fabs(net[s,t]) ^ A_ind[s];

      Ev[choice[s,t]] <- (1 - a_ind[s]) * Ev[choice[s,t]] + a_ind[s] * v;      
      choice[s,t + 1] ~ categorical_logit(Ev * theta); 
    }
  }
}
generated quantities {
  int<lower=0,upper=1> z[4];  // high values of z indicate group distribution 
                              // to be different
  vector<lower=0,upper=1>[2] mu_A; 
  vector<lower=0,upper=5>[2] mu_w; 
  vector<lower=0,upper=1>[2] mu_a;   
  vector<lower=0,upper=5>[2] mu_c; 
  
  for (i in 1:2) {
    mu_A[i] <- Phi_approx(mu_pr[1,i]);
    mu_w[i] <- Phi_approx(mu_pr[2,i]) * 5;
    mu_a[i] <- Phi_approx(mu_pr[3,i]);
    mu_c[i] <- Phi_approx(mu_pr[4,i]) * 5;
  }
  for (i in 1:4) {
    vector[2] prob;
    prob <- softmax(lp_parts[i]);
    z[i] <- bernoulli_rng(prob[2]); 
  }
}