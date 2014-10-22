# PVL-Delta in Stan

data {
  int<lower=1> n_s;                       // Total # subjects
  int<lower=1> n_t;                       // # trials
  int<lower=0,upper=4> choice[n_s, n_t];  // # subj. x # trials matrix with 
                                          //   choices
  real<lower=-35,upper=2> net[n_s, n_t];  // Net amount of wins + losses    
                                          // (# subj. x # trials matrix)      
  real<lower=0,upper=1> mix;             // Prior mixing proportion
}

parameters {
  // Group-level mean parameters
  real mu_A_pr;   
  real mu_a_pr;
  real mu_w_pr;     
  real mu_c_pr;

  // Group-level standard deviation  
  real<lower=0,upper=1.5> sd_A;
  real<lower=0,upper=1.5> sd_a;
  real<lower=0,upper=1.5> sd_w;  
  real<lower=0,upper=1.5> sd_c;    

  vector[n_s] raw[4];
}

transformed parameters {
  vector[2] lp_parts;
  matrix[n_s,n_t-1] lp_tmp[2];
 
  // Individual-level paramters    
  vector<lower=0,upper=1>[n_s] A_ind; 
  vector<lower=0,upper=5>[n_s] w_ind; 
  vector<lower=0,upper=1>[n_s] a_ind;   
  vector<lower=0,upper=5>[n_s] c_ind;    
  
  for (s in 1:n_s) {
    A_ind[s] <- Phi(mu_A_pr + sd_A * raw[1,s]);
    w_ind[s] <- Phi(mu_w_pr + sd_w * raw[2,s]) * 5;
    a_ind[s] <- Phi(mu_a_pr + sd_a * raw[3,s]);    
    c_ind[s] <- Phi(mu_c_pr + sd_c * raw[4,s]) * 5;  
  }
  
  for (s in 1:n_s) { // loop over subjects
    real theta;
    vector[4] Ev_delta;
    vector[4] Ev_decay;

    // Trial 1
    Ev_delta <- rep_vector(0, 4);   // assign 0 to all elements of Ev_delta
    Ev_decay <- rep_vector(0, 4);   // assign 0 to all elements of Ev_delta
    
    theta <- 3 ^ c_ind[s] - 1;     

    // Remaining trials
    for (t in 1:(n_t - 1)) { 
      real v;
      
      if (net[s,t] >= 0)
        v <- (net[s,t]) ^ A_ind[s];
      else
        v <- -w_ind[s] * fabs(net[s,t]) ^ A_ind[s];

      // Delta Learning Rule
      Ev_delta[choice[s,t]] <- (1 - a_ind[s]) * Ev_delta[choice[s,t]] 
                               + a_ind[s] * v;   
      // Decay Lerning Rule      
      Ev_decay <- Ev_decay * a_ind[s];
      Ev_decay[choice[s,t]] <- Ev_decay[choice[s,t]] + v;
            
      lp_tmp[1,s,t] <- categorical_logit_log(choice[s,t+1], Ev_delta * theta); 
      lp_tmp[2,s,t] <- categorical_logit_log(choice[s,t+1], Ev_decay * theta); 
    }
  }  
  lp_parts[1] <- log(mix) + sum(lp_tmp[1]);
  lp_parts[2] <- log1m(mix) + sum(lp_tmp[2]);
}
model {
  # Prior on the group-level mean parameters
  # probit scale [-Inf, Inf]  
  mu_A_pr ~ normal(0, 1);
  mu_a_pr ~ normal(0, 1);
  mu_w_pr ~ normal(0, 1);
  mu_c_pr ~ normal(0, 1);
  for (i in 1:4)
    raw[i] ~ normal(0, 1);   
    
  increment_log_prob(log_sum_exp(lp_parts));   
}
generated quantities {
	real<lower=0,upper=1> mu_A;   
  real<lower=0,upper=5> mu_w; 
  real<lower=0,upper=1> mu_a;
  real<lower=0,upper=5> mu_c;
  vector<lower=0,upper=1>[2] prob;
  int z;  // mixing variable
  
  mu_A <- Phi(mu_A_pr);
  mu_w <- Phi(mu_w_pr) * 5; 
  mu_a <- Phi(mu_a_pr);
  mu_c <- Phi(mu_c_pr) * 5;  
  
  prob <- softmax(lp_parts);
  z <- bernoulli_rng(prob[1]);   
}