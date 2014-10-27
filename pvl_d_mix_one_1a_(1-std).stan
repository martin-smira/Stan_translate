# PVL-Delta in Stan

data {
  int<lower=1> n_s;                       // Total # subjects
  int<lower=1> n_s_1;                     // # subjects in group 1
  int<lower=1> n_s_2;                     // # subjects in group 2    
  int<lower=1> n_t;                       // # trials
  int<lower=0,upper=4> choice[n_s, n_t];  // # subj. x # trials matrix with 
                                          //   choices
  real<lower=-35,upper=2> net[n_s, n_t];  // Net amount of wins + losses    
                                          // (# subj. x # trials matrix) 
  int<lower=0,upper=2> index[n_s];        // Group index  
  real<lower=0,upper=1> mix;         // user specified prior mixing proportion    
}

parameters {
  // Group-level mean parameters
  real mu_w_pr;   
  real mu_a_pr;
  real mu_A_pr_1;  
  real mu_A_pr_2;    
  real mu_c_pr;

  // Group-level standard deviation  
  real<lower=0,upper=1.5> sd_w;
  real<lower=0,upper=1.5> sd_a;
  real<lower=0,upper=1.5> sd_A;  
  real<lower=0,upper=1.5> sd_c;    

  // Individual-level paramters    
  vector[n_s] A_ind_pr; 
  
  vector[n_s] raw[3];
}

transformed parameters {
  vector[2] lp_parts;
  
  // Individual-level paramters    
  vector<lower=0,upper=1>[n_s] A_ind; 
  vector<lower=0,upper=5>[n_s] w_ind; 
  vector<lower=0,upper=1>[n_s] a_ind;   
  vector<lower=0,upper=5>[n_s] c_ind;    

  for (s in 1:n_s)
  {
    A_ind[s] <- Phi(A_ind_pr[s]);
    w_ind[s] <- Phi(mu_w_pr + sd_w * raw[1,s]) * 5;
    a_ind[s] <- Phi(mu_a_pr + sd_a * raw[2,s]);    
    c_ind[s] <- Phi(mu_c_pr + sd_c * raw[3,s]) * 5;  
  }
   
  lp_parts[1] <- log1m(mix) + (normal_log(tail(A_ind_pr, n_s_2), mu_A_pr_1, sd_A));
  lp_parts[2] <- log(mix) + (normal_log(tail(A_ind_pr, n_s_2), mu_A_pr_2, sd_A));
}
model {
  vector[4] p;
  vector[4] Ev;
  vector[4] dummy;
  real theta;
  real v;

  # Prior on the group-level mean parameters
  # probit scale [-Inf, Inf]  
  mu_w_pr ~ normal(0, 1);
  mu_a_pr ~ normal(0, 1);
  mu_A_pr_1 ~ normal(0, 1);
  mu_A_pr_2 ~ normal(0, 1);  
  mu_c_pr ~ normal(0, 1);
  
  for (i in 1:3)
    raw[i] ~ normal(0, 1);
       
  # Group 1
  for (s in 1:n_s_1)
    A_ind_pr[s] ~ normal(mu_A_pr_1, sd_A);         

  # Group 2 
  increment_log_prob(log_sum_exp(lp_parts));
     
  for (s in 1 : n_s)  // loop over subjects
  {  
    theta <- pow(3, c_ind[s]) - 1;     
   
   // Trial 1
    for (d in 1 : 4)  // loop over decks
    {  
      p[d] <- .25;
      Ev[d] <- 0;
    }  
    choice[s,1] ~ categorical(p);

    // Remaining trials
    for (t in 1 : (n_t - 1))  
    { 
      if (net[s,t] >= 0)
        v <- pow(net[s,t], A_ind[s]);
      else
        v <- -1 * w_ind[s] * pow(fabs(net[s,t]), A_ind[s]);

      Ev[choice[s,t]] <- (1 - a_ind[s]) * Ev[choice[s,t]] + a_ind[s] * v;      
      choice[s,t + 1] ~ categorical_logit(Ev * theta); 
    }
  }
}
generated quantities {
	int z;
	real prob;
	
  real<lower=0,upper=1> mu_A_1;   
  real<lower=0,upper=1> mu_a;
  real<lower=0,upper=1> mu_A_2; 
  real<lower=0,upper=5> mu_w;    
  real<lower=0,upper=5> mu_c;
  
  mu_A_1 <- Phi(mu_A_pr_1);
  mu_A_2 <- Phi(mu_A_pr_2);   
  mu_a <- Phi(mu_a_pr);
  mu_w <- Phi(mu_w_pr) * 5; 
  mu_c <- Phi(mu_c_pr) * 5;  
  
	prob <- exp(lp_parts[2]) / (exp(lp_parts[1])+exp(lp_parts[2])); 
	z <- bernoulli_rng(prob);  // low z means evidence for different group distributions


}