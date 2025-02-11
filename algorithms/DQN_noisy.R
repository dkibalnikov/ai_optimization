source("algorithms/functions.R") # helper functions within AI optimization  
library(patchwork)
library(torch)


# Create task environment ---------------------------------------------------------------------------------------------------------------------------------
n_cities <- 16

cities <- generate_task(n_cities)

# Visualize task
p0 <- as.data.frame(cities) |> 
  transform(n=seq_len(n_cities)) |>
  ggplot(aes(x, y)) + 
  geom_point() + 
  geom_point(data = data.frame(n=1, x=0, y=0), shape = 21, size = 5, col = "firebrick") +
  geom_text(aes(label = n), nudge_x = 3)

p0

cities_tnsr <- torch_tensor(cities)
dist_tnsr <- dist(cities) |> as.matrix() |> 
  # apply(1, \(x)ifelse(x>quantile(x, .75), -1000, x)) |>  
  `diag<-`(1000) |> 
  torch_tensor()

# Noisy Nets --------------------------------------------------------------
NoisyLinear <- nn_module(
  "NoisyLinear",
  initialize = function(in_features, out_features, sigma_init = 0.017) {
    self$in_features <- in_features
    self$out_features <- out_features
    
    # Trainable parameters
    self$mu_weight <- nn_parameter(torch_empty(out_features, in_features))
    self$sigma_weight <- nn_parameter(torch_full(c(out_features, in_features), sigma_init))
    self$mu_bias <- nn_parameter(torch_empty(out_features))
    self$sigma_bias <- nn_parameter(torch_full(c(out_features), sigma_init))
    
    
    self$reset_parameters()
    # Non-trainable noise buffers
    #self$register_buffer("epsilon_weight", torch_empty(out_features, in_features))
    #self$register_buffer("epsilon_bias", torch_empty(out_features))
  },
  
  reset_parameters = function() {
    bound <- 1 / sqrt(self$in_features)
    nn_init_uniform_(self$mu_weight, -bound, bound)
    nn_init_uniform_(self$mu_bias, -bound, bound)
  },
  
  forward = function(x) {
    # Sample noise

    # Apply noise
    weight = self$mu_weight + self$sigma_weight * torch_normal(0,1,size = c(self$out_features, self$in_features))
    bias = self$mu_bias + self$sigma_bias * torch_normal(0,1,size = self$out_features)
    
    torch_matmul(x, weight$t()) + bias
  }
)

tst <- NoisyLinear(16, 32)
tst((torch_rand(16)))

# Attention module ----------------------------------------------------------------------------------------------------------------------------------------
attention <- nn_module(
  initialize = function(hidden_size, dropout, seq_len=16){
    
    self$W_q = nn_linear(hidden_size, hidden_size, bias = F)
    self$W_k = nn_linear(hidden_size, hidden_size, bias = F)
    self$w_v = nn_linear(hidden_size, 1, bias = F)
    self$dropout = nn_dropout(dropout)
  },
  forward = function(queries, keys, values){
    queries = self$W_q(queries)
    keys = self$W_k(keys)
    # After dimension expansion, shape of queries: 
    # (batch_size, no. of queries, 1, num_hiddens) 
    # and shape of keys: 
    # (batch_size, 1, no. of key-value pairs, num_hiddens). 
    # Sum them up with broadcasting
    features = torch_tanh(queries$unsqueeze(3) + keys$unsqueeze(2))
    # There is only one output of self.w_v, so we remove the last
    # one-dimensional entry from the shape. Shape of scores: 
    # (batch_size, no. of queries, no. of key-value pairs)
    scores = self$w_v(features)$squeeze(-1)
    
    nnf_softmax(scores, -1) |> 
      self$dropout() |> 
      # Shape of values: (batch_size, no. of key-value pairs, value dimension)
      torch_bmm(values) 
  }
)

# Some tests
queries = torch_rand(1, 3, 2)
keys = torch_rand(1, 6, 2)
values = torch_rand(1, 6, 2)

tst <- cities_tnsr$unsqueeze(1) 
attn_tst <- attention(2, 0)
attn_tst(tst[,1:16,,drop=F], tst[,13:16,], tst[,13:16,])[-1]  |> check_tnsr()
attn_tst(tst[,c(1,9,2,7),,drop=F], tst[,c(1,9,2,7),], tst[,c(1,9,2,7),])[-1] |> check_tnsr(3)

# Noisy attention module
NoisyAttention <- nn_module(
  "NoisyAttention",
  
  initialize = function(hidden_size, dropout, seq_len=16) {
    self$W_q <- NoisyLinear(hidden_size, hidden_size)
    self$W_k <- NoisyLinear(hidden_size, hidden_size)
    self$w_v <- NoisyLinear(hidden_size, 1)
    self$dropout <- nn_dropout(dropout)
  },
  
  forward = function(queries, keys, values) {
    queries <- self$W_q(queries)
    keys <- self$W_k(keys)
    
    # Expand dimensions and compute attention features
    features <- torch_tanh(queries$unsqueeze(3) + keys$unsqueeze(2))
    
    # Compute scores using noisy linear layer
    scores <- self$w_v(features)$squeeze(-1)
    
    # Apply softmax and dropout, then compute weighted sum
    nnf_softmax(scores, -1) |> 
      self$dropout() |> 
      torch_bmm(values) 
  }
)

# Some tests
attn_tst <- NoisyAttention(2, 0)
attn_tst(tst[,1:16,,drop=F], tst[,13:16,], tst[,13:16,])[-1]  |> check_tnsr()

# Replay buffer functions -------------------------------------------------

# Function is required for saving samples to buffer
push2buffer <- function(bfr, buffer_max_size=1000, tnsr){
  if(length(bfr) < buffer_max_size){
    bfr[[length(bfr)+1]] <- tnsr
    return(bfr)
  }
  else{bfr[[length(bfr)+1]] <- tnsr
  return(bfr[-1])
  }
}

# Test function: buffer should not be overwhelmed more than the threshold
NULL |> 
  push2buffer(4, torch_rand(2, 3)) |> 
  push2buffer(4, torch_rand(3, 3)) |> 
  push2buffer(4,torch_rand(1, 2)) |>
  push2buffer(4,torch_rand(5, 3)) |>
  push2buffer(4,torch_rand(6, 2)) 

# Calculate Q value -------------------------------------------------------
# Function allows to process Q value calculatiob requred for Bellman Equotation 
get_next <- function(action, reward, mem, target_net, dist_mtrx, cities_tnsr){
  # Update Q-learning matrix according to Bellman Equation
  if(length(mem) == n_cities-1){ # final destination city 
    -pars$gamma*dist_mtrx[action, mem[1]] + reward # terminate process
  }else{
    with_no_grad({next_q = target_net(mem, cities_tnsr)})
    next_q[mem] = -1e6 # avoid already visited states
    pars$gamma*next_q$max(1)[[1]]$unsqueeze(1) |> as_array() + reward
  }
}

# DQN -----------------------------------------------------------------------------------------------------------------------------------------------------
# DQN is rewritten for noisy nets 
DQN <- nn_module(
  initialize = function(n_hidden, n_cities, n_embeding = 32, dropout = 0.1, glimpse = n_cities){
    self$embedder <- nn_linear(2, n_embeding, bias = F)
    self$attn <- NoisyAttention(n_embeding, dropout) # it seems handcrafted attention is better than standart one
    # self$attn <- nn_multihead_attention(embed_dim = n_embeding, num_heads = 1, dropout = dropout)
    self$glimpse <- glimpse
    
    self$out <- nn_sequential(
      NoisyLinear(n_cities, n_hidden),
      # nn_relu(),
      # nn_linear(n_hidden, n_hidden),
      # nn_relu(),
      # nn_linear(n_hidden, n_hidden),
      nn_relu(),
      NoisyLinear(n_hidden, n_cities),
      nn_relu())
  },
  forward = function(x, coords=cities_tnsr){
    # Glimpse is the quantity of closest cities considered during step
    glimpse <- self$glimpse 
    emb <- self$embedder(coords)$unsqueeze(1) # just straightforward embedder
    pos <- emb[,x[length(x)], drop=FALSE] # current position
    target <- which(!seq_len(coords$size(1)) %in% x) # cities haven't been visited yet 
    
    if(length(target) < glimpse){closest <- target}else{
      # Deifine cities which are the closest ones to current position
      closest <- target[torch_cdist(pos, emb[,target])$topk(glimpse, largest=F)[[2]][-1,-1] |> as_array()]
    }
    
    # Calculate attention point (attention center)
    attn_pnt <- self$attn(pos, emb[,closest,drop=F], emb[,closest,drop=F])#[[1]]
    
    # Create template to be filled up
    Q_dist <- -torch_ones(coords$size(1))*1e3
    
    # Get distance from attention center to the rest cities
    attn_dist <- torch_cdist(attn_pnt, emb)[-1,-1] 
    
    # Use residual connection to calculate drift
    drift <- (self$out(attn_dist) + attn_dist)[closest] 
    
    # Combine closes distance and drift 
    Q_dist[closest] <- -torch_cdist(pos, emb)[-1,-1][closest] + drift 
    
    Q_dist
  },
  reset_parameters = function() {

    self$out$children$`0`$reset_parameters()
    self$out$children$`2`$reset_parameters()
    
    self$attn$children$W_q$reset_parameters
    self$attn$children$W_k$reset_parameters
    self$attn$children$w_v$reset_parameters
  }
)

# Test DQN
DQN(132, n_cities, 2)(c(1, 9, 2, 12, 3))$unsqueeze(2) |> check_tnsr()
DQN(132, n_cities, 8)(1:15)$unsqueeze(2)  |> check_tnsr()
DQN(132, n_cities)(1)$unsqueeze(2) |> check_tnsr()
DQN(132, n_cities)$reset_parameters()

# Train loop ---------------------------------------------------------------------------------------------------------------------------------------------------
# DQN with buffer
Q_train_bfr <- function(cities_mtrx, pars, epochs = 1000, n_hidden = 64, verbose=TRUE){
  
  n_cities = nrow(cities_mtrx)
  seq_cities = seq_len(n_cities)
  dist_mtrx <- dist(cities_mtrx) |> as.matrix()
  cities_tnsr <- torch_tensor(cities_mtrx)
  
  loss_curve = NULL
  rewards_curve = NULL
  best_sol = NULL
  mem_repl = NULL # replay buffer 
  
  state_net <- DQN(n_hidden, n_cities)
  target_net <- DQN(n_hidden, n_cities)
  lapply(target_net$parameters, \(x){x$requires_grad <- FALSE})
  
  optimizer <- optim_adamw(state_net$parameters, lr=1e-3, amsgrad=F)
  
  start_time = Sys.time()
  
  for(j in seq_len(epochs)){
    # Initialize epoch environment
    # mem = 1 # select 1st city 
    mem = sample(n_cities, 1)
    epoch_reward = 0
    
    for(i in 1:(n_cities-1)){
      q = state_net(mem, cities_tnsr) # get Q Vector in particular state
      
      # Define next best action in greedy way 
      q[mem] = -1e6  # avoid already visited states
      action =  q$max(1)[[2]]$unsqueeze(1) |> as_array() # no grad by default for position
     
      reward = -dist_mtrx[mem[length(mem)], action] # get reward for such an action
      
      # Save state, action and reward to buffer
      mem_repl <-  push2buffer(mem_repl, 1000, list(s=mem, a=action, r=reward))
      
      mem <- c(mem, action) # update memory
    }
    
    if(j %% 20 == 0){
     
      # Sample observations from buffer
      smpl <- mem_repl[sample(length(mem_repl), 200, replace = F)]
      # Extract Q value with activated gradient  
      q_approx <- lapply(smpl, \(x)state_net(x$s, cities_tnsr)[x$a]) |> torch_stack()
      target_net$eval() # frozen the target NN
      
      # DQN requires Q calculation utilizing target NN
      # y := r + gamma*(1-done)*Q_(s',a')
      q_calc <- lapply(smpl, \(x)get_next(x$a, x$r, x$s, target_net, dist_mtrx, cities_tnsr)) |> torch_cat()
      
      # Twin DQN for more process stability
      # y := r + gamma*Q_(s', argmaxQ(s', a'))
      #max_a <- lapply(smpl, \(x)state_net(x$a, cities_tnsr)$max(1)[[2]]|>as_array())
      #q_calc <- mapply(FUN = \(s, max_a){target_net(s$s, cities_tnsr)[max_a] + s$r}, smpl, max_a) |> torch_stack()
      
      loss <- nnf_smooth_l1_loss(q_approx, q_calc)

      # Optimize the model
      optimizer$zero_grad()
      loss$backward()
      
      # In-place gradient clipping
      nn_utils_clip_grad_value_(state_net$parameters, 100)
      optimizer$step()
      
      # Reset noise after update
      state_net$reset_parameters()
      
      state_net$eval()
      route <- get_route4tnsr(state_net, cities_tnsr)
      state_net$train()
      
      # Calculate reward 
      reward <- cities_tnsr[route]|> calc_dist4tnsr()|>as_array() |>round(2)
      loss <- as_array(loss)
      
      # Save data for monitoring
      rewards_curve <- c(rewards_curve, reward)
      loss_curve <- c(loss_curve, loss)
      
      if(max(rewards_curve, na.rm = T) <= reward){
        best_sol <- route
        new_dct <- state_net$state_dict()
        # Smooth NN update for stability 
        #new_dct <- mapply(\(t,s)0.5*s + (1-0.5)*t, target_net$state_dict(), state_net$state_dict())
        target_net$load_state_dict(new_dct)
       }
      
      # Print out for intermediate monitoring
      if(verbose) cat("Epoch: ", j, "| Time elapsed: ", format(round(Sys.time() - start_time, 2)), 
                      " | Loss: ", loss," | Reward: ", reward, "\n")
    }
    
  }
  list(last = mem, reward = rewards_curve, best = best_sol, loss = loss_curve, state_net = state_net, target_net=target_net)
}
pars <- list(gamma = 0.99)
res <- Q_train_bfr(cities, pars, epochs = 800, n_hidden = 64)

# Evaluation --------------------------------------------------------------
# Best solution 
p1 <- prep4plot(cities, res$best) |> 
  plot_tour() + 
  labs(title = paste0("Лучшее решение: ", -calc_dist4tnsr(cities_tnsr[res$best])|>as_array() |>round(2)), 
       col = "Маршрут", x = "x", y = "y")

res$state_net$eval()
res$final <- get_route4tnsr(res$state_net, cities_tnsr) # extract final solution 

res$target_net$eval()
res$final <- get_route4tnsr(res$target_net, cities_tnsr) # extract final solution 


# saveRDS(res, "test_samples/DQN.rds")
# res <- readRDS("test_samples/DQN.rds")
# torch_save(res$target_net, "test_samples/DQN_target")

# Last solution 
p2 <- prep4plot(cities, res$final) |> 
  plot_tour() +
  labs(title = paste0("Итоговое решение: ", -calc_dist4tnsr(cities_tnsr[res$final])|>as_array() |>round(2)), 
       col = "Маршрут", x = "x", y = "y")

# Training perfomance metrics 
p3 <- tibble::tibble(n = seq_along(res$loss), loss = res$loss, reward = res$reward) |>
  tidyr::pivot_longer(-1, names_to = "variable") |>
  ggplot(aes(n, value, col = variable)) + 
  geom_line(alpha = .3) + 
  geom_smooth(formula = y ~ s(x, bs = "cs"), method = 'gam') +
  facet_wrap(~variable, nrow = 2, scales = "free") +
  geom_point(data = ~dplyr::filter(., n==which.max(res$reward)), col = "firebrick") + 
  labs(title = "Характеристики обучения нейронной сети", col = "", y = "Значение", x = "Итерация обучения")

# Combine plots to one 
(p1+p2)/p3 + plot_annotation("DQN")

# Check DQN matrix 
seq_along(cities[,1]) |> 
  sapply(\(x)res$target_net(x)|> as.matrix()) |> 
  apply(MARGIN =2, FUN = \(x)round(x, 2)) |> #t() |> 
  as.data.frame() |> 
  emphatic::hl(scale_color_viridis_c())

# Test generalization capability ------------------------------------------
new_task <- generate_task(n_cities = 16, seed = 2022) 
new_res <- get_route4tnsr(res$target_net, torch_tensor(new_task)) # extract final solution 

prep4plot(new_task, new_res) |> 
  plot_tour() +
  labs(title = paste0("Итоговое решение: ", -calc_dist4tnsr(torch_tensor(new_task)[new_res])|>as_array() |>round(2)), 
       col = "Маршрут", x = "x", y = "y")

