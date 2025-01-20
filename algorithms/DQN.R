source("algorithms/functions.R") # helper functions within AI optimization  
library(patchwork)
library(torch)
library(container)


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

# Attention module ----------------------------------------------------------------------------------------------------------------------------------------
attention <- nn_module(
  initialize = function(hidden_size, dropout, seq_len=16){
    
    self$W_q = nn_linear(hidden_size, hidden_size, bias = F)
    self$W_k = nn_linear(hidden_size, hidden_size, bias = F)
    self$w_v = nn_linear(hidden_size, 1, bias = F)
    self$dropout = nn_dropout(dropout)
  },
  forward = function(queries, keys, values){
    #  browser()
    queries = self$W_q(queries)
    keys = self$W_k(keys)
    # After dimension expansion, shape of queries: 
    # (batch_size, no. of queries, 1, num_hiddens) 
    # and shape of keys: 
    # (batch_size, 1, no. of key-value pairs, num_hiddens). 
    # Sum them up with broadcasting
    features =  torch_tanh(queries$unsqueeze(3) + keys$unsqueeze(2))
    # There is only one output of self.w_v, so we remove the last
    # one-dimensional entry from the shape. Shape of scores: 
    # (batch_size, no. of queries, no. of key-value pairs)
    scores = self$w_v(features)$squeeze(-1)
    
    nnf_softmax(scores, -1) |> 
      self$dropout() |> 
      torch_bmm(values) # Shape of values: (batch_size, no. of key-value pairs, value dimension)
  }
)

queries = torch_rand(1, 3, 2)
keys = torch_rand(1, 6, 2)
values = torch_rand(1, 6, 2)

# queries = torch_tensor(c(1,3))$unsqueeze(1)$unsqueeze(3)
# keys = torch_tensor(c(1,16,14))$unsqueeze(1)$unsqueeze(3)
# values = torch_tensor(c(12,16,14))$unsqueeze(1)$unsqueeze(3)

attention(2, 0)(queries, keys, values)$squeeze(1) |> check_tnsr(4)
tst <- cities_tnsr$unsqueeze(1) 
attn_tst <- attention(2, 0)
attn_tst(tst[,1:16,,drop=F], tst[,13:16,], tst[,13:16,])[-1]  |> check_tnsr()
attn_tst(tst[,1:16,,drop=F], tst[,c(1,9,2),], tst[,c(1,9,2),])[-1] |> check_tnsr(3)
attn_tst(tst[,1:16,,drop=F], tst[,c(5,15),], tst[,c(5,15),])[-1] |> check_tnsr(3)
attn_tst(tst[,1:16,,drop=F], tst[,c(12,13,4,14,8,15,11),], tst[,c(12,13,4,14,8,15,11),])[-1] |> check_tnsr(3)


# Replay buffer functions -------------------------------------------------
create_buffer <- function(buffer_size=1000){
  vector(mode = "list", buffer_size)
}

push2buffer <- function(bfr, tnsr){
  idx <- sapply(bfr, \(x)!is.null(x)) |> which()
  if(length(bfr) >= length(idx)+1){
    bfr[[length(idx)+1]] <- tnsr
    return(bfr)
  }
  else{bfr[[length(idx)+1]] <- tnsr
  return(bfr[-1])
  }
}

create_buffer(2) |> 
  push2buffer(torch_rand(2, 3)) |> 
  push2buffer(torch_rand(3, 3)) |> 
  push2buffer(torch_rand(1, 2)) |>
  push2buffer(torch_rand(5, 3)) 

# DQN -----------------------------------------------------------------------------------------------------------------------------------------------------

# NBA calculated as attention approximation
DQN <- nn_module(
  initialize = function(n_hidden, n_cities, n_embeding = 32, dropout = .1, glimpse = 12){
    self$embedder <- nn_linear(2, n_embeding, bias = F)
    self$attn <- attention(n_embeding, dropout)
    self$glimpse <- glimpse
    
    self$out <- nn_sequential(
      nn_linear(n_cities, n_hidden),
      # nn_relu(),
      # nn_linear(n_hidden, n_hidden),
      # nn_relu(),
      # nn_linear(n_hidden, n_hidden),
      nn_relu(),
      nn_linear(n_hidden, n_cities),
      nn_relu())
  },
  forward = function(x, coords=cities_tnsr){
    # browser()
    n_cities <- cities_tnsr$size(1)
    if(n_cities==length(x)) return(-torch_ones(n_cities)*1e3)
    else{
      glimpse <- self$glimpse
      emb <- self$embedder(coords)$unsqueeze(1)
      pos <- emb[,x[length(x)], drop=FALSE] 
      target <- which(!seq_len(n_cities) %in% x)
      
      if(length(target)<glimpse){closest <- target}else{
        closest <- target[torch_cdist(pos, emb[,target])$topk(glimpse, largest=F)[[2]][-1,-1] |> as_array()]
      }
      attn_pnt <- self$attn(pos, emb[,closest,drop=F], emb[,closest,drop=F])
      #self_pnt <- self$attn(emb, emb[,closest,drop=F], emb[,closest,drop=F])
      
      Q_dist <- -torch_ones(n_cities)*1e3
      NBA <- self$out(torch_cdist(attn_pnt, emb)[-1,-1])[closest]
      
      Q_dist[closest] <- -torch_cdist(pos, emb)[-1,-1][closest] + NBA 
      
      #Q_dist[closest] <- self$out(torch_cdist(attn_pnt, emb)[-1,-1])[closest]
      Q_dist
    }
  }
)

DQN(132, n_cities, 2)(c(1, 9, 2, 12, 3))$unsqueeze(2) |> check_tnsr()
DQN(132, n_cities, 8)(1:15)$unsqueeze(2)  |> check_tnsr()
DQN(132, n_cities)(1)$unsqueeze(2) |> check_tnsr()

# Train loop ---------------------------------------------------------------------------------------------------------------------------------------------------
# Train loop with gradient step for move batch (the priority solution)
Q_train <- function(cities_mtrx, pars, epochs = 1000, n_hidden = 64){
  # cities_mtrx <- cities
  
  n_cities = nrow(cities_mtrx)
  seq_cities = seq_len(n_cities)
  dist_mtrx <- dist(cities_mtrx) |> as.matrix()
  
  loss_curve = rewards_curve = rep(-1e4, epochs)
  best_sol = integer(n_cities)
  
  state_net <- DQN(n_hidden, n_cities)
  target_net <- DQN(n_hidden, n_cities)
  lapply(target_net$parameters, \(x){x$requires_grad <- FALSE})
  
  optimizer <- optim_adamw(state_net$parameters, lr=1e-3, amsgrad=F)
  
  current_q_mem <- vector(mode='list', length=n_cities-1) # prepare memory for faster calculation
  next_q_mem <- numeric(length=n_cities-1) # prepare memory for faster calculation
  
  start_time = Sys.time()
  
  mem_rep = deque() # replay buffer 
  
  for(j in seq_len(epochs)){
    # Initialize epoch environment
    #mem = 1 # select 1st city 
    mem = sample(n_cities, 1)
    epoch_reward = 0
    
    for(i in 2:n_cities){
      q = state_net(mem) # get Q Vector in particular state
      
      # Define next best action either in greedy way either explore space 
      if(runif(1) > pars$epsilon){ 
        # EXPLOITATION
        q[mem] = -1e6  # avoid already visited states
        action = q$max(1)[[2]]$unsqueeze(1) |> as_array() # no grad by default for position
      }else{  
        # EXPLORATION
        options = seq_cities[which(!seq_cities %in% mem)] # avoid already visited states
        action = ifelse(length(options) == 1, options, sample(options, size = 1)) #prob = as_array(state_net(mem)[options] |> nnf_softplus(beta = .05))
      }
      
      reward = -dist_mtrx[mem[length(mem)], action] # get reward for such an action
      current_q = q[action]
      current_q_mem[i-1] <- list(current_q) # save to memory
      
      mem = c(mem, action) # update memory
      with_no_grad({next_q = target_net(mem)})
      
      # Update Q-learning matrix according to Bellman Equation
      if(length(mem) == n_cities){ # final destination city 
        next_q_mem[i-1] <- pars$gamma*next_q[mem[1]] |> as_array() + reward # not necessary to update q 
        epoch_reward = epoch_reward + reward - dist_mtrx[action, mem[1]] # update rewards
      }else{
        next_q[mem] = -1e6 # avoid already visited states
        next_q_mem[i-1] <- pars$gamma*next_q$max()$unsqueeze(1) |> as_array() + reward
        epoch_reward = epoch_reward + reward # update rewards
      }
    }
    
    #loss_fun <- nn_mse_loss()
    loss <- nnf_smooth_l1_loss(torch_stack(current_q_mem), next_q_mem)
   
    # Optimize the model
    optimizer$zero_grad()
    loss$backward()
    
    # In-place gradient clipping
    nn_utils_clip_grad_value_(state_net$parameters, 100)
    optimizer$step()
    
    loss <- as_array(loss)
    
    # Lower epsilon for less exploration probability 
    if(pars$epsilon > pars$epsilon_min){pars$epsilon = pars$epsilon_decay*pars$epsilon}
    if(best_sol[1]==0){best_sol <- mem}
    if(max(rewards_curve, na.rm = T) <= epoch_reward){best_sol <- mem}
    if(j %% 30 == 0){target_net$load_state_dict(state_net$state_dict())}
    
    
    rewards_curve[j] <- epoch_reward 
    loss_curve[j] <- loss 
    
    if (j %% 10 == 0) cat("Epoch: ", j, "| Time elapsed: ", format(round(Sys.time() - start_time, 2)), " | Loss: ", loss, " | Reward: ", epoch_reward, "\n")
    # }
  }
  list(last = mem, reward = rewards_curve, best = best_sol, loss = loss_curve, state_net = state_net, target_net=target_net)
}

pars <- list(epsilon = 1, epsilon_min = 0.01, epsilon_decay = 0.995, gamma = 0.95)
res <- Q_train(cities, pars, epochs = 1000, n_hidden = 64)


# Evaluation --------------------------------------------------------------

# Best solution 
p1 <- prep4plot(cities, res$best) |> 
  plot_tour() + 
  labs(title = paste0("Лучшее решение: ", -calc_dist4tnsr(cities_tnsr[res$best])|>as_array() |>round(2)), 
       col = "Маршрут", x = "x", y = "y")

res$final <- get_route4tnsr(res$state_net) # extract final solution 

# saveRDS(res, "test_samples/DQN.rds")
# res <- readRDS("test_samples/DQN.rds")

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
  geom_point(data = ~dplyr::filter(., n==which.min(res$reward)), col = "firebrick") + 
  labs(title = "Характеристики обучения нейронной сети", col = "", y = "Значение", x = "Итерация обучения")

# Combine plots to one 
(p1+p2)/p3 + plot_annotation("DQN")

# Check DQN matrix 
seq_along(cities[,1]) |> 
  sapply(\(x)res$target_net(x)|> as.matrix()) |> 
  apply(MARGIN =2, FUN = \(x)round(x, 2)) |> #t() |> 
  as.data.frame() |> 
  emphatic::hl(scale_color_viridis_c(), rows=row_number()[-1])
