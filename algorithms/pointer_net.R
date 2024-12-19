library(torch) # PyTorch alternative for R
library(patchwork) # package to plot multiple plots 
source("algorithms/functions.R") # helper functions within AI optimization  

# Set up GPU --------------------------------------------------------------
avail_cuda <- cuda_is_available()
dvc <- torch_device(if(avail_cuda)"cuda" else "cpu")
dvc
use_cuda <- function(obj){
  if (avail_cuda){obj<-obj$cuda()}
  obj
} 
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


cities_tnsr <- torch_tensor(cities, device=dvc) # tensor version of task
# Attention module ----------------------------------------------------------------------------------------------------------------------------------------
# Attention based on Bahdanau Attention Mechanism (additive attention scoring)
# Inputs: 
#      - query: [1, 1, hidden_size]
#               calculated over recurrent loop, 
#               comes either from the Decoder (LSTM)
#      - ref:   [1, seq_len, hidden_size]
#.              stands for both: key and value, 
#.              comes from the Decoder (LSTM) main output
#.     - idxs: previously selected cities for masking probabilities 
# Outputs: 
#      - ref: input ref after convolution transformation
#      - logits: probabilities of selecting cities
attention <- nn_module(
  initialize = function(hidden_size, use_tanh=FALSE, C=10){
    
    # Scaling parameters for escaping gradient vanishing
    self$use_tanh <- use_tanh
    self$C <- C
    
    # Bahdanau components
    self$W_query = nn_linear(hidden_size, hidden_size) |> use_cuda() # linear transformation for query
    self$W_ref   = nn_conv1d(hidden_size, hidden_size, 1, 1) |> use_cuda() # simple 1 dim convolution layer
    
    self$V = torch_zeros(hidden_size, dtype = torch_float(), device=dvc) |> nn_parameter()
    self$V$data()$uniform_(-1/sqrt(hidden_size), 1/sqrt(hidden_size))
  },
  forward = function(query, ref, idxs){
    seq_len  = ref$size(2)
    # Bahdanau algo
    query_nn = self$W_query(query$squeeze(1))$`repeat`(c(seq_len, 1)) # [seq_len x hidden_size] 
    ref_nn = self$W_ref(ref$squeeze(1)$permute(2:1))                  # [hidden_size x seq_len] 
    logits = torch_matmul(self$V, torch_tanh(query_nn$permute(2:1) + ref_nn)) # [seq_len] additive attention scoring
    
    if(self$use_tanh){logits = self$C * torch_tanh(logits)} # alleviate gradient vanishing 
    if(!is.null(idxs)){logits[torch_cat(idxs)] <- -Inf} # masking previous choices
    
    list(ref=ref_nn, logits=nnf_softmax(logits, dim =1)) # attention requires softmax 
  }
)
tst_query <- torch_rand(1, 1, 128, device = dvc) # prepare for test
tst_ref <- torch_rand(1, 16, 128, device = dvc) # prepare for test
tst_idx <- torch_tensor(c(2,5))$to(torch_long(), device = dvc) # prepare for test
attention(128)(tst_query, tst_ref, tst_idx) # test attention

# Pointer module ------------------------------------------------------------------------------------------------------------------------------------------
# The pointer network architecture learns the conditional probability of an
# output sequence with elements that are discrete tokens corresponding to positions
# in an input sequence.
# Inputs: 
#      - inputs: [seq_len, 2] 
#                task formulation in coordinate shape
# Outputs: 
#      - reward: [1] route distance 
#      - probs:  [seq_len] probabilities of selecting cities
#      - idxs:   [seq_len] selected cities
#      - mem_probs: list of [seq_len]
#                complex set of probabilities for each step
pointer_net <- nn_module(
  initialize = function(embedding_size, hidden_size, seq_len, tanh_exploration, use_tanh){
    
    self$embedding_size = embedding_size
    self$hidden_size    = hidden_size
    self$seq_len        = seq_len
    
    self$embedder  = nn_linear(2, embedding_size, bias = F) |> use_cuda()
    self$encoder   = nn_lstm(embedding_size, hidden_size, batch_first=T) |> use_cuda()
    self$decoder   = nn_lstm(embedding_size, hidden_size, batch_first=T) |> use_cuda()
    
    self$pointer   = attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration)
    self$glimpse   = attention(hidden_size, use_tanh=F)
    
    self$decoder_start_input = nn_parameter(torch_zeros(seq_len, embedding_size, device=dvc))
    self$decoder_start_input$data()$uniform_(-1/sqrt(embedding_size), 1/sqrt(embedding_size))
  },
  forward = function(inputs){
    seq_len = self$seq_len
    embedded = self$embedder(inputs)
    
    # encoder_outputs = [seq_len, embedding_size, hidden_size]
    # hidden, context = [1, embedding_size, hidden_size]
    enc_res <- self$encoder(embedded$unsqueeze(1)) 
    
    mem_probs    = NULL
    mem_idxs     = NULL
    
    decoder_input = self$decoder_start_input
    
    for (i in seq_len(seq_len)){
      # query, context = [1, embedding_size, hidden_size]
      dcd_res <- self$decoder(decoder_input$unsqueeze(1), enc_res[[2]])[[2]] 
      
      # 1st attention - glimpse
      glimpse_res <- self$glimpse(dcd_res[[1]], enc_res[[1]], mem_idxs)
      query = torch_matmul(glimpse_res$ref, glimpse_res$logits) # ref x logits
      
      # 2nd attention - pointer
      probs = self$pointer(query, enc_res[[1]], mem_idxs)[[2]] 
      idxs = torch_multinomial(probs, num_samples=1)$to(torch_long())
      
      # Prepare for further steps
      decoder_input = embedded[idxs]
      
      mem_probs <- c(mem_probs, probs) # save intermediate probabilities
      mem_idxs <- c(mem_idxs, idxs) # save city indexes
    }
    
    idxs <- torch_cat(mem_idxs)
    # action_probs <- mapply(mem_idxs, mem_probs, FUN=\(x,y)y[x]) |> torch_cat() # basic R 
    probs <- torch_stack(mem_probs) |> 
      torch_gather(dim = 2, torch_stack(mem_idxs)) |> # index selection 
      torch_squeeze(2) 
    reward = -calc_dist4tnsr(inputs[idxs]) # resulting route distance
    
    list(reward=reward, probs=probs, idxs=idxs, mem_probs=mem_probs)
  }
)
pointer_net(30, 128, 16, 10, TRUE)(torch_rand(16, 2, device=dvc)) # test pointer_net

# Training ----------------------------------------------------------------
# Training function based on policy gradient methods
# Inputs: 
#      - model:  object representing the pointer_net model
#      - inputs: [seq_len, 2] 
#                task formulation in coordinate shape
#      - max_grad_norm: clipping gradient parameter
#      - n_epochs:   the quantity of training iterations
#      - beta: smoothing parameter
# Outputs: 
#      - loss:   training loss across iterations
#      - reward: training reward across iterations
#      - best:   best route found
#      - last:   final route found
#      - mdl:    trained model 
trainer <- function(model, train_data, max_grad_norm=2, n_epochs, beta, logs=TRUE){
  
  # Initialize the optimizer for the model's parameters using Adam optimizer
  actor_optim = optim_adam(model$parameters, lr=1e-4)
  
  # Prepare containers to store training loss and rewards for each epoch
  train_loss = vector(mode = "double", length = n_epochs)
  train_tour = rep(Inf, n_epochs)
  
  epochs = 1 # Epoch counter
  start_time = Sys.time() # record start time for tracking elapsed time
  
  # Initialize the critic's exponential moving average for reward
  critic_exp_mvg_avg = torch_zeros(1, device = dvc)
  
  # Set the model to training mode
  model$train()
  
  # Main training loop for the specified number of epochs
  for (epoch in seq_len(n_epochs)){
    # Log progress every 10 epochs
    if (epoch %% 10 == 0 & logs) 
      cat("Epoch: ", epoch, "| Time elapsed: ", format(round(Sys.time() - start_time, 2)), " | ")
    
    # Get model predictions on the training data
    mdl_res <- model(train_data)
    
    # Update the critic's exponential moving average for the reward
    critic_exp_mvg_avg = critic_exp_mvg_avg * beta + (1 - beta) * mdl_res$reward
    
    # Calculate the advantage: difference between the reward and the critic's estimate
    advantage = mdl_res$reward - critic_exp_mvg_avg
    
    # Calculate the reinforcement signal using log probabilities weighted by the advantage
    reinforce = advantage * torch_log(mdl_res$probs)$sum()
    
    # Reset gradients to avoid accumulation
    actor_optim$zero_grad()
    
    # Compute gradients for the reinforce signal
    reinforce$backward()
    
    # Clip gradients to prevent exploding gradients
    nn_utils_clip_grad_norm_(model$parameters, max_grad_norm, norm_type=2)
    
    # Update model parameters using the optimizer
    actor_optim$step()
    
    # Detach the critic's moving average to stop tracking gradients
    critic_exp_mvg_avg = critic_exp_mvg_avg$detach()
    
    # Convert loss and reward to double for storage
    loss = as.double(reinforce)
    reward = as.double(mdl_res$reward)
    
    # Track the best model based on minimum reward
    if (epochs > 1){
      if (min(train_tour) > reward){best <- mdl_res}
    } else{best <- mdl_res}
    
    # Store the loss and reward for this epoch
    train_loss[epochs] <- loss
    train_tour[epochs] <- reward
    
    # Increment epoch counter
    epochs <- epochs + 1
    
    # Log the loss and reward every 10 epochs
    if (epoch %% 10 == 0 & logs) 
      cat("Loss: ", loss, " | Reward: ", reward, "\n")
  }
  
  # Set the model to evaluation mode
  model$eval()
  
  # Get final predictions from the model
  res <- model(train_data)
  
  # Return a list containing the training results
  list(loss = train_loss, reward = train_tour, best = best, last = res, mdl = model)
}

# Training 
res <- pointer_net(n_cities, 128, n_cities, 10, TRUE) |> # configure model 
  trainer(cities_tnsr, 2, 1000, 0.99) # configure and perform training 

res4save <- res[1:2]
res4save$best$idx <- as_array(res$best$idxs)
res4save$last$idx <- as_array(res$last$idxs)
res4save$best$reward <- as_array(res$best$reward)
res4save$last$reward <- as_array(res$last$reward)

torch_save(res$mdl, "test_samples/pointer_mdl") # save for publishing 
saveRDS(res4save, "test_samples/pointer_res.rds")  # save for publishing 

# torch_load("test_samples/pointer_mdl")
# readRDS("test_samples/pointer_res.rds")

# Evaluate NN ---------------------------------------------------------------------------------------------------------------------------------------------

# Best solution 
p1 <- prep4plot(cities, as_array(res$best$idxs)) |> 
  plot_tour() + 
  labs(title = paste0("Лучшее решение: ", as_array(res$best$reward)|>round(2)), 
       col = "Маршрут", x = "x", y = "y")

# Last solution 
p2 <- prep4plot(cities, as_array(res$last$idxs)) |> 
  plot_tour() +
  labs(title = paste0("Итоговое решение: ", as_array(res$last$reward)|>round(2)), 
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
(p1+p2)/p3 + plot_annotation("Pointer Network")

# Probability matrix for best solution 
res$best$mem_probs |> 
  torch_stack(dim = 2) |> # stack probabilities 
  glimpse_tnsr(rnd = 1)

# Probability matrix for final solution 
res$last$mem_probs |> 
  torch_stack(dim = 2) |>  # stack probabilities 
  glimpse_tnsr(rnd = 1)

# Check generalization capability 
new_task <- generate_task(seed = 2023)  
new_task_tnsr <- torch_tensor(new_task)
new_task_sol <- as_array(res$mdl(new_task_tnsr)$idxs)
new_task_sol_dist <- new_task_tnsr[c(1, new_task_sol)] |> 
  calc_dist4tnsr() |> 
  as_array() |> 
  round(2)

prep4plot(new_task, new_task_sol) |> 
  plot_tour() +
  labs(title = paste0("Итоговое решение: ", -new_task_sol_dist), 
       col = "Маршрут", x = "x", y = "y")

# Wrap model and calculate batch ------------------------------------------
get_pointer <- function(task){
  # browser()
  cities_tnsr <- torch_tensor(task, device=dvc) # tensor version of task
  n_cities <- nrow(task)
  start_time = Sys.time() 
  
  # Training 
  res <- pointer_net(n_cities, 128, n_cities, 10, TRUE) |> # configure model 
    trainer(cities_tnsr, 2, 1000, 0.99, logs=F)   # configure and perform training 
  
  duration <- Sys.time() - start_time
  
  tibble::tibble(model = "Pointer Net", duration = duration, 
                 distance = as_array(res$best$reward), route = list(as_array(res$best$idxs)))
}

get_pointer(generate_task())

res_16 <- calc_tours(get_pointer, n_cities = 16, runs = 10)
res_32 <- calc_tours(get_pointer, n_cities = 32, runs = 10) 
res_64 <- calc_tours(get_pointer, n_cities = 64, runs = 10) 

saveRDS(res_16, "test_results/pointer_16nodes.rds")
saveRDS(res_32, "test_results/pointer_32nodes.rds")
saveRDS(res_64, "test_results/pointer_64nodes.rds")





