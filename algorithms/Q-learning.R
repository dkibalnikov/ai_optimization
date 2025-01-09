source("algorithms/functions.R") # helper functions within AI optimization  
library(patchwork)
library(data.table)

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

dist_mtrx <- as.matrix(dist(cities))

# Q-learning  -------------------------------------------------------------
Q_train <- function(cities, pars, epochs, dist_mtrx){
  n_cities = nrow(cities)
  n_seq = seq_len(n_cities)
  Q_mtrx = matrix(data = 0L, n_cities, n_cities)
  rewards_curve = rep(-Inf, epochs)
  best_sol = integer(n_cities)
  # browser()
  
  for(j in seq_len(epochs)){
    # Initialize epoch environment
    state = sample(n_cities, 1)
    mem = state
    epoch_reward = 0
    
    for(i in 2:n_cities){
      # Define next best action either in greedy way either explore space 
      if(runif(1) > pars$epsilon){ # exploitation
        q = Q_mtrx[state,] # get Q Vector in particular state
        q[mem] = -Inf  # avoid already visited states
        action = which.max(q)
      }
      else{  # exploration
        options = n_seq[which(!n_seq %in% mem)] # avoid already visited states
        action = ifelse(length(options) == 1, options, sample(options, size = 1))
      }
      
      reward = -dist_mtrx[state, action] # get reward for such a action
      
      # c(Q_mtrx, pars$epsilon) %<-% train(state, action, reward, Q_mtrx, pars) # update knowledge in the Q-table
      next_q = Q_mtrx[action,]
      mem = c(mem, action) # append new_state to memory 
      
      # Update Q-learning matrix according to Bellman Equation
      if(i == n_cities){
        Q_mtrx[state, action] = Q_mtrx[state, action] + pars$lr*(reward + pars$gamma*next_q[mem[1]] - Q_mtrx[state, action])
      }else{
        next_q[mem] = -Inf  # avoid already visited states
        Q_mtrx[state, action] = Q_mtrx[state, action] + pars$lr*(reward + pars$gamma*which.max(next_q) - Q_mtrx[state, action])
      }
      epoch_reward = epoch_reward + reward # update rewards
      state = action # switch to new state
    }
    # Lower epsilon for less exploration probability 
    if(pars$epsilon > pars$epsilon_min){pars$epsilon = pars$epsilon_decay*pars$epsilon}
    if(best_sol[1]==0){best_sol <- mem}
    if(max(rewards_curve, na.rm = T) < epoch_reward){best_sol <- mem}
    
    rewards_curve[j] <- epoch_reward
  }
  list(Q_mtrx=Q_mtrx, last=mem, reward=rewards_curve, best=best_sol)
}
pars <- list(epsilon = 1, epsilon_min = 0.01, epsilon_decay = 0.99, 
             gamma = 0.99, lr = 0.99)
system.time(res <- Q_train(cities, pars, 1000, dist_mtrx))


# Evaluation --------------------------------------------------------------

# Best solution 
p1 <- prep4plot(cities, res$best) |> 
  plot_tour() + 
  labs(title = paste0("Лучшее решение: ", calc_dist4mtrx(dist_mtrx, res$best) |>round(2)), 
       col = "Маршрут", x = "x", y = "y")

res$final <- get_route4mtrx(res$Q_mtrx)

# Last solution 
p2 <- prep4plot(cities, res$final) |> 
  plot_tour() +
  labs(title = paste0("Итоговое решение: ", calc_dist4mtrx(dist_mtrx, res$final)|>round(2)), 
       col = "Маршрут", x = "x", y = "y")

# Training perfomance metrics 
p3 <- tibble::tibble(n = seq_along(res$reward), reward = res$reward) |>
  tidyr::pivot_longer(-1, names_to = "variable") |>
  ggplot(aes(n, value, col = variable)) + 
  geom_line(alpha = .3) + 
  geom_smooth(formula = y ~ s(x, bs = "cs"), method = 'gam') +
  facet_wrap(~variable, nrow = 2, scales = "free") +
  geom_point(data = ~dplyr::filter(., n==which.min(res$reward)), col = "firebrick") + 
  labs(title = "Характеристики обучения Q-learning", col = "", y = "Значение", x = "Итерация обучения")

# Combine plots to one 
(p1+p2)/p3 + plot_annotation("Q-learning")

# Check Q-learning matrix 
apply(res$Q_mtrx, 2, round) |> 
  as.data.frame() |>
  emphatic::hl(scale_color_viridis_c())

# Animated solution  ------------------------------------------------------
data4anim <- res$Q_mtrx[,res$final] |> # take cols from Q matrix in sol order
  as.data.table() |> 
  melt(measure.vars = 1:n_cities, value.name = "qval",  variable.name="order") |> # stack cols 
  _[, id:=rep(seq_len(n_cities), n_cities)] |> # set initial ids
  _[, order:=as.integer(order)] |> # set the order 
  merge(data.table(cities)[,id:=.I][res$final,][,order:=c(16, 1:(n_cities-1))][], 
        by = c("id"), all.x=TRUE,suffixes=c("","_")) |> # add coordinates info
  _[,`:=`(step=fifelse(order==order_, TRUE, FALSE))] |> # define records with transition
  _[,step_:=Reduce(x=step, \(input, output){if(output) TRUE else input}, accumulate = T), by = "id"] |> # mark records after transition
  _[, `:=`(x_=fifelse(step_,x,NA), y_=fifelse(step_,y,NA))] |> # define vars for geom_path()
  _[,step_:=fifelse(step_==step, FALSE, TRUE), by = "id"] |> # mask transition record
  _[,qval_:=fifelse(!step_,qval,NA)]  %>% # deifine masked q vals 
  list(., # outcome from previous step 
       split(., by =  "order")[[n_cities]][,order:=order+1], # get last order part
       data.table(order = 17, order_ = 17, x_=cities[res$final[2],1], y_=cities[res$final[2],2])) |> # add final destination 
  rbindlist(fill=TRUE) |> 
  setkey(order, order_) # ordering is vital for animation 


p <- data4anim[!is.na(x_)][order(order)] |> 
  ggplot(aes(col = order)) +
  geom_point(aes(x, y, size = qval_), alpha=.5, data =data4anim ) + 
  geom_path(aes(x_, y_), arrow=arrow(angle = 20, length = unit(0.20, "inches"),
                                       ends = "last", type = "open")) +
  geom_point(aes(x, y)) +
  geom_text(aes(x, y, label = id), nudge_x = 3, nudge_y = 3, size =5, col= "black") +
  #scale_size_area() +
  scale_size_continuous(range = c(0,10)) +
  #scale_color_viridis_c()+
  gganimate::transition_states(order, transition_length = 1, state_length = 100, wrap = F) +
  gganimate::enter_appear(early = T)

gganimate::animate(p, renderer = gifski_renderer())





# Wrap model and calculate batch ------------------------------------------
get_Qlearn <- function(task){
  # browser()
  dist_mtrx <- as.matrix(dist(task))
  
  start_time = Sys.time() 
  res <- Q_train(cities = task, 
                 pars = list(epsilon = 1, epsilon_min = 0.01, epsilon_decay = 0.99, 
                            gamma = 0.99, lr = 0.99),
                 epochs=1000, dist_mtrx = dist_mtrx)
  
  duration <- Sys.time() - start_time
  
  final <- get_route4mtrx(res$Q_mtrx)
  
  tibble::tibble(model = "Q-learning", duration = duration, 
                 distance = calc_dist4mtrx(dist_mtrx, final), route = list(final))
}

get_Qlearn(generate_task())

res_16 <- calc_tours(get_Qlearn, n_cities = 16, runs = 10)
res_32 <- calc_tours(get_Qlearn, n_cities = 32, runs = 10) 
res_64 <- calc_tours(get_Qlearn, n_cities = 64, runs = 10) 

saveRDS(res_16, "test_results/qlearn_16nodes.rds")
saveRDS(res_32, "test_results/qlearn_32nodes.rds")
saveRDS(res_64, "test_results/qlearn_64nodes.rds")
