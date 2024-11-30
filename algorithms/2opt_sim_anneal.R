library(ggplot2)
source("algorithms/functions.R")
# source("https://raw.githubusercontent.com/dkibalnikov/ai_optimization/refs/heads/main/algorithms/functions.R")

# Create task environment ---------------------------------------------------------------------------------------------------------------------------------
n_cities <- 16

cities <- generate_task()

p0 <- as.data.frame(cities) |> 
  transform(n=seq_len(n_cities)) |>
  ggplot(aes(x, y)) + 
  geom_point() + 
  geom_point(data = data.frame(n=1, x=0, y=0), shape = 21, size = 5, col = "firebrick") +
  geom_text(aes(label = n), nudge_x = 3)

p0

dist_mtrx <- as.matrix(dist(cities))

# Functions -----------------------------------------------------------------------------------------------------------------------------------------------
# function to generate a random route
random_route <- function(cities){
  n_cities <- seq_len(nrow(cities))
  route <- sample(n_cities[-1])
  route <- c(1,route,1) # return back home
  return(route)
}
rand_route <- random_route(cities)

# function to swap two cities within a route (= rewire)
swap <- function(route, i, k){
  c(route[1:(i-1)], route[k:i], route[(k+1):length(route)] )
}
swap(rand_route, 8, 2) 

# Closest neighborhood -----------------------------------------------------------------------------------------------------------------------------------
res_closest <- get_route4mtrx(-dist_mtrx)

sol_dist_closest <- calc_dist4mtrx(dist_mtrx, res_closest) |> round(2)

prep4plot(cities, res_closest) |> 
  plot_tour() + 
  labs(title = paste0("Closes neighborhood solution: ", sol_dist_closest)) 

# 2-opt ---------------------------------------------------------------------------------------------------------------------------------------------------
# two-opt algorithm
two_opt <- function(cities, dist_mtrx){
  # browser()
  
  # start with random route
  best_route <- random_route(cities)
  min_d <- calc_dist4mtrx(dist_mtrx, best_route)
  
  # variable tracking
  track_dist <- c()
  
  while(T){ # while-loop to perform complete 2-opt swap
    old_d <- min_d # record distance before looping through i and k
    break_loop <- F # for-loops through values for i and k
    for(i in 2:(nrow(cities)-1)) {
      for(k in (i+1):nrow(cities)) {
        # perform swap
        route <- swap(best_route, i, k)
        new_d <- calc_dist4mtrx(dist_mtrx, route)
        
        # update distance 
        if(new_d < min_d){
          min_d <- new_d
          best_route <- route
          # cat("Best route: ", min_d, "\n")
          break_loop <- T
          break() # break out of inner loop
        } # end outer if-statement
      } 
      # if(old_d == new_d & new_d < 407) browser()
      if(break_loop) break() # break out of outer loop
    } 
    if(old_d == min_d) break() # check if the for-loops made any improvements
    track_dist <- c(track_dist, new_d) # update on variable tracking
  } 
  list(distance = min_d, route = best_route, track_dist = track_dist)
}

system.time(res_two_opt <- two_opt(cities, dist_mtrx))

prep4plot(cities, res_two_opt$route[-length(res_two_opt$route)]) |> 
  plot_tour() + 
  labs(title = paste0("2-opt solution: ", round(res_two_opt$distance, 2))) 

# Simulated annealing -------------------------------------------------------------------------------------------------------------------------------------
sim_anneal <- function(cities, dist_mtrx, temp=1e4, cooling=5e-3, break_after=1e2){
  # start with random route
  best_route <- random_route(cities)
  min_d <- calc_dist4mtrx(dist_mtrx, best_route)
  
  # variable tracking
  track_temp <- NULL
  track_prob <- NULL
  track_dist <- NULL
  
  # iterative loop
  stable_count <- 0
  while(stable_count < break_after){
    
    # conduct swap
    ik <- sort(sample(2:nrow(cities), 2))
    new_route <- swap(best_route, i=ik[1], k=ik[2])
    new_d <- calc_dist4mtrx(dist_mtrx, new_route)
    
    # probability of adjusting route
    improvement <- min_d - new_d
    p_adjust <- ifelse(improvement > 0, 1, exp(improvement/temp))
    
    # adjust route?
    adjust <- ifelse(p_adjust >= runif(1,0,1), T, F)
    
    # if adjustment
    if(adjust) {
      # if(new_d/min_d < 0.95)cat("Best route: ", min_d, "| Temperature: ", temp, "\n")
      best_route <- new_route
      min_d <- new_d
      stable_count <- 0
    } else {stable_count <- stable_count+1}
    # update on variable tracking
    track_temp <- c(track_temp, temp)
    track_prob <- c(track_prob, p_adjust)
    track_dist <- c(track_dist, new_d)
    # cool down
    temp <- temp*(1-cooling)
  } # end of iterative loop
  list(distance = min_d, route = best_route, track_temp = track_temp, track_prob = track_prob, track_dist = track_dist)
}
system.time(res_sim_ann <- sim_anneal(cities, dist_mtrx))
prep4plot(cities, res_sim_ann$route[-length(res_sim_ann$route)]) |> 
  plot_tour() + 
  labs(title = paste0("Simulated annealing solution: ", round(res_sim_ann$distance, 2))) 


# Wrap model and calculate batch ------------------------------------------

get_closest <- function(task){
  dist_mtrx <- as.matrix(dist(task))
  
  start_time = Sys.time() 
  res <- get_route4mtrx(-dist_mtrx)
  duration <- Sys.time() - start_time
  
  tibble::tibble(model = "Closest neighborhood", duration = duration, distance = calc_dist4mtrx(dist_mtrx, res), route = list(res))
}
get_closest(cities)
res_16 <- calc_tours(get_closest, n_cities = 16)
res_32 <- calc_tours(get_closest, n_cities = 32) 
res_64 <- calc_tours(get_closest, n_cities = 64) 

saveRDS(res_16, "test_results/closest_16nodes.rds")
saveRDS(res_32, "test_results/closest_32nodes.rds")
saveRDS(res_64, "test_results/closest_64nodes.rds")

get_2opts <- function(task){
  dist_mtrx <- as.matrix(dist(task))
  
  start_time = Sys.time() 
  res <- two_opt(task, dist_mtrx) 
  duration <- Sys.time() - start_time
  
  tibble::tibble(model = "2 opt", duration = duration, distance = res$distance, route = list(res$route))
  
}
get_2opts(cities)
res_16 <- calc_tours(get_2opts, n_cities = 16, runs = 10)
res_32 <- calc_tours(get_2opts, n_cities = 32, runs = 10) 
res_64 <- calc_tours(get_2opts, n_cities = 64, runs = 10) 

saveRDS(res_16, "test_results/2opts_16nodes.rds")
saveRDS(res_32, "test_results/2opts_32nodes.rds")
saveRDS(res_64, "test_results/2opts_64nodes.rds")

get_sim_anneal <- function(task){
  dist_mtrx <- as.matrix(dist(task))
  
  start_time = Sys.time() 
  res <- sim_anneal(task, dist_mtrx) 
  duration <- Sys.time() - start_time
  
  tibble::tibble(model = "2 opt + Sim. annealing", duration = duration, distance = res$distance, route = list(res$route))
  
}
get_sim_anneal(cities)
res_16 <- calc_tours(get_sim_anneal, n_cities = 16, runs = 10)
res_32 <- calc_tours(get_sim_anneal, n_cities = 32, runs = 10) 
res_64 <- calc_tours(get_sim_anneal, n_cities = 64, runs = 10) 

saveRDS(res_16, "test_results/sim_anneal_16nodes.rds")
saveRDS(res_32, "test_results/sim_anneal_32nodes.rds")
saveRDS(res_64, "test_results/sim_anneal_64nodes.rds")
