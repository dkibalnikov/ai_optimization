library(ggplot2)

# Terms 
# TASK[matrix or tensor] - just set of cities (points) coordinates
# ROUTE[vector] - the order of visits (solution)
# TOUR[matrix or tensor] - the coordinates shaped in solution sequence 

# to generate task (cities layout)
generate_task <- function(n_cities=16, seed=2021){
  set.seed(seed)
  data.frame(x=0, y= 0) |> 
    rbind(data.frame(x = runif(n_cities-1, max = 100), y = runif(n_cities-1, max = 100))) |> 
    as.matrix()
}

# function to compute the distance of a route
calc_dist4mtrx <- function(dist_mtrx, route){
  # route <- random_route(cities)
  mapply(\(x,y)dist_mtrx[x,y], route, c(route[-1], route[1])) |> sum()
}

# function to compute the distance of a route for solution tensor 
calc_dist4tnsr <- function(tour){
  # solution_smpl <- torch_cat(res1$actions)
  
  n <- tour$size(1) - 1
  
  indx <- c(cumsum(c(1, n:2)), n)
  -torch_pdist(tour)[indx]$sum()
}

# derives tour from Q matrix or any other prob matrix
get_route4mtrx <- function(Q_mtrx){
  # browser()
  diag(Q_mtrx) <- -Inf
  state <- apply(Q_mtrx, 2, max) |> which.max()
  n_seq <- nrow(Q_mtrx)
  mem <- rep(NA_integer_, n_seq)
  mem[1] <- state
  next_state <- NULL
  
  for(i in 2:n_seq){
    next_state <- which.max(Q_mtrx[,state])
    Q_mtrx[state, ] <- -Inf
    mem[i] <- next_state
    state <- next_state
  }
  mem
}

# derives tour from state net 
get_route4state_net <- function(state_net){
  # browser()
  # state_net <- res$state_net
  mem <- 1L
  n_seq <- state_net(1L)$size()
  
  for(i in 2:n_seq){
    state_opts <- state_net(mem)
    state_opts[mem] <- -Inf
    state <- state_opts$max(1)[[2]]|>as_array()
    mem <- c(mem, state)
  }
  mem
}

# prepares table with coordinates and order in required sequence
prep4plot <- function(task, route){
  # browser()
  tour <- task[route,]
  depot_pos <- which(tour[,1] == 0)
  if(depot_pos == 1){
    new_order <- seq_along(route)
    buity_tour <- tour
    }else{
    new_order <- c(depot_pos:nrow(tour), 1:(depot_pos - 1))
    buity_tour <-  tour[new_order,]
    }
  
  rbind(buity_tour, c(0, 0)) |> 
    tibble::as_tibble(.name_repair = ~c("x", "y")) |>
    dplyr::mutate(order = dplyr::row_number()) |>
    dplyr::mutate(init_order = c(route[new_order], 1))
}

# function to plot solution
plot_tour <- function(tour_prep, init_order=TRUE){
  n_cities <- nrow(tour_prep) - 1 
  
  tour_prep |> 
    ggplot(aes(x, y, col = order)) + 
    list(
      geom_point(), 
      geom_path(arrow = arrow(angle = 20, length = unit(0.10, "inches"),
                              ends = "last", type = "open")), 
      geom_text(aes(label = order), nudge_x = 3, check_overlap = T), 
      if(init_order)geom_text(aes(label = init_order), nudge_x = -3, check_overlap = T, col = "firebrick")else NULL, 
      scale_color_binned(breaks = 1:n_cities),
      guides(color = guide_colourbar(barheight =n_cities), type = "viridis"), 
      geom_point(data = data.frame(order=1, x=0, y=0), shape = 21, size = 5, col = "firebrick"),
      theme(legend.position = "none")
    )
}

# to calculate multiple optimization runs for multiple seeds 
calc_tours <- function(opt_fun, seeds=2021:2030, n_cities = 16, runs = 1){
  purrr::map(seeds, .progress = TRUE, \(x){
    task <- generate_task(n_cities, x)
    # browser()
    seq_len(runs) |> 
      purrr::map(\(y)opt_fun(task)|>dplyr::mutate(run=y)) |>  
      purrr::list_rbind() |> 
      dplyr::mutate(seed=x)
  }) |> 
    purrr::list_rbind()
}

# to calculate multiple optimization runs for multiple seeds in parallel
calc_tours_multi <- function(opt_fun, seeds=2021:2030, n_cities = 16, runs = 1, workers = 4){
  future::plan(multisession, workers = workers)
  
  furrr::future_map(seeds, .progress = TRUE, \(x){
    task <- generate_task(n_cities, x)
    # browser()
    seq_len(runs) |> 
      purrr::map(\(y)opt_fun(task)|>dplyr::mutate(run=y)) |>  
      purrr::list_rbind() |> 
      dplyr::mutate(seed=x)
  }) |> 
    purrr::list_rbind()
}

# to check tensor visually 
glimpse_tnsr <- function(tnsr, rnd = 2){
  as.matrix(tnsr) |> 
    apply(2, \(x)round(x, rnd)) |> 
    as.data.frame() |> 
    emphatic::hl(scale_color_viridis_c())
}

# formatted tensor viewer
check_tnsr <- function(tnsr, rnd = 2){
  tnsr |> 
    as_array() |> 
    apply(2, \(x)round(x, rnd)) |> 
    as.data.frame() |>
    emphatic::hl(scale_color_viridis_c())
}

# get route out of NN
get_route4tnsr <- function(state_net, cities_tnsr){
  # state_net <- res$state_net
  mem <- 1L
  n_seq <- cities_tnsr$size(1)
  
  for(i in 2:n_seq){
    state_opts <- state_net(mem, cities_tnsr)
    state_opts[mem] <- -Inf
    state <- state_opts$max(1)[[2]]|>as_array()
    mem <- c(mem, state)
  }
  mem
}

# use GPU if possible
use_cuda <- function(obj){
  if (cuda_is_available()){obj<-obj$cuda()}
  obj
} 
