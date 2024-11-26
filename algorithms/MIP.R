library(ompr)
library(ompr.roi)
library(ROI.plugin.glpk)
library(ggplot2)
source("algorithms/functions.R")

# Create task environment ---------------------------------------------------------------------------------------------------------------------------------
n_cities <- 16

cities <- generate_task(n_cities)

p0 <- as.data.frame(cities) |> 
  transform(n=seq_len(n_cities)) |>
  ggplot(aes(x, y)) + 
  geom_point() + 
  geom_point(data = data.frame(n=1, x=0, y=0), shape = 21, size = 5, col = "firebrick") +
  geom_text(aes(label = n), nudge_x = 3)

p0

dist_mtrx <- as.matrix(dist(cities))

# MIP model ---------------------------------------------------------------

# This is classical MIP solution for TSP 
# https://dirkschumacher.github.io/ompr/articles/milp-modelling.html

mdl_MIP <- MIPModel() |>
  # we create a variable that is 1 iff we travel from city i to j
  add_variable(x[i, j], i = 1:n_cities, j = 1:n_cities, type = "integer", lb = 0, ub = 1) |>
  # a helper variable for the MTZ formulation of the tsp
  add_variable(u[i], i = 1:n_cities, lb = 1, ub = n_cities) |> 
  # minimize travel distance
  set_objective(sum_expr(dist_mtrx[i, j] * x[i, j], i = 1:n_cities, j = 1:n_cities), "min") |>
  # you cannot go to the same city
  set_bounds(x[i, i], ub = 0, i = 1:n_cities) |>
  # leave each city
  add_constraint(sum_expr(x[i, j], j = 1:n_cities) == 1, i = 1:n_cities) |>
  # visit each city
  add_constraint(sum_expr(x[i, j], i = 1:n_cities) == 1, j = 1:n_cities) |>
  # ensure no subtours (arc constraints)
  add_constraint(u[i] >= 2, i = 2:n_cities) |> 
  add_constraint(u[i] - u[j] + 1 <= (n_cities - 1) * (1 - x[i, j]), i = 2:n_cities, j = 2:n_cities)

mdl_MIP

# Solve and examinate results ---------------------------------------------
system.time(res_MIP <- solve_model(mdl_MIP, with_ROI(solver = "glpk", verbose = TRUE)))

solution <- get_solution(res_MIP, x[i, j]) |> 
  subset(value > 0) 

MIP_route <- tidyr::pivot_wider(solution, id_cols = "i", names_from = "j", values_fill = 0) |> 
  dplyr::arrange(i) |> 
  dplyr::select(-i) |>
  as.matrix() |> 
  get_route4mtrx() 
  
prep4plot(cities, MIP_route) |>
  plot_tour() +
  labs(title = paste0("MIP solution: ", round(res_MIP$objective_value, 2))) 

# Wrap model and calculate batch  -----------------------------------------

get_MIP <- function(task){
  
  dist_mtrx <- as.matrix(dist(task))
  n_cities <- nrow(task)
  
  start_time = Sys.time()
  
  mdl_MIP <- MIPModel() |>
    # we create a variable that is 1 iff we travel from city i to j
    add_variable(x[i, j], i = 1:n_cities, j = 1:n_cities, type = "integer", lb = 0, ub = 1) |>
    # a helper variable for the MTZ formulation of the tsp
    add_variable(u[i], i = 1:n_cities, lb = 1, ub = n_cities) |> 
    # minimize travel distance
    set_objective(sum_expr(dist_mtrx[i, j] * x[i, j], i = 1:n_cities, j = 1:n_cities), "min") |>
    # you cannot go to the same city
    set_bounds(x[i, i], ub = 0, i = 1:n_cities) |>
    # leave each city
    add_constraint(sum_expr(x[i, j], j = 1:n_cities) == 1, i = 1:n_cities) |>
    # visit each city
    add_constraint(sum_expr(x[i, j], i = 1:n_cities) == 1, j = 1:n_cities) |>
    # ensure no subtours (arc constraints)
    add_constraint(u[i] >= 2, i = 2:n_cities) |> 
    add_constraint(u[i] - u[j] + 1 <= (n_cities - 1) * (1 - x[i, j]), i = 2:n_cities, j = 2:n_cities)
  
  duration <- Sys.time() - start_time
    
  res_MIP <- solve_model(mdl_MIP, with_ROI(solver = "glpk", verbose = FALSE))

  route <- get_solution(res_MIP, x[i, j]) |> 
    subset(value > 0) |> 
    tidyr::pivot_wider(id_cols = "i", names_from = "j", values_fill = 0) |> 
    dplyr::arrange(i) |> 
    dplyr::select(-i) |>
    as.matrix() |> 
    get_route4mtrx() 
  
  tibble::tibble(model = "MIP", duration = duration, distance = res_MIP$objective_value, route = list(route))
}
tst <- get_MIP(cities)

res_16 <- calc_tours(get_MIP, n_cities = 16)
# res_32 <- calc_tours(get_MIP, n_cities = 32) # don't even try 

saveRDS(res_16, "test_results/MIP_16nodes.rds")
