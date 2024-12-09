library(igraph)
library(tidyverse)
library(lpSolve)
source("algorithms/functions.R")


# Create task environment ---------------------------------------------------------------------------------------------------------------------------------
n_cities <- 16

cities <- generate_task()

cities_df <- as.data.frame(cities) |> 
  transform(n=seq_len(n_cities)) 

p0 <- ggplot(cities_df, aes(x, y)) + 
  geom_point() + 
  geom_point(data = data.frame(n=1, x=0, y=0), shape = 21, size = 5, col = "firebrick") +
  geom_text(aes(label = n), nudge_x = 3)

p0

dist_mtrx <- as.matrix(dist(cities))


# Generate minimum spanning tree (MST) ------------------------------------
G <- graph_from_adjacency_matrix(dist_mtrx, mode='undirected', weighted = TRUE)
MST <- mst(G)

# Find odd vertices -------------------------------------------------------
odd_vertices <- V(MST)[degree(MST) %% 2 == 1] |> as.integer()
# Number of vertices (odd-degree vertices)
n <- length(odd_vertices)

# Visualization
MST_df <- igraph::as_data_frame(MST) |> 
  mutate(across(1:2, as.integer)) |>
  left_join(cities_df, by=c("from"="n")) |> 
  left_join(cities_df, by=c("to"="n"), suffix = c("_from", "_to")) 
  
p0 + 
  geom_segment(data=MST_df, aes(x = x_from, y = y_from, xend = x_to, yend = y_to)) + 
  geom_point(data = cities[odd_vertices,], col="firebrick", size = 5, alpha = .5)


# Find Minimum Weight Perfect Matching (MPM) ------------------------------
# Define distance matrix for odd vertices
odd_dist_mtrx <- cities[odd_vertices, ] |> dist() |> as.matrix()
# Flatten the upper triangle of the distance matrix to form the cost vector
cost_vector <- odd_dist_mtrx[lower.tri(odd_dist_mtrx)]

# Set up the constraint matrix
constraint_matrix <- matrix(0, nrow = n, ncol = length(cost_vector))

# Generate all pairs (combinations of 2 vertices from n)
combinations <- combn(n, 2)

for (col in 1:ncol(combinations)) {
  constraint_matrix[combinations[1, col], col] <- 1
  constraint_matrix[combinations[2, col], col] <- 1
}

# Solve the linear programming problem
solution <- lp("min", cost_vector, constraint_matrix, 
               const.dir = rep("=", n), 
               const.rhs = rep(1, n), # Each vertex must be matched exactly once
               binary.vec = 1:length(cost_vector))

cat("Total cost of the matching: ", solution$objval, "\n")

matched_pairs <- combinations[1:2, which(solution$solution == 1)] 

# Visualization
matched_pairs_df <- data.frame(from=odd_vertices[matched_pairs[1,]], to=odd_vertices[matched_pairs[2,]]) 
MPM_df <- igraph::as_data_frame(MST) |> 
  mutate(across(1:2, as.integer)) |> 
  bind_rows(matched_pairs_df) |> 
  left_join(cities_df, by=c("from"="n")) |> 
  left_join(cities_df, by=c("to"="n"), suffix = c("_from", "_to")) |> 
  mutate(wired = if_else(is.na(weight), "yes", "no"))

p0 + 
  geom_segment(data=MPM_df, aes(x = x_from, y = y_from, xend = x_to, yend = y_to, col = wired)) + 
  geom_point(data = cities[odd_vertices,], col="firebrick", size = 5, alpha = .5) + 
  scale_color_manual(values = c(yes="firebrick", no= "black")) 

# Find eulerian path ------------------------------------------------------
EP <- add_edges(MST, odd_vertices[as.vector(matched_pairs)]) |>
  eulerian_path()

# EP <- graph_from_data_frame(MPM_df, directed = FALSE) |> 
#   eulerian_path()

p0 + 
  geom_path(data = cities_df[names(EP$vpath),], aes(col=seq_along(EP$vpath))) +
  scale_color_viridis_c() +
  labs(col="Очередность")

# Find the Hamiltonian path -----------------------------------------------
shortcut_path <- function(EP){
  HP <- c()
  for(vertex in EP){
    if (!vertex %in% HP | vertex == EP[1L]){
      HP <- c(HP, vertex)
    }
  }
  return(HP)
}

HP <- shortcut_path(names(EP$vpath)|> as.integer())

p0 + 
  geom_path(data = cities_df[HP,], aes(col=seq_along(HP))) +
  scale_color_viridis_c() +
  labs(col="Очередность", 
       title = paste0("Cristofidies solution: ", calc_dist4mtrx(dist_mtrx, HP)|>round(2)))

# Wrap model and calculate batch ------------------------------------------
get_cristo <- function(task){
  # browser()
  dist_mtrx <- as.matrix(dist(task))
  
  start_time = Sys.time() 
  
  # Generate minimum spanning tree (MST)
  G <- graph_from_adjacency_matrix(dist_mtrx, mode='undirected', weighted = TRUE)
  MST <- mst(G)
  
  # Find odd vertices 
  odd_vertices <- V(MST)[degree(MST) %% 2 == 1] |> as.integer()
  odd_dist_mtrx <- task[odd_vertices, ] |> dist() |> as.matrix()
  n_odd <- length(odd_vertices)
  
  # Find Minimum Weight Perfect Matching (MPM) 
  cost_vector <- odd_dist_mtrx[lower.tri(odd_dist_mtrx)]
  constraint_matrix <- matrix(0, nrow = n_odd, ncol = length(cost_vector))
  combinations <- combn(n_odd, 2)
  for (col in 1:ncol(combinations)) {
    constraint_matrix[combinations[1, col], col] <- 1L
    constraint_matrix[combinations[2, col], col] <- 1L
  }
  solution <- lp("min", cost_vector, constraint_matrix, 
                 const.dir = rep("=", n_odd), 
                 const.rhs = rep(1, n_odd), # Each vertex must be matched exactly once
                 binary.vec = seq_along(cost_vector))
  matched_pairs <- combinations[1:2, which(solution$solution == 1)] 
  
  # Find eulerian path 
  EP <- add_edges(MST, odd_vertices[as.vector(matched_pairs)]) |>
    eulerian_path()
  EP <- names(EP$vpath)|> as.integer()
  
  # Find the Hamiltonian path 
  HP <- c()
  for(vertex in EP){
    if (!vertex %in% HP | vertex == EP[1L]){
      HP <- c(HP, vertex)
    }
  }
  
  duration <- Sys.time() - start_time
  
  tibble::tibble(model = "Cristofidies", duration = duration, distance = calc_dist4mtrx(dist_mtrx, HP), route = list(HP))
}

get_cristo(generate_task())

res_16 <- calc_tours(get_cristo, n_cities = 16)
res_32 <- calc_tours(get_cristo, n_cities = 32) 
res_64 <- calc_tours(get_cristo, n_cities = 64) 

saveRDS(res_16, "test_results/cristo_16nodes.rds")
saveRDS(res_32, "test_results/cristo_32nodes.rds")
saveRDS(res_64, "test_results/cristo_64nodes.rds")

