# This is the components which did not go to main script 


# Alternative way to fill constraint matrix -------------------------------
# Replaced by Pascal triangle with option to prohibit existing edges

# Fill in the constraint matrix
col_index <- 1
for (i in 1:(n-1)) {
  for (j in (i+1):n) {
    constraint_matrix[i, col_index] <- 1
    constraint_matrix[j, col_index] <- 1
    col_index <- col_index + 1
  }
}


# Extract the matching pairs
matching_edges <- which(solution$solution == 1)
cost_vector[matching_edges]

matched_pairs <- c()
col_index <- 1
for (i in 1:(n-1)) {
  for (j in (i+1):n) {
    if (solution$solution[col_index] == 1) {
      matched_pairs <- rbind(matched_pairs, c(odd_vertices[i], odd_vertices[j]))
    }
    col_index <- col_index + 1
  }
}

matched_pairs_df <- as.data.frame(matched_pairs) |> 
  `names<-`(c("from", "to")) |>
  mutate(wired=TRUE)


# Combinations with removed repeated edges --------------------------------
# Add one more row to indicate edges which must not be chosen 
combinations <- t(combinations) |> 
  as_tibble() |>
  mutate(from = odd_vertices[V1], to = odd_vertices[V2]) |> 
  left_join(MST_df[, 1:3], by = c("from", "to")) |> 
  mutate(constr = if_else(is.na(weight), 1, 0)) |> 
  select(c(1:2, 6)) |> 
  as.matrix() |> 
  t()

# Initialize constraint matrix
constraint_matrix <- matrix(0, nrow = n, ncol = ncol(combinations))

# Fill the matrix
for (col in 1:ncol(combinations)) {
  constraint_matrix[combinations[1, col], col] <- combinations[3, col]
  constraint_matrix[combinations[2, col], col] <- combinations[3, col]
}

# Find neighbours ---------------------------------------------------------
# Replaced by igraph::neighborhood()

MST <- path_df[,1:2]
# find neigbours
neighbours = list()
for(edge in seq_len(nrow(MST))){
  s <- MST[edge, 1] |> as.character()
  f <- MST[edge, 2] |> as.character()
  
  if(!s %in% names(neighbours)){neighbours[[s]] <- f}else{
    neighbours[[s]] <- c(neighbours[[s]], f)
  }
  if(!f %in% names(neighbours)){neighbours[[as.character(f)]] <- s}else{
    neighbours[[f]] <- c(neighbours[[f]], s)
  }
}

neighbours <- lapply(neighbours, \(x)x[!duplicated(x)])
 
  

# Logic to connect unwired vertices to closest ones -----------------------
# It brakes condition needed to build eulerian path

unlnkd <- V(MST_MWM)[degree(MST_MWM) == 1]
unlnkd_neighbor <- neighborhood(MST_MWM, nodes = unlnkd) |> 
  sapply(\(x)as.vector(names(x)) |> as.integer()) |> 
  t() |> 
  as_tibble() |> 
  `names<-`(c("from", "to"))|> 
  mutate(neighbor=TRUE) 

diag(dist_mtrx) <- 1000

fix_edges <- dist_mtrx[, names(unlnkd)] |> 
  apply(2, \(x)order(x)[1:2]) |> 
  as_tibble() |> 
  pivot_longer(everything(), names_to = "from", values_to = "to", names_transform = as.integer) |>
  left_join(unlnkd_neighbor, by=c("from", "to")) |> 
  filter(is.na(neighbor)) |>
  select(1:2) 

path_df2 <- path_df[,1:2] |>
  distinct() |> 
  bind_rows(fix_edges)  
  
  
 