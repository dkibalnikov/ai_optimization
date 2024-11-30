library(TSP)
source("https://raw.githubusercontent.com/dkibalnikov/ai_optimization/refs/heads/main/algorithms/functions.R")

# script works only under Linux 
# Install and configure Concorde: works for Linux executable --------------
system("mkdir src") # create directory to place concorde solver
system("wget https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/linux24/concorde.gz") # download concorde
system("gunzip concorde.gz") # extract 
system("mv concorde src") # move to src folder
system("chmod u+x src/concorde") # grant permission 

# Check Concorde is working properly --------------------------------------
concorde_path("src/")
data("USCA50")
solve_TSP(USCA50, method = "concorde")

# Check Concorde for basic task  ------------------------------------------
tst_task <- generate_task()

tst_dist <- dist(tst_task) |> 
  as.matrix()
 
tst_res <- as.TSP(tst_dist) |> 
  solve_TSP(method = "concorde")

prep4plot(tst_task, tst_res) |>
  plot_tour() + 
  labs(title = paste0("Concorde solution: ", calc_dist4mtrx(tst_dist, tst_res)|> round(2)))

# Wrap model and calculate batch ------------------------------------------

get_concorde <- function(task){
  dist_mtrx <- as.matrix(dist(task))
  
  start_time = Sys.time() 
  res <- as.TSP(dist_mtrx) |> 
    solve_TSP(method = "concorde") 
  duration <- Sys.time() - start_time
  
  tibble::tibble(model = "Concorde", duration = duration, 
                 distance = calc_dist4mtrx(dist_mtrx,res), 
                 route = list(as.integer(res)))
}

res_16 <- calc_tours(get_concorde, n_cities = 16)
res_32 <- calc_tours(get_concorde, n_cities = 32) 
res_64 <- calc_tours(get_concorde, n_cities = 64) 

saveRDS(res_16, "test_results/concorde_16nodes.rds")
saveRDS(res_32, "test_results/concorde_32nodes.rds")
saveRDS(res_64, "test_results/concorde_64nodes.rds")
