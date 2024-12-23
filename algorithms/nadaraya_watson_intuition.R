library(torch)
library(ggplot2)
library(patchwork)


# Define some kernels
gaussian <- function(x) torch_exp(-x**2 / 2)
boxcar <- function(x) torch_abs(x) < 1
constant <- function(x) 1 + 0 * x
epanechikov <- function(x) torch_max(1 - torch_abs(x), other = torch_zeros_like(x))
kernel <- list(`gaussian`, `boxcar`, `constant`, `epanechikov`)
kernel_names <- c("gaussian", "boxcar", "constant", "epanechikov")

# Function to plot kernels
plts <- mapply(kernel, kernel_names, SIMPLIFY = F, FUN=function(f, names){
  ggplot() + 
    xlim(-2, 2) +
    geom_function(fun = \(x)f(x) |> torch_tensor(dtype = torch_float()) |> as_array()) +
    labs(title = names)
})
Reduce(`+`, plts) # kernels visualization

# Generate some task for Nadaraya-Watson estimation
f <- \(x) 2 * torch_sin(x) + x
n = 40
x_train = torch_sort(torch_rand(n) * 5)[[1]]   # keys - actual values for independent variable
y_train = f(x_train) + torch_randn(n)          # values - actual values for depended variable
x_val = torch_arange(0, 5, 0.1)                # queries - new x values to be queered 
y_val = f(x_val)                               # truth depended variable to be estimated

# Function to make Nadaraya-Watson estimation for whatever kernel
nadaraya_watson <- function(x_train, y_train, x_val, kernel){
  # Define distance among each key and query pair 
  dists = x_train$reshape(c(-1, 1)) - x_val$reshape(c(1, -1))
  # Each column/row corresponds to each query/key
  k = kernel(dists) |> torch_tensor(dtype = torch_float()) # kernel calculation
  # Normalization over keys for each query
  attention_w = k/k$sum(1)
  y_hat = torch_matmul(y_train, attention_w)
  
  list(y_hat, attention_w)
}

# Check whether function is working 
# nadaraya_watson(x_train, y_train, x_val, kernel = `constant`)

# Create data.frame with true keys, values
truth_df <- data.frame(x_train=as_array(x_train), y_train=as_array(y_train))

# Function to plot fitted values 
fit_plts <- mapply(kernel, kernel_names, SIMPLIFY = F, FUN=function(f, names){
 # browser()
  data.frame(x_val=as_array(x_val), y_val = as_array(y_val), y_train=nadaraya_watson(x_train, y_train, x_val, kernel =f)[[1]] |> as_array()) |>     
    ggplot() + 
    geom_line(aes(x_val, y_train, col = "model"), lty = 5) + 
    geom_line(aes(x_val, y_val, col = "truth")) + 
    geom_point(data = truth_df, aes(x_train, y_train), alpha = .3) +
    scale_color_manual(values = c("model" = "firebrick", "truth" = "black")) +
    labs(title = names)
})
Reduce(`+`, fit_plts) + plot_layout(guides = "collect") # visualize fitted values 
   
# Function to plot attention matrix 
attn_plts <- mapply(kernel, kernel_names, SIMPLIFY = F, FUN=function(f, names){
  #  browser()
  nadaraya_watson(x_train, y_train, x_val, kernel =f)[[2]]  |>     
    as_array() |> 
    apply(MARGIN = 2, FUN = \(x)round(x, 2)) |>
    as.data.frame(row.names = T) |> 
    dplyr::mutate(keys = seq_len(n)) |> 
    tidyr::pivot_longer(!keys, names_to = "queries") |>
    dplyr::mutate(queries=gsub("V", "", queries) |> as.numeric()) |> 
    ggplot() + 
    geom_raster(aes(queries, keys, fill = value)) + 
    scale_fill_viridis_c() + 
    scale_y_reverse() +
    labs(title = names)
})
Reduce(`+`, attn_plts) # visualize attention matrix 



