
library(tidyverse)

tsts <- list.files("test_results/", full.names = T) |> 
  lapply(readRDS) |> 
  list_rbind()

mutate(tsts, task_size = lapply(route, \(x)ifelse(length(x)%%2==0, length(x), length(x)-1)) |> unlist()) |> 
  ggplot(aes(str_wrap(model, 8), distance, fill = model, col= model)) + 
  geom_violin(alpha = .2) + 
  geom_jitter(alpha = .5)+
  facet_grid(~task_size, scales = "free") + 
  coord_flip() + 
  theme(legend.position = "none")
  
dplyr::mutate(tsts, task_size = lapply(route, \(x)ifelse(length(x)%%2==0, length(x), length(x)-1)) |> unlist()) |> 
  ggplot(aes(str_wrap(model, 8), log(as.numeric(duration)), fill = model, col= model)) + 
  geom_violin(alpha = .2) + 
  geom_jitter(alpha = .5)+
  facet_grid(~task_size, scales = "free") + 
  coord_flip() + 
  theme(legend.position = "none")

example_route <- filter(tsts, model == "MIP" & seed ==2021)[["route"]][[1]] 
swap_route <- example_route
swap_route[2] <- example_route[15]
swap_route[15] <- example_route[2] 

cites <- generate_task()

generate_task() |>
 prep4plot(swap_route) |>
  plot_tour() + 
  geom_line(data = cities[swap_route[2:3], ], aes(x, y), col = "white", lty = 3, linewidth=1) + 
  geom_line(data = cities[swap_route[14:15], ], aes(x, y), col = "white", lty = 3, linewidth=1) +
  geom_line(data = cities[example_route[2:3], ], aes(x, y), col = "firebrick", lty = 3, linewidth=1) + 
  geom_line(data = cities[example_route[14:15], ], aes(x, y), col = "firebrick", lty = 3, linewidth=1)


