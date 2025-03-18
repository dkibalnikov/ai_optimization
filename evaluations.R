library(tidyverse)

tsts <- list.files("test_results/", full.names = T) |> 
  lapply(readRDS) |> 
  list_rbind()

dplyr::mutate(tsts, task_size = lapply(route, \(x)ifelse(length(x)%%2==0, length(x), length(x)-1)) |> unlist()) |> 
  ggplot(aes(str_wrap(model, 8), distance, fill = model, col= model)) + 
  geom_violin(alpha = .2) + 
  geom_jitter(alpha = .5) +
  facet_grid(~task_size, scales = "free") + 
  coord_flip() + 
  theme(legend.position = "none") + 
  labs(title = "Accuracy comparison among various TSP algorithms", , subtitle = "for 16, 32, 64 size task",
       x = "Algorithms", y="Solution distance")

ggsave("diagrams/accuracy.png", device = "png")
  
dplyr::mutate(tsts, task_size = lapply(route, \(x)ifelse(length(x)%%2==0, length(x), length(x)-1)) |> unlist()) |> 
  ggplot(aes(str_wrap(model, 8), log(as.numeric(duration)), fill = model, col= model)) + 
  geom_violin(alpha = .2) + 
  geom_jitter(alpha = .5)+
  facet_grid(~task_size, scales = "free") + 
  coord_flip() + 
  theme(legend.position = "none")  + 
  labs(title = "Duration comparison among various TSP algorithms", subtitle = "for 16, 32, 64 size task",
       x = "Algorithms", y="log(Duration)")

ggsave("diagrams/duration.png", device = "png")

