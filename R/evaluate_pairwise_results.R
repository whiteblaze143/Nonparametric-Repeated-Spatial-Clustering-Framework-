# This script contains functions to plot pairwise results as a graph and convert
# to a matrix

library(igraph)

#' This function plots pairwise results as a graph. The nodes are the initial
#' clusters. An edge represent two clusters with similar underlying distribution.
#' 
#' @param df A Dataframe containing the pairwise test results.
#' @param plot Boolean to plot the graph of pairwise results.
#' @return A matrix of the pairwise results. Obs_mmd_sq value if the two underlying
#' distributions are similar else 0.
#' 
pairwise_results_to_matrix <- function(df, plot = TRUE){
  
  link <- ifelse(df$adj_p >= 0.05, 1, 0)
  df$link <- link
  all_nodes <- unique(c(df$region_1, df$region_2))
  edges <- df[df$link == 1, c("region_1", "region_2", "obs_mmd_sq")]
  
  graph <- graph_from_data_frame(
    d = edges, vertices = data.frame(name = all_nodes), directed = FALSE
  )
  
  result_matrix <- as_adjacency_matrix(graph, attr = "obs_mmd_sq", sparse = FALSE)
  
  num_nodes <- vcount(graph)
  layout <- layout_in_circle(graph)
  
  if (plot) {
    plot(
      graph,
      layout = layout,
      edge.label = round(E(graph)$obs_mmd_sq,2), # Display weights on edges
      vertex.size = 30,
      vertex.label.cex = 1.5,
      edge.curved = 0.1,
      edge.width = 2,
      edge.color = "lightgrey",
      vertex.color = "white",
      vertex.label.color = "black"
    )
  }
  
  return(result_matrix)
}