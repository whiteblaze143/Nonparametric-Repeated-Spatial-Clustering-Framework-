# This script contains all the functions to calculate the modified silhouette score

#' This function calculates the a(i) term.
#' 
#' @param i A single observation.
#' @param clusters A vector containing the clusters of the data points.
#' @param dist_matrix The distance matrix of the features.
#' @return(a_i) A numerical value (the average distance between point i and
#' other observations in the same cluster).
#' 
calculate_ai <- function(i, clusters, dist_matrix) {
  
  # Get the current cluster for observation i
  current_cluster <- clusters[i]
  
  # Get the observation numbers in the same cluster as i
  current_cluster_obs <- which(clusters == current_cluster)
  
  # Exclude observation i from the list of current cluster observations
  current_cluster_obs <- current_cluster_obs[current_cluster_obs != i]
  
  if (length(current_cluster_obs) == 0) {
    return(0)
  }
  
  # Calculate the average distance between i and other observations in the same cluster
  a_i <- mean(dist_matrix[i, current_cluster_obs])
  
  return(a_i)
}


#' This function finds clusters that have links between them.
#' 
#' @param clusters A vector containing the clusters of the data points.
#' @param neighborhood_matrix The binary neighborhood matrix.
#' @return(neighboring_clusters) A list containing neighboring clusters for
#'  each cluster.
#' 
calculate_neighboring_clusters <- function(clusters, neighborhood_matrix) {
  
  # Get unique cluster labels
  unique_clusters <- unique(clusters)
  
  # Initialize an empty list to store neighboring clusters for each cluster
  neighboring_clusters <- list()
  
  # For each cluster, find neighboring clusters based on the neighborhood matrix
  for (cluster in unique_clusters) {
    cluster_obs <- which(clusters == cluster)
    neighboring_clusters[[as.character(cluster)]] <- unique(
      clusters[
        (colSums(neighborhood_matrix[cluster_obs, , drop = FALSE]) > 0) & 
        (clusters != cluster)
      ] 
    )
  }
  
  return(neighboring_clusters)
}

#' This function calculates the b(i) and the nearest neighboring cluster
#' 
#' @param i A single observation.
#' @param clusters A vector containing the clusters of the data points.
#' @param dist_matrix The distance matrix of the features.
#' @param neighboring_clusters A list containing neighboring clusters for each 
#' clusters.
#' @return A list containing the nearest neighboring cluster and b(i)
#' 
calculate_bi_nb <- function(i, clusters, dist_matrix, neighboring_clusters) {
  
  # Get the current cluster for observation i
  current_cluster <- clusters[i]
  
  # Get precomputed neighboring clusters for the current cluster
  neighbors <- neighboring_clusters[[as.character(current_cluster)]]
  
  # If no neighboring clusters, return Inf for b_i
  if (length(neighbors) == 0) {
    return(list(neighbor = NA, b_i = Inf))
  }
  
  # Calculate the average distance to observations in neighboring clusters
  b_i_values <- sapply(neighbors, function(cluster) {
    cluster_points <- which(clusters == cluster)
    mean(dist_matrix[i, cluster_points])
  })
  
  # Get the smallest average distance (min b_i) and corresponding neighboring cluster
  min_b_i <- min(b_i_values)
  neighbor <- neighbors[which.min(b_i_values)]
  
  return(list(neighbor = neighbor, b_i = min_b_i))
}

#' This function calculates the modified silhouette score for a given set of clusters
#' 
#' @param clusters A vector containing the clusters of the data points.
#' @param dist_matrix The distance matrix of the features.
#' @param neighborhood_matrix The binary neighborhood matrix.
#' @return(wds) A silhouette class object.

calculate_modified_silhouette <- function(clusters, dist_matrix, neighborhood_matrix) {
  
  # Convert dist object to matrix
  dist_matrix <- as.matrix(dist_matrix)
  
  # Precompute neighboring clusters for each cluster
  neighboring_clusters <- calculate_neighboring_clusters(clusters, neighborhood_matrix)
  
  # Sequential computation of silhouette widths and neighbors
  results <- lapply(seq_len(length(clusters)), function(i) {
    a_i <- calculate_ai(i, clusters, dist_matrix)
    b_i_nb <- calculate_bi_nb(i, clusters, dist_matrix, neighboring_clusters)
    sil_width <- (b_i_nb$b_i - a_i) / max(a_i, b_i_nb$b_i)
    
    if (is.nan(sil_width)){
      sil_width = 1
    }
    
    # Return silhouette width and neighbor
    list(sil_width = sil_width, neighbor = b_i_nb$neighbor)
  })
  
  # Extract silhouette widths and neighbors from results
  sil_widths <- sapply(results, function(res) res$sil_width)
  neighbors <- sapply(results, function(res) res$neighbor)
  
  # Combine results into a matrix for output
  wds <- cbind(
    cluster = clusters,
    neighbor = neighbors,
    sil_width = sil_widths
  )
  
  attr(wds, "Ordered") <- FALSE
  attr(wds, "call") <- match.call()
  class(wds) <- "silhouette"
  
  return(wds)
}