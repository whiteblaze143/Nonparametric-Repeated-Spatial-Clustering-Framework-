# This script contains the function to plot and store modified silhouette scores for different
# number of clusters and neighborhood sizes.


# Load libraries
library(adespatial)
library(spdep)
library(dplyr)

# Load the modified silhouette score script
source("modified_silhouette.R")

#' This function stores the average modified silhouette score for each
#' combination of neighborhood size and number of clusters.
#' 
#' @param num_clusters A vector containing a range of cluster sizes.
#' @param num_neighbors A vector containing a range of neighborhood sizes.
#' @param linkage The linkage method for the constrained clustering.
#' @param dist_features Dist object of feature distances.
#' @param coords_mat Matrix containing the coordinates.
#' @param plot Boolean to plot the modified silhouette scores
#' @return Data frame containing the average modified silhouette score for each
#' combination.
#' 
evaluate_modified_silhouette <- function(
    num_clusters, num_neighbors,
    linkage = c("ward.D2", "single", "complete", "average"),
    dist_features, coords_mat, plot = TRUE){
  
  if (is.null(linkage) || length(linkage) != 1L || !linkage %in% c("ward.D2", "single", "complete", "average")) {
    stop("'linkage' argument must be one of: 'Ward.D2','single', 'complete', 'avereage'")
  }
  
  # Obtain the linkage method.
  linkage <- match.arg(linkage)
  
  # Pre-allocate a list store the average modified silhouette score for each combination
  results <- vector("list", length(num_clusters)*length(num_neighbors))
  
  counter = 1
  
  for (knn in num_neighbors) {
    
    # Create new a new neighborhood structure.
    nb <- knn2nb(knearneigh(coords_mat, k = knn)) %>%
      nb2listw(style = "W")
    
    # Creating the data frame needed for the constr.hclust function.
    neighbors <- listw2sn(nb)[,1:2]
    
    # Constraint hierarchical clustering
    constr.clust <- constr.hclust(
      d = dist_features, method = linkage,
      links = neighbors, coords = coords_mat
    )
    
    for (i in num_clusters) {
      
      # Cut the tree to obtain different number of clusters
      clusters <- cutree(constr.clust, k = i)
      
      # Apply the modified silhouette score method
      sil_modified <- calculate_modified_silhouette(
        clusters, dist_features, listw2mat(nb)
      )
      
      results[[counter]] <-list(
        num_neighbors = knn,
        num_clusters = i,
        avg_modified_sil = mean(sil_modified[,3])
      )
      counter <- counter + 1
      
      if (plot) {
        plot(sil_modified, main = paste(i, "Clusters with", knn, "neighbors"), border = NA)
      }
    }
  }
  return(as.data.frame(bind_rows(results)))
}

