# This script contains the function to create blocks within clusters
library(dplyr)


#' This function creates blocks within each cluster.
#' @param num_blks The number of blocks,
#' @param patient_data A dataframe containing features and initial clusters.
#' The inital set of columns should be the features.
#' @param num_features The number of features
#' @return A dataframe containing the cluster and the block for each observation.

create_blocks  <- function(patient_data, num_features, knn) {
  
  # Initialize an empty list to blocks of each region
  blk_data <- list()
  regions <- unique(patient_data$region)
  
  # Store the initial order of the rows
  patient_data$idx <- seq(1:nrow(patient_data))
  
  # For each region perform kmeans blocking
  for (i in 1:length(regions)) {
    
    region_data <- patient_data[patient_data$region == regions[i],]
    
    # Extracting the features for kmeans
    df <- region_data[,1:num_features]
    
    # Number of blocks based on the cluster size
    num_blks <- floor(nrow(df)/knn)
    
    # If the cluster is smaller than the neighborhood size or the number of
    # distinct rows is less than the number of blocks, use 1 block
    
    num_blks <- ifelse(num_blks == 0 | num_blks >= nrow(unique(df)), 1, num_blks)
    
    if (num_blks == 1) {
      region_data$polygon_id = 1
    }
    else{
      
      kmeans_result <- kmeans(df,  num_blks)
      region_data$polygon_id = kmeans_result$cluster
    }
    
    blk_data[[i]] <- region_data
    
  }
  
  combined_sf_df <- do.call(rbind, blk_data)
  
  # Re-arrange the dataframe to the original order
  combined_sf_df <- combined_sf_df[order(combined_sf_df$idx),]
  
  # Remove the idx column
  combined_sf_df <- combined_sf_df %>% select(-idx)
  
  return(combined_sf_df)
}