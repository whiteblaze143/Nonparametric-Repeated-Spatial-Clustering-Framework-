# This script contains functions needed to calculate mmd^2, perform block permutation and
# conduct multiple comparison
library(purrr)

#' This function calculated MMD^2 for clusters.
#'
#' @param sample1_idx Indices of cluster 1 observations aligning with the
#' indices of dist_matrix. This is a vector.
#' @param sample2_idx Indices of cluster 2 observations aligning with the
#' indices of dist_matrix. This is a vector.
#' @param dist_matrix Distance matrix of the features.
#' @param kernel Kernel function for calculating MMD.
#' @param kernel_param Parameter value required for the kernel function.
#' @return(mmd_sq) Numerical value.
#'
compute_mmd_sq <- function(
    sample1_idx, sample2_idx, dist_matrix, 
    kernel = c("Gaussian", "IMQ"), kernel_param = 1 
) {
  
  if (!kernel %in% c("Gaussian", "IMQ")){
    stop("'kernel' must be one of: 'Gaussian','IMQ'")
  }
  
  N <- length(sample1_idx)
  M <- length(sample2_idx)
  
  # Obtain the kernel function.
  kernel = match.arg(kernel)
  
  if (kernel == "Gaussian") {
    # First term: 1/N^2 sum K(xn, xn')
    term1 <- sum(exp(-dist_matrix[sample1_idx, sample1_idx]^2/kernel_param)) / (N * N)
    
    # Second term: -2/NM sum K(xn, xm')
    term2 <- -2 * sum(exp(-dist_matrix[sample1_idx, sample2_idx]^2/kernel_param)) / (N * M)
    
    # Third term: 1/M^2 sum K(x'm, x'm')
    term3 <- sum(exp(-dist_matrix[sample2_idx, sample2_idx]^2/kernel_param)) / (M * M)
    
    # MMD^2 value
    mmd_sq <- term1 + term2 + term3
  }
  else{
    # First term: 1/N^2 sum K(xn, xn')
    term1 <- sum(1/(sqrt(kernel_param+(dist_matrix[sample1_idx, sample1_idx])^2))) / (N * N)
  
    # Second term: -2/NM sum K(xn, xm')
    term2 <- -2 * sum(1/(sqrt(kernel_param+(dist_matrix[sample1_idx, sample2_idx])^2))) / (N * M)
    
    # Third term: 1/M^2 sum K(x'm, x'm')
    term3 <- sum(1/(sqrt(kernel_param+(dist_matrix[sample2_idx, sample2_idx])^2))) / (M * M)
    
    # MMD^2 value
    mmd_sq <- term1 + term2 + term3
  }
  
  return(mmd_sq)
}

#' This function permute labels between two clusters and calculates the MMD^2 for
#' the permuted data.
#' 
#' @param sample1_idx Indices of cluster 1 observations aligning with the
#' indices of dist_matrix. This is a vector.
#' @param sample2_idx Indices of cluster 2 observations aligning with the
#' indices of dist_matrix. This is a vector.
#' @param dist_matrix Distance matrix of the features.
#' @param patient_data Data set containing features, original cluster labels (regions) 
#' and polygon_ids.
#' @return(perm_mmd_sq) Numerical value.
#' 
compute_perm_mmd_sq <- function(
    sample1_idx, sample2_idx,
    dist_matrix, patient_data,
    kernel = "IMQ", kernel_param = 1
) {
  
  # Saving the original order.
  patient_data$point <- 1:nrow(patient_data)
  
  #Store the indices to a list and sort it based on size.
  indices <- list(sample1_idx, sample2_idx)
  indices <- indices[order(sapply(indices, length))]
  
  # Extract data points from the dataset.(data from the smaller cluster is at the top)
  subset_data <- patient_data[c(indices[[1]], indices[[2]]),]
  
  # Creating a block_id column to have a unique id for each polygon.
  subset_data <- subset_data |> 
    group_by(region, polygon_id) |>
    mutate(block_id = cur_group_id()) |>
    ungroup()

  # Size of the smaller cluster
  n_size <- sum(subset_data$region == unique(subset_data$region)[1])
  
  # To store the sampled blocks and the count of observations sampled.
  blk_ids <- c()
  n_permuted <- 0
  
  while (n_permuted < n_size) {
    new_block <- sample(
      setdiff(
        unique(subset_data$block_id), 
        blk_ids), 
      1
    )
    blk_ids <- c(blk_ids, new_block)
  
    # The number of data points in the sampled blocks.
    n_permuted <- length(subset_data$polygon_id[subset_data$block_id %in% blk_ids])
  }
  
  # Store the permuted cluster label in permuted_region column.
  subset_data <- subset_data |>
    mutate(
      permuted_region = ifelse(
        block_id %in% blk_ids, 
        unique(subset_data$region)[1], 
        unique(subset_data$region)[2]
      )
    )
    
    # Computing the MMD^2 for the permuted data
    perm_mmd_sq <- compute_mmd_sq(
      subset_data$point[subset_data$permuted_region == unique(subset_data$permuted_region)[1]],
      subset_data$point[subset_data$permuted_region == unique(subset_data$permuted_region)[2]],
      dist_matrix, kernel, kernel_param
    )
    
    return(perm_mmd_sq)
  
}

#' This function conducts 2 sample MMD^2 nonparametric test.
#' 
#' @param sample1_idx Indices of cluster 1 observations aligning with the
#' indices of dist_matrix. This is a vector.
#' @param sample2_idx Indices of cluster 2 observations aligning with the
#' indices of dist_matrix. This is a vector.
#' @param dist_matrix Distance matrix of the features.
#' @param patient_data Data set containing features, original cluster labels (regions) 
#' and polygon_ids.
#' @param kernel Kernel function for calculating MMD.
#' @param kernel_param Parameter value required for the kernel function.
#' @param nperm Number of permutations.
#' @return(two_sample_list) A list containing observed MMD^2, p-value and null distribution.
#'
two_sample_mmd <- function(
    sample1_idx, sample2_idx,
    dist_matrix, patient_data,
    kernel = "IMQ", kernel_param = 1, nperm = 200){
  
  # Observed MMD values
  obs_mmd_sq <- compute_mmd_sq(sample1_idx, sample2_idx, dist_matrix, kernel, kernel_param)
  
  # Distribution of MMDs under the null hypothesis
  null_mmd_sq <- lapply(
    1:nperm, 
    function(i){
      compute_perm_mmd_sq(sample1_idx, sample2_idx, dist_matrix, patient_data, kernel, kernel_param)
    }
  ) |> unlist()
  
  p_value <- mean(null_mmd_sq >= obs_mmd_sq)
  
  two_sample_result <- list(obs_mmd_sq = obs_mmd_sq, p_value = p_value, null_dist = null_mmd_sq)
  
  return(two_sample_result)
}

#' This functions conducts multiple hypothesis tesing
#' 
#' @param patien_data Data set containing features, original cluster labels (regions) 
#' and polygon_ids.
#' @param dist_matrix Distance matrix of the features.
#' @param kernel Kernel function for calculating MMD.
#' @param kernel_param Parameter value required for the kernel function.
#' @param nperm Number of permutations.
#' @param adj_p Adjusted p-value method
#' @return(results_summary) A data frame consisting of all pairwise test results.
#'
multiple_comparison <- function(
    patient_data, dist_matrix, kernel = "IMQ",
    kernel_param = 1, nperm = 200, adj_p = c("BH", "bonferroni", "holm")
){
  
  
  adj_p <- match.arg(adj_p)
  
  # list to store pairwise results
  pairwise_results <- list()
  
  # Unique cluster pairs
  cluster_pairs <- combn(unique(patient_data$region), 2)
  
  # Conduct 2 sample test for each pair
  for (i in 1:ncol(cluster_pairs)) {
    
    cluster_1 <- which(patient_data$region == cluster_pairs[1, i])
    cluster_2 <- which(patient_data$region == cluster_pairs[2, i])
    
    pairwise_results[[i]] <- c(
      list(clusters = c(cluster_pairs[1, i], cluster_pairs[2, i])),
      two_sample_mmd(
        cluster_1, cluster_2, dist_matrix, patient_data,
        kernel, kernel_param, nperm
      )
    )
  
  }
  
  # Create a dataframe of all multiple comparison results
  results_summary <- map_dfr(pairwise_results, ~ data.frame(
    region_1 = .x$clusters[1],
    region_2 = .x$clusters[2],
    obs_mmd_sq = .x$obs_mmd_sq,
    p_value = .x$p_value,
    null_dist = I(list(.x$null_dist))
  ))
  
  # Add an adjusted p-value column for needed for multiple hypothesis testing
  results_summary$adj_p <- p.adjust(results_summary$p_value, method = adj_p)
  
  
  return(results_summary)
}