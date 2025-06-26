#!/bin/bash

# Process multiple samples with the generate_proust_clusters.py script

# Create output directory
OUTPUT_DIR="spatial_clusters_for_mmd"
mkdir -p "$OUTPUT_DIR"

# Define samples to process
SAMPLES=(
  "Sample_04"
  "Sample_05"
  "Sample_26"
  "Sample_39"
)

# Define cluster counts for each sample (matched to samples array)
CLUSTER_COUNTS=(
  "7" # Sample_04
  "4" # Sample_05
  "7" # Sample_26
  "5" # Sample_39
)

# Define KNN values for each sample
KNN_VALUES=(
  "8" # Sample_04
  "8" # Sample_05
  "6" # Sample_26
  "8" # Sample_39
)

# Process each sample
for i in "${!SAMPLES[@]}"; do
  SAMPLE="${SAMPLES[$i]}"
  CLUSTERS="${CLUSTER_COUNTS[$i]}"
  KNN="${KNN_VALUES[$i]}"
  
  echo "========================================================"
  echo "Processing $SAMPLE with $CLUSTERS clusters and KNN=$KNN"
  echo "========================================================"
  
  # Run the Python script for this sample (using Proust with KMeans fallback)
  python generate_proust_clusters.py \
    --sample "$SAMPLE" \
    --clusters "$CLUSTERS" \
    --knn "$KNN" \
    --output "$OUTPUT_DIR"
  
  # Check if the script was successful
  if [ $? -eq 0 ]; then
    echo "Successfully processed $SAMPLE"
  else
    echo "Failed to process $SAMPLE"
  fi
  
  echo ""
done

echo "All samples processed. Results are available in $OUTPUT_DIR"
echo "You can now use these results with your MMD spatial refinement workflow in R" 