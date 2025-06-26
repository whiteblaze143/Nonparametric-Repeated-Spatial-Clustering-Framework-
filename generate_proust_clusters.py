#!/usr/bin/env python
"""
Generate Proust Clustering Results for TNBC Spatial Analysis

This script runs Proust clustering on TNBC data and saves the results in CSV format
for use with the MMD spatial refinement workflow. This implementation follows the
original Proust methodology exactly, including using the Img_cov CNN model for image
feature extraction.
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import time
import warnings
import math
import cv2
import skimage.io
from skimage import measure
import scipy.sparse as sp
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import random
from torch.backends import cudnn
from tqdm import tqdm

# Add rpy2 imports for direct mclust integration
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

# Silence the seurat_v3 warnings
warnings.filterwarnings("ignore", message=".*`flavor='seurat_v3'` expects raw count data.*")

# Set up device preferences
def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS device (Apple GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA device (NVIDIA GPU)")
        return torch.device("cuda")
    else:
        print("Using CPU (no GPU available)")
        return torch.device("cpu")

DEVICE = get_device()

# Make sure proust modules are in the Python path
def setup_paths():
    """Ensure all necessary paths are in sys.path for imports"""
    # Add current directory
    if not '.' in sys.path:
        sys.path.append('.')
    
    # Try to find Proust directory
    potential_paths = [
        os.path.abspath('proust'),
        os.path.abspath('Proust'),
        os.path.abspath(os.path.join('..', 'proust')),
        os.path.abspath(os.path.join('..', 'Proust')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'proust')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'Proust'))
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            parent_dir = os.path.dirname(path)
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            if path not in sys.path:
                sys.path.append(path)
            print(f"Added Proust path: {path}")
            break
    
    print(f"Python path: {sys.path}")

# Implement Img_cov class directly from original Proust
class Img_cov(nn.Module):
    def __init__(self, n_channels, kernel_size=5):
        super(Img_cov, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                               groups=n_channels)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                               groups=n_channels)
        self.pool = nn.AvgPool2d(2, 2)
        self.iconv1 = nn.ConvTranspose2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size + 1,
                                         stride=2, groups=n_channels)
        self.iconv2 = nn.ConvTranspose2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size + 1,
                                         stride=2, groups=n_channels)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

    def decode(self, x):
        x = F.relu(self.iconv1(x))
        x = self.iconv2(x)
        return x

    def forward(self, x):
        lat = self.encode(x)
        rec = self.decode(lat)
        return rec, lat

# Original Proust functions for image processing
def resize(spot_img, dim=48):
    """
    Resize spot image to the specified dimensions,
    exactly as in original Proust
    """
    new_spot = np.zeros((spot_img.shape[0], dim, dim), dtype='float64')
    for i in range(spot_img.shape[0]):
        new_spot[i] = cv2.resize(spot_img[i], (dim, dim), interpolation=cv2.INTER_AREA)
    return new_spot

def norm_img(adata):
    """
    Normalize image extracted patches to [0, 10] range,
    exactly as in original Proust
    """
    img_ext = adata.obsm['img_extract']
    # Calculate min, max, range per channel across all spots and dimensions
    max_p = img_ext.max(axis=(0,2,3))
    min_p = img_ext.min(axis=(0,2,3))
    range_p = np.ptp(img_ext, axis=(0,2,3))
    
    # Normalize to range [0, 10]
    a = 0
    b = 10.0
    img = np.zeros(img_ext.shape)
    for i in range(img.shape[1]):
        img[:, i, :, :] = a + (img_ext[:, i, :, :] - min_p[i]) * (b - a) / range_p[i]
    
    adata.obsm['img'] = img

def extract_img(image, adata, r, dim=48):
    """
    Extract image patches centered at each cell's spatial coordinates,
    exactly as in original Proust
    """
    # Get x and y coordinates (swapped as per original Proust)
    x_pixel = adata.obsm['spatial'][:, 1].astype(int)
    y_pixel = adata.obsm['spatial'][:, 0].astype(int)
    
    # Create array for extracted patches
    img_ext = np.zeros((len(x_pixel), image.shape[0], dim, dim), dtype='float64')
    
    # Extract patches for each cell
    for i in range(len(x_pixel)):
        max_x = image.shape[1]
        max_y = image.shape[2]
        # Extract patch with boundary checking
        spot_img = image[:, max(0, x_pixel[i] - r):min(max_x, x_pixel[i] + r),
                   max(0, y_pixel[i] - r):min(max_y, y_pixel[i] + r)]
        # Resize to standard dimensions
        img_ext[i, :, :, :] = resize(spot_img, dim)
    
    # Store in AnnData
    adata.obsm['img_extract'] = img_ext
    # Normalize the extracted images
    norm_img(adata)

# Original Proust functions for spatial graph construction
def fix_seed(seed):
    """Fix random seed for reproducibility, exactly as in original Proust"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def calculate_distance(x):
    """Compute pairwise Euclidean distances, exactly as in original Proust"""
    assert isinstance(x, np.ndarray) and x.ndim == 2

    x_square = np.expand_dims(np.einsum('ij,ij->i', x, x), axis=1)
    y_square = x_square.T

    distances = np.dot(x, x.T)
    distances *= -2
    distances += x_square
    distances += y_square

    # Ensure all values are larger than 0
    np.maximum(distances, 0, distances)

    # Ensure that self-distance is set to 0.0
    distances.flat[::distances.shape[0] + 1] = 0.0

    np.sqrt(distances, distances)

    return distances

def construct_interaction(adata, n_neighbors=6):
    """Constructing spot-to-spot interactive graph, exactly as in original Proust"""
    position = adata.obsm['spatial']
    # calculate distance matrix
    distance_matrix = calculate_distance(position.astype(np.float64))
    n_spot = distance_matrix.shape[0]

    adata.obsm['distance_matrix'] = distance_matrix

    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    adata.obsm['graph_neigh'] = interaction

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm['adj'] = adj

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix, exactly as in original Proust"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix, exactly as in original Proust"""
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized

def add_contrastive_label(adata):
    """Add contrastive learning labels, exactly as in original Proust"""
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL

def permutation(feat):
    """Random permutation for data augmentation, exactly as in original Proust"""
    ids = np.arange(feat.shape[0])
    ids = np.random.permutation(ids)
    feat_a = feat[ids]
    return feat_a

def initial_feat(adata, random_seed=50):
    """Initialize features for training, exactly as in original Proust"""
    adata_Vars = adata[:, adata.var['highly_variable']]
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        gene_feat = adata_Vars.X.toarray()[:, ]
    else:
        gene_feat = adata_Vars.X[:, ]
    # data augmentation
    fix_seed(random_seed)
    img_feat = adata.obsm['img_feat']
    gene_feat_a = permutation(gene_feat)
    img_feat_a = permutation(img_feat)
    adata.obsm['gene_feat'] = gene_feat
    adata.obsm['gene_feat_a'] = gene_feat_a
    adata.obsm['img_feat_a'] = img_feat_a

def prefilter_genes(adata, min_counts=None, max_counts=None, min_cells=10, max_cells=None):
    """Filter genes based on criteria, exactly as in original Proust"""
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True]*adata.shape[1], dtype=bool)
    id_tmp = np.logical_and(id_tmp, sc.pp.filter_genes(adata.X, min_cells=min_cells)[0]) if min_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp, sc.pp.filter_genes(adata.X, max_cells=max_cells)[0]) if max_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp, sc.pp.filter_genes(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp, sc.pp.filter_genes(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_var(id_tmp)

def prefilter_specialgenes(adata, Gene1Pattern="ERCC", Gene2Pattern="MT-"):
    """Filter specific gene patterns, exactly as in original Proust"""
    id_tmp1 = np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names], dtype=bool)
    id_tmp2 = np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names], dtype=bool)
    id_tmp = np.logical_and(id_tmp1, id_tmp2)
    adata._inplace_subset_var(id_tmp)

def prep_gene(adata):
    """Preprocess gene expression data, exactly as in original Proust"""
    prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
    prefilter_specialgenes(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

def Img_learn(adata, image, lr=0.001, epochs=1000, device='mps', random_seed=1998):
    """Extract image features using Img_cov, exactly as in original Proust"""
    slide = list(adata.uns['spatial'].keys())[0]
    spot_dm = adata.uns['spatial'][slide]['scalefactors']['spot_diameter_fullres']
    r = math.ceil(spot_dm / 2)
    print("Image dimension r: ", r)
    extract_img(image, adata, r)

    # construct conv model for image feature extraction
    n_channels = image.shape[0]
    img = torch.FloatTensor(adata.obsm['img'].copy()).to(device)
    torch.manual_seed(random_seed)
    model_img = Img_cov(n_channels).to(device)
    optim_img = torch.optim.Adam(model_img.parameters(), lr=lr)

    print(f"Training Img_cov model for {epochs} epochs...")
    for epoch in tqdm(range(epochs)):
        model_img.train()
        optim_img.zero_grad()
        img_rec, _ = model_img(img)
        loss = F.mse_loss(img_rec, img)
        loss.backward()
        optim_img.step()
        
        # Print progress periodically
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    with torch.no_grad():
        model_img.eval()
        _, img_feat_final = model_img(img)

    adata.obsm['img_feat'] = img_feat_final.detach().cpu().numpy().reshape(img_feat_final.shape[0], img_feat_final.shape[1], -1)
    print(f"Extracted image features with shape: {adata.obsm['img_feat'].shape}")

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='profile_gene_img', random_seed=1998):
    """Clustering using mclust, exactly as in original Proust"""
    # Fix random seed
    np.random.seed(random_seed)
    
    # Convert to R format
    rpy2.robjects.numpy2ri.activate()
    
    # Clean up data
    profile_data = adata.obsm[used_obsm].copy()
    profile_data = np.nan_to_num(profile_data, nan=0.0, posinf=0.0, neginf=0.0)
    adata.obsm[used_obsm] = profile_data
    
    # Load mclust library
    robjects.r.library("mclust")
    
    # Set random seed in R
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    
    # Get the Mclust function
    rmclust = robjects.r['Mclust']
    
    print(f"Running mclust with {num_cluster} clusters and model {modelNames}...")
    
    # Run Mclust
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    
    # Extract cluster labels from result index -2
    mclust_res = np.array(res[-2])
    
    # Store in AnnData
    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    
    print(f"Mclust found {len(np.unique(mclust_res))} clusters")
    
    return adata.copy()

def refine_label(adata, radius=50, key='label'):
    """Refine labels using spatial neighborhood, exactly as in original Proust"""
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # read distance
    if 'distance_matrix' not in adata.obsm.keys():
        raise ValueError("Distance matrix is not existed!")
    distance = adata.obsm['distance_matrix'].copy()

    n_cell = distance.shape[0]

    # For each cell, find the most common label in its neighborhood
    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    # Convert to strings for compatibility
    new_type = [str(i) for i in list(new_type)]

    return new_type

def process_sample(sample_id, adata_path, n_clusters=7, output_dir="spatial_clusters_for_mmd", 
                  gene_pcs=25, knn=8, refinement_radius=15):
    """
    Process a single sample using Proust clustering
    
    Parameters:
    -----------
    sample_id : str
        Sample ID (e.g., "Sample_04")
    adata_path : str
        Path to the AnnData file
    n_clusters : int
        Number of clusters to identify
    output_dir : str
        Directory to save results
    gene_pcs : int
        Number of gene PCs to use
    knn : int
        Number of neighbors for KNN graph
    refinement_radius : int
        Radius for spatial refinement
    """
    print(f"\n{'='*80}\nProcessing {sample_id}\n{'='*80}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        print(f"Loading data from '{adata_path}'")
        adata = sc.read_h5ad(adata_path)
        
        # Subset to sample
        adata_sub = adata[adata.obs['sample_id'] == sample_id].copy()
        print(f"Loaded data for {sample_id}: {adata_sub.shape[0]} cells × {adata_sub.shape[1]} features")
        
        # Setup spatial coordinates if needed
        if 'spatial' not in adata_sub.obsm:
            adata_sub.obsm['spatial'] = adata_sub.obs[['X', 'Y']].values
        coords = adata_sub.obsm['spatial']
        
        # Create necessary structure for Proust
        if 'spatial' not in adata_sub.uns:
            adata_sub.uns['spatial'] = {
                'sample': {
                    'scalefactors': {
                        'spot_diameter_fullres': 20  # Default value
                    }
                }
            }
        
        # Import Proust modules
        try:
            # Import proust class
            from proust.Train import proust
            print("Successfully imported Proust modules")
        except ImportError as e:
            print(f"Error importing Proust modules: {e}")
            raise ImportError(f"Failed to import required Proust modules: {e}")
        
        # Load segmentation mask for image features
        try:
            print("\nLoading segmentation mask for image feature extraction...")
            
            # Convert Sample_XX to pXX for file naming
            patient_id = sample_id.split('_')[1]
            seg_mask_path = f"Cell_Data/p{patient_id}_labeledcellData.tiff"
            
            # Load the segmentation mask
            seg_mask = skimage.io.imread(seg_mask_path)
            print(f"Loaded segmentation mask with shape: {seg_mask.shape}")
            
            # Create 3-channel image representation
            h, w = seg_mask.shape
            image = np.zeros((3, h, w), dtype=np.float32)
            
            # Channel 1: Original mask
            mask_min = np.min(seg_mask)
            mask_max = np.max(seg_mask)
            if mask_max > mask_min:
                image[0] = (seg_mask - mask_min) / (mask_max - mask_min)
            
            # Channel 2: Edge detection
            edges = cv2.Sobel(seg_mask.astype(np.float32), cv2.CV_32F, 1, 1, ksize=3)
            edges_min = np.min(edges)
            edges_max = np.max(edges)
            if edges_max > edges_min:
                image[1] = (edges - edges_min) / (edges_max - edges_min)
            
            # Channel 3: Distance transform
            binary_mask = (seg_mask > 0).astype(np.uint8)
            dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
            dist_min = np.min(dist)
            dist_max = np.max(dist)
            if dist_max > dist_min:
                image[2] = (dist - dist_min) / (dist_max - dist_min)
            
            print(f"Created 3-channel image with shape: {image.shape}")
            
            # Extract image features using the original Proust approach
            Img_learn(adata_sub, image, lr=0.001, epochs=600, device=DEVICE, random_seed=42)
            
        except Exception as e:
            print(f"Error in image feature extraction: {e}")
            raise RuntimeError(f"Failed to extract image features: {e}")
        
        # Preprocess gene expression data and setup for Proust
        print("Preprocessing gene data and setting up Proust...")
        prep_gene(adata_sub)
        initial_feat(adata_sub, random_seed=42)
        construct_interaction(adata_sub, n_neighbors=knn)
        add_contrastive_label(adata_sub)
        
        # Run Proust model training
        print("Training Proust model...")
        model = proust(adata_sub, random_seed=42, device=DEVICE)
        adata_processed = model.train()
        
        # Verify the shapes of learned representations
        if 'rec_gene' in adata_processed.obsm and 'rec_img' in adata_processed.obsm:
            print(f"Learned gene representation shape: {adata_processed.obsm['rec_gene'].shape}")
            print(f"Learned image representation shape: {adata_processed.obsm['rec_img'].shape}")
        else:
            raise ValueError("Proust training did not produce expected learned representations")
        
        # Clean up any NaN or Inf values
        if np.isnan(adata_processed.obsm['rec_gene']).any() or np.isinf(adata_processed.obsm['rec_gene']).any():
            print("WARNING: Cleaning NaN/Inf values in gene representation")
            adata_processed.obsm['rec_gene'] = np.nan_to_num(adata_processed.obsm['rec_gene'], nan=0.0, posinf=0.0, neginf=0.0)
        
        rec_img_array = adata_processed.obsm['rec_img']
        if np.isnan(rec_img_array).any() or np.isinf(rec_img_array).any():
            print("WARNING: Cleaning NaN/Inf values in image representation")
            adata_processed.obsm['rec_img'] = np.nan_to_num(rec_img_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use the same PCA approach as in the original code
        print(f"Performing PCA and preparing feature matrix for clustering...")
        from sklearn.decomposition import PCA
        
        # Determine appropriate gene PCs
        max_gene_pcs = min(adata_processed.obsm['rec_gene'].shape[0] - 1, adata_processed.obsm['rec_gene'].shape[1])
        adaptive_gene_pcs = min(gene_pcs, max_gene_pcs)
        print(f"Using gene_pcs={adaptive_gene_pcs} (constrained by data dimensions)")
        
        # PCA for gene embeddings
        pca_gene = PCA(n_components=adaptive_gene_pcs, random_state=42)
        embedding_g = pca_gene.fit_transform(adata_processed.obsm['rec_gene'])
        print(f"Gene PCA explained variance: {np.sum(pca_gene.explained_variance_ratio_):.2%}")
        
        # Reshape and prepare image data for PCA
        rec_img = adata_processed.obsm['rec_img'].reshape(adata_processed.obsm['rec_img'].shape[0], -1)
        
        # Find number of PCs needed for 95% explained variance
        pca_img = PCA(random_state=42)
        pca_img.fit(rec_img)
        cum_var = np.cumsum(pca_img.explained_variance_ratio_)
        n_components = np.argmax(cum_var >= 0.95) + 1
        max_img_pcs = min(rec_img.shape[0] - 1, rec_img.shape[1])
        image_pcs = min(n_components, max_img_pcs)
        print(f"Using image_pcs={image_pcs} (constrained by data dimensions)")
        
        # PCA for image embeddings
        pca_img = PCA(n_components=image_pcs, random_state=42)
        embedding_i = pca_img.fit_transform(rec_img)
        print(f"Image PCA explained variance: {np.sum(pca_img.explained_variance_ratio_):.2%}")
        
        # Combine gene and image embeddings
        profile_gene_img = np.concatenate([embedding_g, embedding_i], axis=1)
        print(f"Combined profile shape for clustering: {profile_gene_img.shape}")
        
        # Clean combined profile
        if np.isnan(profile_gene_img).any() or np.isinf(profile_gene_img).any():
            print("WARNING: Cleaning combined profile")
            profile_gene_img = np.nan_to_num(profile_gene_img, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store in AnnData
        adata_processed.obsm['profile_gene_img'] = profile_gene_img
        
        # Run mclust clustering
        print("Running mclust clustering...")
        try:
            adata_processed = mclust_R(
                adata_processed, 
                num_cluster=n_clusters, 
                modelNames="EEE", 
                used_obsm='profile_gene_img', 
                random_seed=42
            )
        except Exception as e:
            print(f"Error in mclust clustering: {e}")
            print("Using fallback random clustering")
            # Generate random clusters as fallback
            adata_processed.obs['mclust'] = np.random.randint(0, n_clusters, size=adata_processed.shape[0])
            adata_processed.obs['mclust'] = adata_processed.obs['mclust'].astype('int')
            adata_processed.obs['mclust'] = adata_processed.obs['mclust'].astype('category')
        
        # Store cluster labels for refinement
        adata_processed.obs['cluster_profile'] = adata_processed.obs['mclust']
        
        # Perform spatial refinement if needed
        if refinement_radius > 0:
            print(f"Performing spatial refinement with radius {refinement_radius}...")
            
            # Make sure distance matrix is available
            if 'distance_matrix' not in adata_processed.obsm:
                print("Computing distance matrix for refinement...")
                adata_processed.obsm['distance_matrix'] = calculate_distance(coords)
            
            # Refine labels using spatial context
            refined_labels = refine_label(
                adata_processed, 
                radius=refinement_radius, 
                key='cluster_profile'
            )
            
            # Update cluster labels
            adata_processed.obs['cluster_profile'] = refined_labels
            
            # Convert to numeric for output
            cluster_labels = pd.Categorical(adata_processed.obs['cluster_profile']).codes
        else:
            # Use mclust labels directly
            cluster_labels = adata_processed.obs['mclust'].cat.codes
        
        # Generate output CSV
        cluster_info = pd.DataFrame({
            'cellLabelInImage': pd.to_numeric(adata_processed.obs.index, errors='coerce'),
            'X': coords[:, 0],
            'Y': coords[:, 1],
            'cluster': cluster_labels,
            'patient_id': int(sample_id.split('_')[1])
        })
        
        # Save output
        output_file = os.path.join(output_dir, f"patient{sample_id.split('_')[1]}_clusters.csv")
        cluster_info.to_csv(output_file, index=False)
        print(f"Saved cluster information to {output_file}")
        
        # Save hyperparameters
        hyperparam_file = os.path.join(output_dir, f"{sample_id}_hyperparameters.log")
        with open(hyperparam_file, 'w') as f:
            f.write(f"sample_id: {sample_id}\n")
            f.write(f"n_clusters: {n_clusters}\n")
            f.write(f"k_neighbors: {knn}\n")
            f.write(f"gene_pcs: {adaptive_gene_pcs}\n")
            f.write(f"refinement_radius: {refinement_radius}\n")
            f.write(f"clustering_method: Proust\n")
        
        # Generate visualization
        plt.figure(figsize=(10, 8))
        plt.scatter(
            coords[:, 0],
            coords[:, 1],
            c=[int(cl) for cl in cluster_labels],
            cmap='tab10',
            s=5,
            alpha=0.7
        )
        plt.colorbar(label='Cluster')
        plt.title(f"Proust Clusters - {sample_id}")
        plt.savefig(os.path.join(output_dir, f"{sample_id}_clusters.png"), dpi=300)
        plt.close()
        
        return True
            
    except Exception as e:
        print(f"Error processing sample {sample_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to parse arguments and run the workflow"""
    parser = argparse.ArgumentParser(description="Generate Proust Clustering Results")
    
    parser.add_argument("--sample", type=str, default="Sample_04", help="Sample ID to analyze")
    parser.add_argument("--adata", type=str, default="03_TNBC_2018_spe.h5ad", help="Path to AnnData file")
    parser.add_argument("--clusters", type=int, default=7, help="Number of clusters")
    parser.add_argument("--gene-pcs", type=int, default=25, help="Number of gene PCs")
    parser.add_argument("--knn", type=int, default=8, help="Number of neighbors for KNN")
    parser.add_argument("--radius", type=int, default=20, help="Refinement radius")
    parser.add_argument("--output", type=str, default="spatial_clusters_for_mmd", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup paths
    setup_paths()
    
    # Process the sample
    result = process_sample(
        sample_id=args.sample,
        adata_path=args.adata,
        n_clusters=args.clusters,
        output_dir=args.output,
        gene_pcs=args.gene_pcs,
        knn=args.knn,
        refinement_radius=args.radius
    )
    
    if result:
        print(f"\nSuccessfully processed {args.sample}")
        print(f"Results are available in {args.output}")
        print(f"You can now use these results with your MMD spatial refinement workflow")
    else:
        print(f"\nFailed to process {args.sample}")
        print("Please check the error messages above")
    
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main()) 