import torch
import torch.optim as optim
import torch.nn as nn
from .model_core import CORAL_model
from typing import Dict, Tuple, List
import logging
import numpy as np
from torch.distributions import Distribution,constraints,  Normal, LogNormal,Gamma, Poisson, Categorical, kl_divergence as kl
import random
from torch_geometric.utils import to_dense_adj

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def create_model(lowres_dim, hires_dim, lowres_size, hires_size, cell_type_dim, latent_dim=50, hidden_channels=16, v_dim=10, high_res_data_dist='Gamma',low_res_data_dist='NB'):
    model = CORAL_model(lowres_dim, hires_dim, lowres_size, hires_size, cell_type_dim, latent_dim, hidden_channels, v_dim,high_res_data_dist,low_res_data_dist)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model, optimizer





def spatial_attention(edge_index, spatial_coords):
    """
    Compute edge weights based on spatial proximity.
    
    Parameters:
    - edge_index: Tensor of shape (2, num_edges), representing the indices of the connected nodes.
    - spatial_coords: Tensor of shape (num_nodes, 2), representing the x, y coordinates of each node.
    
    Returns:
    - edge_weights: Tensor of shape (num_edges,), representing the computed edge weights.
    """
    num_edges = edge_index.size(1)
    edge_weights = torch.zeros(num_edges).to(spatial_coords.device)
    
    for i in range(num_edges):
        src, dst = edge_index[:, i]
        spatial_dist = torch.norm(spatial_coords[src] - spatial_coords[dst])
        edge_weights[i] = torch.exp(-spatial_dist)  # Exponential decay with distance
    
    return edge_weights


        


def main(adata_downsampled, adata_adt):
    # Step 1: Data Preprocessing
    combined_expr, visium_coords = utils.preprocess_data(adata_smoothed, adata_adt)
    
    # Step 2: Define Model
    input_dim = combined_expr.shape[1]
    model, optimizer = utils.create_model(input_dim)
    
    # Step 3: Prepare Data
    dataloader = utils.prepare_data(combined_expr, codex_coords)
    
    # Step 4: Train Model
    train_model(model, optimizer, dataloader)
    
    # Step 5: Generate and Validate Data
    generated_expr, latent_rep = generate_and_validate(model, dataloader)
    
    return generated_expr, latent_rep, model 

#generated_expr, latent_rep, model = main(adata_smoothed, adata_adt)
