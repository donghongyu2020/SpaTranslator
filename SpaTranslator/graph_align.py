import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


from .utils import generate_mnn_dict
from .gat import GraphAttentionLayer

cudnn.deterministic = True
cudnn.benchmark = True

class SpaTranslator_Aligner(torch.nn.Module):
    """
    Spatial Transformer Aligner - A graph neural network model for aligning spatial data
    Implements an encoder-decoder architecture with graph attention mechanisms
    """
    def __init__(self, network_architecture):
        super(SpaTranslator_Aligner, self).__init__()
        
        # Extract architecture dimensions
        [input_dimension, latent_dimension, embedding_dimension] = network_architecture
        
        # Define encoder components
        self.encoder_layer1 = GraphAttentionLayer(
            input_dimension, 
            latent_dimension, 
            attention_heads=1, 
            concatenate_heads=False,
            attention_dropout=0, 
            include_self_edges=False, 
            use_bias=False
        )
        
        self.encoder_layer2 = GraphAttentionLayer(
            latent_dimension, 
            embedding_dimension, 
            attention_heads=1, 
            concatenate_heads=False,
            attention_dropout=0, 
            include_self_edges=False, 
            use_bias=False
        )
        
        # Define decoder components (weights tied to encoder)
        self.decoder_layer1 = GraphAttentionLayer(
            embedding_dimension, 
            latent_dimension, 
            attention_heads=1, 
            concatenate_heads=False,
            attention_dropout=0, 
            include_self_edges=False, 
            use_bias=False
        )
        
        self.decoder_layer2 = GraphAttentionLayer(
            latent_dimension, 
            input_dimension, 
            attention_heads=1, 
            concatenate_heads=False,
            attention_dropout=0, 
            include_self_edges=False, 
            use_bias=False
        )
    
    def forward(self, x, graph_edges):
        """
        Forward pass through the network
        
        Args:
            x: Input feature matrix
            graph_edges: Edge indices defining graph structure
            
        Returns:
            embeddings: The latent embeddings
            reconstructed: The reconstructed input
        """
        # Encoder pathway
        intermediate = F.elu(self.encoder_layer1(x, graph_edges))
        embeddings = self.encoder_layer2(intermediate, graph_edges, compute_attention=False)
        
        # Tie weights between encoder and decoder
        self.decoder_layer1.weight_matrix.data = self.encoder_layer2.weight_matrix.transpose(0, 1)
        self.decoder_layer1.shared_weights.data = self.encoder_layer2.shared_weights.transpose(0, 1)
        self.decoder_layer2.weight_matrix.data = self.encoder_layer1.weight_matrix.transpose(0, 1)
        self.decoder_layer2.shared_weights.data = self.encoder_layer1.shared_weights.transpose(0, 1)
        
        # Decoder pathway
        decoder_intermediate = F.elu(self.decoder_layer1(
            embeddings, 
            graph_edges, 
            compute_attention=True,
            external_attention=self.encoder_layer1.attention_weights
        ))
        
        reconstructed = self.decoder_layer2(decoder_intermediate, graph_edges, compute_attention=False)
        
        return embeddings, reconstructed