from typing import Union, Tuple, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn
from torch_sparse import SparseTensor, set_diag
from torch_geometric.typing import OptPairTensor, Adj, Size, NoneType, OptTensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class GraphAttentionLayer(MessagePassing):
    """
    Graph Attention Layer - Implementation of the GAT mechanism
    Based on the paper: "Graph Attention Networks" (https://arxiv.org/abs/1710.10903)
    """
    def __init__(
        self, 
        input_channels: Union[int, Tuple[int, int]],
        output_channels: int, 
        attention_heads: int = 1, 
        concatenate_heads: bool = True,
        leaky_slope: float = 0.2, 
        attention_dropout: float = 0.0,
        include_self_edges: bool = True, 
        use_bias: bool = True,
        edge_pruning_weight: float = 0.0, 
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super(GraphAttentionLayer, self).__init__(node_dim=0, **kwargs)
        
        # Save parameters
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.attention_heads = attention_heads
        self.concatenate_heads = concatenate_heads
        self.leaky_slope = leaky_slope
        self.attention_dropout = attention_dropout
        self.include_self_edges = include_self_edges
        self.edge_pruning_weight = edge_pruning_weight
        
        # Initialize transformation matrices
        self.weight_matrix = nn.Parameter(torch.zeros(size=(input_channels, attention_heads * output_channels)))
        nn.init.xavier_normal_(self.weight_matrix.data, gain=1.414)
        self.shared_weights = self.weight_matrix
        
        # Initialize attention mechanisms
        self.key_attention = Parameter(torch.Tensor(1, attention_heads, output_channels))
        self.query_attention = Parameter(torch.Tensor(1, attention_heads, output_channels))
        nn.init.xavier_normal_(self.key_attention.data, gain=1.414)
        nn.init.xavier_normal_(self.query_attention.data, gain=1.414)
        
        # Storage for attention coefficients
        self.attention_coeffs = None
        self.attention_weights = None

    def forward(
        self, 
        node_features: Union[Tensor, OptPairTensor], 
        connectivity: Adj, 
        pruned_connectivity: Adj = None,
        dim_size: Size = None, 
        return_attention: bool = None, 
        compute_attention: bool = True, 
        external_attention: OptTensor = None
    ):
        H, C = self.attention_heads, self.output_channels
        
        # Transform input features
        if isinstance(node_features, Tensor):
            assert node_features.dim() == 2, "Non-static graphs required for GraphAttentionLayer"
            transformed_features = transformed_neighbors = torch.mm(node_features, self.weight_matrix).view(-1, H, C)
        else:  # Handle bipartite graph case
            source_features, target_features = node_features
            assert source_features.dim() == 2, "Non-static graphs required for GraphAttentionLayer"
            transformed_features = torch.mm(source_features, self.weight_matrix).view(-1, H, C)
            transformed_neighbors = None
            if target_features is not None:
                transformed_neighbors = torch.mm(target_features, self.shared_weights).view(-1, H, C)
        
        transformed_pair = (transformed_features, transformed_neighbors)
        
        # Skip attention computation if requested
        if not compute_attention:
            return transformed_pair[0].mean(dim=1)
        
        # Compute attention weights or use provided ones
        if external_attention is None:
            query_score = (transformed_features * self.query_attention).sum(dim=-1)
            key_score = None if transformed_neighbors is None else (transformed_neighbors * self.key_attention).sum(-1)
            attention_scores = (query_score, key_score)
            self.attention_weights = attention_scores
        else:
            attention_scores = external_attention
        
        # Add self-loops to the graph structure if needed
        if self.include_self_edges:
            if isinstance(connectivity, Tensor):
                num_nodes = transformed_features.size(0)
                if transformed_neighbors is not None:
                    num_nodes = min(num_nodes, transformed_neighbors.size(0))
                num_nodes = min(dim_size) if dim_size is not None else num_nodes
                connectivity, _ = remove_self_loops(connectivity)
                connectivity, _ = add_self_loops(connectivity, num_nodes=num_nodes)
            elif isinstance(connectivity, SparseTensor):
                connectivity = set_diag(connectivity)
        
        # Apply message passing with pruning if specified
        if self.edge_pruning_weight == 0:
            output = self.propagate(connectivity, 
                                    x=transformed_pair, 
                                    alpha=attention_scores, 
                                    size=dim_size)
        else:
            standard_pass = self.propagate(connectivity, 
                                          x=transformed_pair, 
                                          alpha=attention_scores, 
                                          size=dim_size)
            pruned_pass = self.propagate(pruned_connectivity, 
                                        x=transformed_pair, 
                                        alpha=attention_scores, 
                                        size=dim_size)
            output = (1-self.edge_pruning_weight) * standard_pass + self.edge_pruning_weight * pruned_pass
        
        # Store attention coefficients for later use
        attention_weights = self.attention_coeffs
        self.attention_coeffs = None
        
        # Apply final transformations
        if self.concatenate_heads:
            output = output.view(-1, self.attention_heads * self.output_channels)
        else:
            output = output.mean(dim=1)
        
        # Return attention weights if requested
        if isinstance(return_attention, bool):
            if isinstance(connectivity, Tensor):
                return output, (connectivity, attention_weights)
            elif isinstance(connectivity, SparseTensor):
                return output, connectivity.set_value(attention_weights, layout='coo')
        else:
            return output

    def message(
        self, 
        x_j: Tensor, 
        alpha_j: Tensor, 
        alpha_i: OptTensor,
        index: Tensor, 
        ptr: OptTensor,
        size_i: Optional[int]
    ) -> Tensor:
        # Combine attention coefficients
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        
        # Use sigmoid instead of leaky relu for attention activation
        alpha = torch.sigmoid(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        
        # Store for potential return
        self.attention_coeffs = alpha
        
        # Apply dropout regularization
        alpha = F.dropout(alpha, p=self.attention_dropout, training=self.training)
        
        # Apply attention weights to the features
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(
            self.__class__.__name__,
            self.input_channels,
            self.output_channels, 
            self.attention_heads
        )