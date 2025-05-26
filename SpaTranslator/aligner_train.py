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
from .graph_align import SpaTranslator_Aligner

cudnn.deterministic = True
cudnn.benchmark = True


def train_spatial_aligner(adata, hidden_dims=[512, 128], num_epochs=1000, learning_rate=0.001, output_key='AlignedEmbedding',
                          grad_clip=5.0, l2_reg=0.0001, triplet_margin=1.0, verbose=False,
                          seed=666, integration_order=None, knn_neighbors=100,
                          device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """
    Trains a graph attention-based autoencoder for spatial transcriptomic batch correction.

    Parameters
    ----------
    adata : AnnData
        The AnnData object from the Scanpy package.
    hidden_dims : list
        List specifying the encoder architecture: [input_dim, hidden_dim, output_dim].
    num_epochs : int
        Number of epochs for training.
    learning_rate : float
        Learning rate for Adam optimizer.
    output_key : str
        Key under which the latent embeddings will be stored in `adata.obsm`.
    grad_clip : float
        Gradient clipping threshold.
    l2_reg : float
        Weight decay (L2 regularization) for Adam optimizer.
    triplet_margin : float
        Margin parameter for triplet loss, enforcing separation of negative pairs.
    integration_order : tuple or None
        Order of integration for multiple slices. E.g., `(0, 1)` aligns slice 0 to slice 1.
    knn_neighbors : int
        Number of nearest neighbors for constructing mutual nearest neighbors (MNN).
    device : torch.device
        Device to use for training (`cuda` or `cpu`).

    Returns
    -------
    AnnData
        The modified AnnData object with learned embeddings.
    """

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Extract unique batch names and edge list from AnnData
    batch_labels = np.array(adata.obs['batch_name'].unique())
    edge_data = adata.uns['edgeList']

    # Prepare graph data for PyTorch
    graph_data = Data(
        edge_index=torch.LongTensor(np.array([edge_data[0], edge_data[1]])),
        prune_edge_index=torch.LongTensor(np.array([])),
        x=torch.FloatTensor(adata.X.todense())
    ).to(device)

    # Initialize the spatial graph aligner model
    model = SpaTranslator_Aligner(network_architecture=[graph_data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    if verbose:
        print(model)

    print('Pretraining')
    for epoch in tqdm(range(500)):
        model.train()
        optimizer.zero_grad()
        latent_repr, reconstructed = model(graph_data.x, graph_data.edge_index)

        loss = F.mse_loss(graph_data.x, reconstructed)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    # Save the pretraining embeddings
    with torch.no_grad():
        latent_repr, _ = model(graph_data.x, graph_data.edge_index)
    adata.obsm['SpaAligner'] = latent_repr.cpu().detach().numpy()

    # Training phase with triplet loss
    print('Training with MNN loss')
    for epoch in tqdm(range(500, num_epochs)):
        # if epoch % 10 == 0 or epoch == 500:
        #     if verbose:
        #         print(f'Updating triplets at epoch {epoch}')

            adata.obsm['SpaAligner'] = latent_repr.cpu().detach().numpy()

            # Construct mutual nearest neighbors (MNN)
            mnn_dict = generate_mnn_dict(
                adata, representation='SpaAligner', batch_key='batch_name', k=knn_neighbors,
                pairs=integration_order, verbosity=0
            )

            anchor_idx, positive_idx, negative_idx = [], [], []

            for batch_pair, mnn_pairs in mnn_dict.items():
                batch_labels_mapped = adata.obs['batch_name'][mnn_pairs.keys()]
                batch_cell_dict = {
                    batch_id: adata.obs_names[adata.obs['batch_name'] == batch_id].values
                    for batch_id in batch_labels
                }

                for anchor in mnn_pairs.keys():
                    positive_spot = mnn_pairs[anchor][0]  # Select the first MNN
                    negative_spot = np.random.choice(batch_cell_dict[batch_labels_mapped[anchor]])

                    anchor_idx.append(anchor)
                    positive_idx.append(positive_spot)
                    negative_idx.append(negative_spot)

            # Convert spot names to numerical indices
            cell_idx_map = dict(zip(adata.obs_names, range(adata.shape[0])))
            anchor_idx = np.array([cell_idx_map[cell] for cell in anchor_idx])
            positive_idx = np.array([cell_idx_map[cell] for cell in positive_idx])
            negative_idx = np.array([cell_idx_map[cell] for cell in negative_idx])

            model.train()
            optimizer.zero_grad()
            latent_repr, reconstructed = model(graph_data.x, graph_data.edge_index)

            # Compute MSE loss
            mse_loss = F.mse_loss(graph_data.x, reconstructed)

            # Compute triplet loss
            triplet_loss_fn = torch.nn.TripletMarginLoss(margin=triplet_margin, p=2, reduction='mean')
            triplet_loss_value = triplet_loss_fn(latent_repr[anchor_idx], latent_repr[positive_idx], latent_repr[negative_idx])

            # Total loss and optimization step
            total_loss = mse_loss + triplet_loss_value
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

    # Save the final learned embeddings
    model.eval()
    adata.obsm[output_key] = latent_repr.cpu().detach().numpy()
    return adata
