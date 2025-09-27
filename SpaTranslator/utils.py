import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import itertools
import hnswlib
import networkx as nx
import anndata as ad
import scanpy as sc
import scipy.linalg
import torch
import episcanpy.api as epi
import random
from torch.utils.data import Dataset
from scipy import sparse

def build_spatial_graph(adata, radius_threshold=None, knn_neighbors=None, 
                        max_neighbors=50, method='Radius', verbose=True):
    """
    Constructs a spatial neighborhood graph for an AnnData object.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing spatial coordinates in `adata.obsm['spatial']`.
    radius_threshold : float, optional
        Distance threshold for linking cells when using the 'Radius' method.
    knn_neighbors : int, optional
        Number of nearest neighbors to connect each cell to when using the 'KNN' method.
    max_neighbors : int, default=50
        Maximum number of neighbors considered when searching for spatial relationships.
    method : str, default='Radius'
        Method to construct the spatial graph:
        - 'Radius': Connects points within `radius_threshold` distance.
        - 'KNN': Links each point to its `knn_neighbors` closest neighbors.
    verbose : bool, default=True
        If True, prints details about the generated graph.

    Returns
    -------
    None
        Updates `adata.uns['SpatialGraph']` with edge list and `adata.uns['adj_matrix']` with adjacency matrix.
    """

    assert method in ['Radius', 'KNN'], "Invalid method! Choose either 'Radius' or 'KNN'."

    if verbose:
        print("Constructing the spatial graph")

    # Retrieve spatial coordinates
    spatial_coords = pd.DataFrame(adata.obsm['spatial'], index=adata.obs.index, columns=['x', 'y'])

    # Compute spatial neighbors using BallTree
    neighbor_model = NearestNeighbors(n_neighbors=max_neighbors + 1, algorithm='ball_tree')
    neighbor_model.fit(spatial_coords)
    dist_matrix, neighbor_indices = neighbor_model.kneighbors(spatial_coords)

    if method == 'KNN':
        neighbor_indices, dist_matrix = neighbor_indices[:, 1:knn_neighbors + 1], dist_matrix[:, 1:knn_neighbors + 1]
    else:  # 'Radius' method
        neighbor_indices, dist_matrix = neighbor_indices[:, 1:], dist_matrix[:, 1:]

    # Create edge list DataFrame
    edge_list = []
    for i in range(neighbor_indices.shape[0]):
        for j in range(neighbor_indices.shape[1]):
            edge_list.append((i, neighbor_indices[i, j], dist_matrix[i, j]))

    spatial_graph_df = pd.DataFrame(edge_list, columns=['Source', 'Target', 'Distance'])

    # Apply radius-based filtering if applicable
    if method == 'Radius' and radius_threshold is not None:
        spatial_graph_df = spatial_graph_df[spatial_graph_df['Distance'] < radius_threshold]

    # Convert numeric indices back to cell names
    index_to_cell = {i: cell for i, cell in enumerate(spatial_coords.index)}
    spatial_graph_df['Source'] = spatial_graph_df['Source'].map(index_to_cell)
    spatial_graph_df['Target'] = spatial_graph_df['Target'].map(index_to_cell)

    # Store spatial graph in AnnData
    adata.uns['SpatialGraph'] = spatial_graph_df

    if verbose:
        print(f"Generated graph with {spatial_graph_df.shape[0]} edges across {adata.n_obs} cells.")
        print(f"Average neighbors per cell: {spatial_graph_df.shape[0] / adata.n_obs:.4f}")

    # Construct adjacency matrix
    if 'SpatialGraph' not in adata.uns:
        raise ValueError("SpatialGraph not found! Ensure `build_spatial_graph` was executed successfully.")

    cell_to_index = {cell: idx for idx, cell in enumerate(adata.obs.index)}
    
    spatial_graph_df['Source'] = spatial_graph_df['Source'].map(cell_to_index)
    spatial_graph_df['Target'] = spatial_graph_df['Target'].map(cell_to_index)

    adjacency_matrix = sp.coo_matrix(
        (np.ones(spatial_graph_df.shape[0]), (spatial_graph_df['Source'], spatial_graph_df['Target'])),
        shape=(adata.n_obs, adata.n_obs)
    ) + sp.eye(adata.n_obs)  # Add self-loops

    adata.uns['adj_matrix'] = adjacency_matrix



def generate_mnn_dict(adata, representation, batch_key, k=50, save_disk=True, use_approx=True, verbosity=1, pairs=None):
    cell_labels = adata.obs_names
    batch_categories = adata.obs[batch_key].unique()
    data_splits, feature_matrices, cell_groups = [], [], []
    
    for batch in batch_categories:
        data_splits.append(adata[adata.obs[batch_key] == batch])
        feature_matrices.append(adata[adata.obs[batch_key] == batch].obsm[representation])
        cell_groups.append(cell_labels[adata.obs[batch_key] == batch])
    
    batch_reference = pd.DataFrame(batch_categories)
    mnn_pairs = {}
    
    if pairs is None:
        pairs = list(itertools.combinations(range(len(cell_groups)), 2))
    
    for pair in pairs:
        i, j = pair
        pair_key = f"{batch_reference.iloc[i, 0]}_{batch_reference.iloc[j, 0]}"
        mnn_pairs[pair_key] = {}
        
        if verbosity > 0:
            print(f'Processing dataset pair: {i}, {j}')
        
        cell_set1, cell_set2 = list(cell_groups[j]), list(cell_groups[i])
        features1, features2 = adata[cell_set1].obsm[representation], adata[cell_set2].obsm[representation]
        
        matched_pairs = find_mnn(features1, features2, cell_set1, cell_set2, knn=k, save_disk=save_disk, approx=use_approx)
        
        graph = nx.Graph()
        graph.add_edges_from(matched_pairs)
        node_array = np.array(graph.nodes)
        adjacency = nx.adjacency_matrix(graph)
        adjacency_lists = np.split(adjacency.indices, adjacency.indptr[1:-1])
        
        for idx, node in enumerate(node_array):
            neighbors = list(node_array[adjacency_lists[idx]])
            mnn_pairs[pair_key][node] = neighbors
    
    return mnn_pairs



def approximate_nn(query_set, reference_set, query_labels, reference_labels, knn=50):
    dim = reference_set.shape[1]
    num_elements = reference_set.shape[0]
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=100, M=16)
    index.set_ef(10)
    index.add_items(reference_set)
    
    neighbors, _ = index.knn_query(query_set, k=knn)
    matches = {(query_labels[i], reference_labels[j]) for i, js in enumerate(neighbors) for j in js}
    return matches


def exact_nn(query_set, reference_set, query_labels, reference_labels, knn=50, metric_order=2):
    neighbor_finder = NearestNeighbors(n_neighbors=knn, p=metric_order)
    neighbor_finder.fit(reference_set)
    neighbor_indices = neighbor_finder.kneighbors(query_set, return_distance=False)
    
    matches = {(query_labels[i], reference_labels[j]) for i, js in enumerate(neighbor_indices) for j in js}
    return matches


def annoy_nn(query_set, reference_set, query_labels, reference_labels, knn=20, metric='euclidean', n_trees=50, save_disk=True):
    index = AnnoyIndex(reference_set.shape[1], metric=metric)
    if save_disk:
        index.on_disk_build('annoy.index')
    for i in range(reference_set.shape[0]):
        index.add_item(i, reference_set[i, :])
    index.build(n_trees)
    
    neighbor_indices = [index.get_nns_by_vector(vec, knn, search_k=-1) for vec in query_set]
    matches = {(query_labels[i], reference_labels[j]) for i, js in enumerate(neighbor_indices) for j in js}
    return matches


def find_mnn(set1, set2, labels1, labels2, knn=20, save_disk=True, approx=True):
    if approx:
        matches1 = approximate_nn(set1, set2, labels1, labels2, knn=knn)
        matches2 = approximate_nn(set2, set1, labels2, labels1, knn=knn)
    else:
        matches1 = exact_nn(set1, set2, labels1, labels2, knn=knn)
        matches2 = exact_nn(set2, set1, labels2, labels1, knn=knn)
    
    return matches1 & {(b, a) for a, b in matches2}


def find_peak_overlaps(query, key):
    q_seqname = np.array(query.get_seqnames())
    k_seqname = np.array(key.get_seqnames())
    q_start = np.array(query.get_start())
    k_start = np.array(key.get_start())
    q_width = np.array(query.get_width())
    k_width = np.array(key.get_width())
    q_end = q_start + q_width
    k_end = k_start + k_width

    q_index = 0
    k_index = 0
    overlap_index = [[] for i in range(len(query))]
    overlap_count = [0 for i in range(len(query))]
    query = query.sort_values(["seqname", "start"])
    key = key.sort_values(["seqname", "start"])


    while True:
        if q_index == len(query) or k_index == len(key):
            return overlap_index, overlap_count

        if q_seqname[q_index] == k_seqname[k_index]:
            if k_start[k_index] >= q_start[q_index] and k_end[k_index] <= q_end[q_index]:
                overlap_index[q_index].append(k_index)
                overlap_count[q_index] += 1
                k_index += 1
            elif k_start[k_index] < q_start[q_index]:
                k_index += 1
            else:
                q_index += 1
        elif q_seqname[q_index] < k_seqname[k_index]:
            q_index += 1
        else:
            k_index += 1
            
def TFIDF(count_mat, type_=2):
    # Perform TF-IDF (count_mat: peak*cell)
    def tfidf1(count_mat):
        if not scipy.sparse.issparse(count_mat):
            count_mat = scipy.sparse.coo_matrix(count_mat)

        nfreqs = count_mat.multiply(1.0 / count_mat.sum(axis=0))
        tfidf_mat = nfreqs.multiply(np.log(1 + 1.0 * count_mat.shape[1] / count_mat.sum(axis=1)).reshape(-1, 1)).tocoo()

        return scipy.sparse.csr_matrix(tfidf_mat)

    # Perform Signac TF-IDF (count_mat: peak*cell) [selected]
    def tfidf2(count_mat):
        if not scipy.sparse.issparse(count_mat):
            count_mat = scipy.sparse.coo_matrix(count_mat)

        tf_mat = count_mat.multiply(1.0 / count_mat.sum(axis=0))
        signac_mat = (1e4 * tf_mat).multiply(1.0 * count_mat.shape[1] / count_mat.sum(axis=1).reshape(-1, 1))
        signac_mat = signac_mat.log1p()

        return scipy.sparse.csr_matrix(signac_mat)

    # Perform TF-IDF (count_mat: ?)
    from sklearn.feature_extraction.text import TfidfTransformer
    def tfidf3(count_mat):
        model = TfidfTransformer(smooth_idf=False, norm="l2")
        model = model.fit(np.transpose(count_mat))
        model.idf_ -= 1
        tf_idf = np.transpose(model.transform(np.transpose(count_mat)))

        return scipy.sparse.csr_matrix(tf_idf)

    if type_ == 1:
        return tfidf1(count_mat)
    elif type_ == 2:
        return tfidf2(count_mat)
    else:
        return tfidf3(count_mat)
    
    
def peak_sets_alignment(adata_list, sep=(":", "-"), min_width=20, max_width=10000, min_gap_width=1,
                        ):
    from genomicranges import GenomicRanges
    from iranges import IRanges
    from biocutils.combine import combine

    ## Peak merging
    gr_list = []
    for i in range(len(adata_list)):
        seq_names = []
        starts = []
        widths = []
        regions = adata_list[i].var_names
        for region in regions:
            seq_names.append(region.split(sep[0])[0])
            if sep[0] == sep[1]:
                start, end = region.split(sep[0])[1:]
            else:
                start, end = region.split(sep[0])[1].split(sep[1])
            width = int(end) - int(start) + 1
            starts.append(int(start))
            widths.append(width)
        gr = GenomicRanges(seqnames=seq_names, ranges=IRanges(starts, widths)).sort()
        peaks = [seqname + sep[0] + str(start) + sep[1] + str(end) for seqname, start, end in
                 zip(gr.get_seqnames(), gr.get_start(), gr.get_end())]
        adata_list[i] = adata_list[i][:, peaks]
        gr_list.append(gr)

    gr_combined = combine(*gr_list)
    gr_merged = gr_combined.reduce(min_gap_width=min_gap_width).sort()
    print("Peak merged")

    ## Peak filtering
    # filter by intesect
    overlap_index_list = []
    index = np.ones(len(gr_merged)).astype(bool)
    for gr in gr_list:
        overlap_index, overlap_count = find_peak_overlaps(gr_merged, gr)
        index = (np.array(overlap_count) > 0) * index
        overlap_index_list.append(overlap_index)
    # filter by width
    index = index * (gr_merged.get_width() > min_width) * (gr_merged.get_width() < max_width)
    gr_merged = gr_merged.get_subset(index)
    common_peak = [seqname + ":" + str(start) + "-" + str(end) for seqname, start, end in
                   zip(gr_merged.get_seqnames(), gr_merged.get_start(), gr_merged.get_end())]
    print("Peak filtered")

    ## Merge count matrix
    adata_merged_list = []
    for adata, overlap_index in zip(adata_list, overlap_index_list):
        overlap_index = [overlap_index[i] for i in range(len(index)) if index[i]]
        X = adata.X.tocsc()
        X_merged = scipy.sparse.hstack([scipy.sparse.csr_matrix(X[:, cur].sum(axis=1)) for cur in overlap_index])
        adata_merged_list.append(
            sc.AnnData(X_merged, obs=adata.obs, var=pd.DataFrame(index=common_peak), obsm=adata.obsm))
    print("Matrix merged")

    return adata_merged_list

def preprocess_CAS(adata_list, adata_concat, binarize=False, use_fragment_count=False, tfidf=True,
                   min_cells_rate=0.03, min_features=1, tfidf_type=2):

    epi.pp.filter_features(adata_concat, min_cells=int(min_cells_rate * adata_concat.shape[0]))
    epi.pp.filter_cells(adata_concat, min_features=min_features)

    if binarize and use_fragment_count:
        raise ValueError("'binarize' and 'use_fragment_count' cannot be set to True at the same time !")

    elif binarize:
        epi.pp.binarize(adata_concat)

    elif use_fragment_count:
        adata_concat.X = scipy.sparse.csr_matrix(np.ceil((adata_concat.X / 2).toarray()))

    if tfidf:
        adata_concat.X = TFIDF(adata_concat.X.T, type_=tfidf_type).T.copy()
    else:
        epi.pp.normalize_total(adata_concat, target_sum=10000)
        epi.pp.log1p(adata_concat)

    for i in range(len(adata_list)):
        obs_list = [item for item in adata_list[i].obs_names if item in adata_concat.obs_names]
        var_names = adata_concat.var_names
        adata_list[i] = adata_list[i][obs_list, var_names]
        
        
def find_peak_overlaps(query, key):
    q_seqname = np.array(query.get_seqnames())
    k_seqname = np.array(key.get_seqnames())
    q_start = np.array(query.get_start())
    k_start = np.array(key.get_start())
    q_width = np.array(query.get_width())
    k_width = np.array(key.get_width())
    q_end = q_start + q_width
    k_end = k_start + k_width

    q_index = 0
    k_index = 0
    overlap_index = [[] for i in range(len(query))]
    overlap_count = [0 for i in range(len(query))]

    while True:
        if q_index == len(query) or k_index == len(key):
            return overlap_index, overlap_count

        if q_seqname[q_index] == k_seqname[k_index]:
            if k_start[k_index] >= q_start[q_index] and k_end[k_index] <= q_end[q_index]:
                overlap_index[q_index].append(k_index)
                overlap_count[q_index] += 1
                k_index += 1
            elif k_start[k_index] < q_start[q_index]:
                k_index += 1
            else:
                q_index += 1
        elif q_seqname[q_index] < k_seqname[k_index]:
            q_index += 1
        else:
            k_index += 1
 

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    
  
def split_dataset_by_slices(RNA_data, ATAC_data, seed=52340):
    """Splits the dataset into training and validation sets."""
    
    if seed is not None:
        setup_seed(seed)
    
    indices = list(range(len(RNA_data.obs_names)))
    random.shuffle(indices)
    
    val_size = int(0.2 * len(indices))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    print(f"Training set size: {len(train_indices)}")
    print(f"Validation set size: {len(val_indices)}")
    
    return train_indices, val_indices


def clustering(adata, n_clusters=7, add_key='leiden', method='mclust', start=0.1, end=3.0, increment=0.01, use_pca=False, n_comps=30,n_neighbors_input=20):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float 
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.  
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """
    
    if method == 'leiden':
       if use_pca: 
          res = search_res(adata, n_clusters, method=method, start=start, end=end, increment=increment, n_neighbors =n_neighbors_input)
       else:
          res = search_res(adata, n_clusters, method=method, start=start, end=end, increment=increment,n_neighbors =n_neighbors_input) 
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['leiden']
       
def search_res(adata, n_clusters, method='leiden', start=0.1, end=3.0, increment=0.01,n_neighbors=20 ):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            return res

    #assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
    res = 0.1
    return res 

def pca(adata, use_reps=None, n_comps=10):
    
    """Dimension reduction with PCA algorithm"""
    
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray()) 
       else:   
          feat_pca = pca.fit_transform(adata.X)
    
    return feat_pca


def preprocess_rna(
    RNA_data,
    normalize_total=True,
    log1p=True,
    use_hvg=True,
    n_top_genes=3000,
):
    """
    Preprocess RNA sequencing data using Scanpy, including normalization, log transformation, 
    and selection of highly variable genes.
    
    Parameters:
    ----------
    RNA_data : AnnData
        Input RNA data in AnnData format.
    
    normalize_total : bool, default=True
        Whether to normalize the total counts per cell.
    
    log1p : bool, default=True
        Whether to apply log(1p) transformation.
    
    use_hvg : bool, default=True
        Whether to select highly variable genes.
    
    n_top_genes : int, default=3000
        Number of highly variable genes to retain (only used if `use_hvg=True`).
    
    
    Returns:
    ---------
    RNA_data_processed : AnnData
        Preprocessed RNA data.
    """
    RNA_data_processed = RNA_data.copy()
    
    RNA_data_processed.var_names_make_unique()

    if not (normalize_total and log1p and use_hvg):
        print("Recommended to use default settings for optimal results.")

    if normalize_total:
        print("Applying total count normalization for RNA.")
        sc.pp.normalize_total(RNA_data_processed)
        
    if log1p:
        print("Applying log1p transformation for RNA.")
        sc.pp.log1p(RNA_data_processed)
    
    if use_hvg:
        print(f"Selecting top {n_top_genes} highly variable genes for RNA.")
        sc.pp.highly_variable_genes(RNA_data_processed, n_top_genes=n_top_genes)
        RNA_data_processed = RNA_data_processed[:, RNA_data_processed.var['highly_variable']]
    
    return RNA_data_processed



def preprocess_atac(
    ATAC_data,
    binary_data=True,
    filter_features=True,
    fpeaks=0.005,
    tfidf=True,
    normalize=True,
):
    """
    Preprocess ATAC sequencing data using binarization, peak filtering, TF-IDF transformation, 
    and normalization.

    Parameters:
    ----------
    ATAC_data : AnnData
        Input ATAC data in AnnData format.

    binary_data : bool, default=True
        Whether to binarize the ATAC data.

    filter_features : bool, default=True
        Whether to filter out low-expressed peaks.

    fpeaks : float, default=0.005
        Minimum fraction of cells a peak must be present in to be retained.

    tfidf : bool, default=True
        Whether to apply TF-IDF transformation.

    normalize : bool, default=True
        Whether to scale data to the range [0, 1].


    Returns:
    ---------
    ATAC_data_processed : AnnData
        Preprocessed ATAC data.

    divide_title : np.ndarray
        Matrix used in the TF-IDF transformation.

    multiply_title : np.ndarray
        Matrix used in the TF-IDF transformation.

    max_temp : float
        Maximum scaling factor used in normalization.
    """
    ATAC_data_processed = ATAC_data.copy()
    divide_title, multiply_title, max_temp = None, None, None

    if not (binary_data and filter_features and tfidf and normalize):
        print("Recommended to use default settings for optimal results.")

    if binary_data:
        print("Binarizing ATAC data.")
        epi.pp.binarize(ATAC_data_processed)
        
    if filter_features:
        min_cells = np.ceil(fpeaks * ATAC_data.shape[0])
        print(f"Filtering out peaks present in fewer than {fpeaks*100:.2f}% of cells.")
        epi.pp.filter_features(ATAC_data_processed, min_cells=min_cells)

    if tfidf:
        print("Applying TF-IDF transformation.")
        count_matrix = ATAC_data_processed.X.copy()
        ATAC_data_processed.X = TFIDF(count_matrix)
    
    if normalize:
        print("Normalizing data to range [0, 1] for ATAC.")
        max_temp = np.max(ATAC_data_processed.X)
        ATAC_data_processed.X /= max_temp

    return ATAC_data_processed, max_temp


def convert_tensor_to_anndata(tensor_list, threshold=1e-4):
    """
    Convert a list of tensors into an AnnData object.

    Parameters
    ----------
    tensor_list : list of torch.Tensor
        The list of tensors to concatenate and convert.

    threshold : float, optional
        Threshold below which values are set to zero for sparsity. Default is 1e-4.

    Returns
    -------
    AnnData
        An AnnData object containing the sparse matrix.
    """
    concatenated = torch.cat(tensor_list)
    masked_tensor = torch.where(concatenated < threshold, torch.tensor(0.0), concatenated)
    sparse_matrix = sparse.csr_matrix(masked_tensor)
    return ad.AnnData(sparse_matrix)


class Single_omics_dataset(Dataset):
    def __init__(self, input_matrix):
        """
        Dataset class for single-omics data.

        Parameters
        ----------
        input_matrix : np.ndarray
            A 2D matrix representing omics data.
        """
        self.dataset = input_matrix
        self.id_list = list(range(input_matrix.shape[0]))

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        row_data = self.dataset[self.id_list[idx], :]
        return torch.from_numpy(row_data), self.id_list[idx]



class PairedOmicsDataset(Dataset):
    def __init__(self, rna_mat, atac_mat, spatial_rna, spatial_atac, rna_indices, atac_indices):
        """
        Dataset class for paired RNA and ATAC data with optional spatial information.

        Parameters
        ----------
        rna_mat : np.ndarray
            RNA expression matrix.

        atac_mat : np.ndarray
            ATAC accessibility matrix.

        spatial_rna : np.ndarray
            RNA adjacency matrix.

        spatial_atac : np.ndarray
            ATAC adjacency matrix.

        rna_indices : list of int
            Row indices for RNA data.

        atac_indices : list of int
            Row indices for ATAC data.
        """

        self.RNA_dataset = rna_mat      
        self.ATAC_dataset = atac_mat
        self.id_list_r = rna_indices
        self.id_list_a = atac_indices
        self.r_count = len(self.id_list_r)
        self.a_count = len(self.id_list_a)
        self.RNA_data_spatial = spatial_rna
        self.ATAC_data_spatial = spatial_atac

    def __len__(self):
        return self.r_count

    def __getitem__(self, idx):
        rna_feat = torch.from_numpy(self.RNA_dataset[self.id_list_r[idx], :])
        atac_feat = torch.from_numpy(self.ATAC_dataset[self.id_list_a[idx], :])
        joint_feat = torch.cat([rna_feat, atac_feat])
        return joint_feat, self.id_list_r[idx], self.id_list_a[idx]

    

class EarlyStopping:
    """Cite from https://github.com/Bjarten/early-stopping-pytorch"""
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save_model_dict(path) # save best model here
        self.val_loss_min = val_loss

