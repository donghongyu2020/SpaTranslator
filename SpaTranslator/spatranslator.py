import random
import gc
import scanpy as sc
import anndata as ad
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from SpaTranslator.utils import *
from SpaTranslator.translation_train import Model
class SpaTranslator:
    def __init__(self):
        """
        SpaTranslator model.
        """
        self.set_random_seed(52340)
        print("SpaTranslator model initialized.")

    def set_random_seed(self, seed):
        random.seed(seed)

    def load_data(self, RNA_data, ATAC_data, other_data, train_id, validation_id=None,mode="R2A"):
        """
        Load data into the SpaTranslator model.
        """
        self.RNA_data = RNA_data.copy()
        self.ATAC_data = ATAC_data.copy()
        self.other_data = other_data.copy()
        self.mode = mode
            
        self.is_processed = False

        if validation_id is None:
            self.train_id = train_id[:]
            random.shuffle(self.train_id)
            split_idx = int(len(self.train_id) * 0.8)
            self.train_id_r = self.train_id[:split_idx]
            self.train_id_a = self.train_id[:split_idx]
            self.validation_id_r = self.train_id[split_idx:]
            self.validation_id_a = self.train_id[split_idx:]
            del self.train_id
        else:
            self.train_id_r = train_id[:]
            self.train_id_a = train_id[:]
            self.validation_id_r = validation_id[:]
            self.validation_id_a = validation_id[:]

        print("Data successfully loaded.")

    def preprocess_data(self, normalize_total=True, log1p=True, use_hvg=True, n_top_genes=3000,
                        binary_data=True, filter_features=True, fpeaks=0.005, tfidf=True, normalize=True):
        """
        Preprocess RNA and ATAC data for SpaTranslator.
        """
        if hasattr(self, "is_processed") and self.is_processed:
            print("Warning: Data has already been processed!")
            return
        
        self.RNA_data_p = preprocess_rna(self.RNA_data, normalize_total, log1p, use_hvg, n_top_genes)
        self.ATAC_data_p = preprocess_atac(self.ATAC_data, binary_data, filter_features, fpeaks, tfidf, normalize)[0]
        self.is_processed = True
        print("Data preprocessing completed.")

    def augment_data(self, aug_type=None, MultiVI_path=None):
        """
        Perform data augmentation.
        """
        if aug_type in ["cell_type_augmentation", "type_cluster_augmentation"] and "cell_type" not in self.RNA_data.obs:
            print('Cell type information not found. Switching to MultiVI-based augmentation.')
            aug_type = "MultiVI_augmentation"
            
        if aug_type == 'cell_type_augmentation' and 'cell_type' in self.RNA_data.obs.keys():
            print("Performing cell type-based augmentation")

            # Set the number of copies for data augmentation
            copy_count = 3
            random.seed(52340)

            # Reassign unique string indices to ATAC_data observations
            self.ATAC_data.obs.index = [str(i) for i in range(len(self.ATAC_data.obs.index))]

            # Extract cell type information for training set
            cell_type = self.ATAC_data.obs.cell_type.iloc[self.train_id_a]

            # Identify the most frequent cell type
            max_cell_type_name = cell_type.value_counts().idxmax()

            # Iterate through all cell types
            for cell_type_name in cell_type.cat.categories:
                # Get indices of cells belonging to the current cell type
                cell_indices = list(cell_type[cell_type == cell_type_name].index.astype(int))
                
                # Perform augmentation for underrepresented cell types
                for _ in range(copy_count - 1):
                    random.shuffle(cell_indices)
                    self.train_id_r.extend(cell_indices)

                    random.shuffle(cell_indices)
                    self.train_id_a.extend(cell_indices)
                    
            print("Data augmentation completed.")


        if aug_type == "MultiVI_augmentation":
            import scvi
            print("Performing MultiVI-based augmentation...")
            adata = ad.AnnData(sp.hstack((self.RNA_data.X, self.ATAC_data.X)))
            adata.X = adata.X.tocsr()
            adata.obs = self.RNA_data.obs.copy()

            m = len(self.RNA_data.var.index)
            n = len(self.ATAC_data.var.index)
            adata.var.index = pd.Series(
                [self.RNA_data.var.index[i] if i < m else self.ATAC_data.var.index[i - m] for i in range(m + n)],
                dtype="object"
            )
            adata.var["modality"] = pd.Series(
                ["Gene Expression" if i < m else "Peaks" for i in range(m + n)], dtype="object"
            ).values

            adata.var_names_make_unique()
            adata_mvi = scvi.data.organize_multiome_anndatas(adata)
            sc.pp.filter_genes(adata_mvi, min_cells=int(adata_mvi.shape[0] * 0.01))

            train_adata = adata_mvi[self.train_id_r, :]

            if MultiVI_path is None:
                print("Training a new MultiVI model...")
                scvi.model.MULTIVI.setup_anndata(train_adata, batch_key="modality")
                mvi = scvi.model.MULTIVI(
                    train_adata,
                    n_genes=(adata_mvi.var["modality"] == "Gene Expression").sum(),
                    n_regions=(adata_mvi.var["modality"] == "Peaks").sum(),
                )
                mvi.train()
            else:
                print(f"Loading pretrained MultiVI model from {MultiVI_path}")
                mvi = scvi.model.MULTIVI.load(MultiVI_path, adata=train_adata)

            train_adata.obsm["MultiVI_latent"] = mvi.get_latent_representation()
            leiden_adata = ad.AnnData(train_adata.obsm["MultiVI_latent"])
            sc.pp.neighbors(leiden_adata)
            sc.tl.leiden(leiden_adata, resolution=3)

            gc.collect()
            
            copy_count = 3
            random.seed(19193)
            cell_clusters = leiden_adata.obs.leiden

            for cluster in cell_clusters.cat.categories:
                cluster_indices = list(cell_clusters[cell_clusters == cluster].index.astype(int))
                for _ in range(copy_count - 1):
                    random.shuffle(cluster_indices)
                    for idx in cluster_indices:
                        self.train_id_r.append(self.train_id_r[idx])
                    random.shuffle(cluster_indices)
                    for idx in cluster_indices:
                        self.train_id_a.append(self.train_id_a[idx])

            print("Data augmentation completed.")
            
    def construct_model(
        self,
        chrom_list,
        R_encoder_nlayer=2, 
        A_encoder_nlayer=2,
        R_decoder_nlayer=2, 
        A_decoder_nlayer=2,
        R_encoder_dim_list=[256, 128],
        A_encoder_dim_list=[32, 128],
        R_decoder_dim_list=[128, 256],
        A_decoder_dim_list=[128, 32],
        R_encoder_act_list=[nn.LeakyReLU(), nn.LeakyReLU()],
        A_encoder_act_list=[nn.LeakyReLU(), nn.LeakyReLU()],
        R_decoder_act_list=[nn.LeakyReLU(), nn.LeakyReLU()],
        A_decoder_act_list=[nn.LeakyReLU(), nn.Sigmoid()],
        translator_embed_dim=128, 
        translator_input_dim_rna=128,
        translator_input_dim_atac=128,
        translator_embed_act_list=[nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()],
        discriminator_nlayer=1,
        discriminator_dim_list_rna=[128],
        discriminator_dim_list_atac=[128],
        discriminator_act_list=[nn.Sigmoid()],
        dropout_rate=0.1,
        R_noise_rate=0.5,
        A_noise_rate=0.3,
    ):
        
        # Adjust input dimensions
        R_encoder_dim_list.insert(0, self.RNA_data_p.X.shape[1]) 
        A_encoder_dim_list.insert(0, self.ATAC_data_p.X.shape[1])
        R_decoder_dim_list.append(self.RNA_data_p.X.shape[1]) 
        A_decoder_dim_list.append(self.ATAC_data_p.X.shape[1])

        # Scale ATAC dimensions based on chromosome count
        A_encoder_dim_list[1] *= len(chrom_list)
        A_decoder_dim_list[1] *= len(chrom_list)

        # Initialize the model
        self.model = Model(
            R_encoder_nlayer=R_encoder_nlayer, 
            A_encoder_nlayer=A_encoder_nlayer,
            R_decoder_nlayer=R_decoder_nlayer, 
            A_decoder_nlayer=A_decoder_nlayer,
            R_encoder_dim_list=R_encoder_dim_list,
            A_encoder_dim_list=A_encoder_dim_list,
            R_decoder_dim_list=R_decoder_dim_list,
            A_decoder_dim_list=A_decoder_dim_list,
            R_encoder_act_list=R_encoder_act_list,
            A_encoder_act_list=A_encoder_act_list,
            R_decoder_act_list=R_decoder_act_list,
            A_decoder_act_list=A_decoder_act_list,
            translator_embed_dim=translator_embed_dim, 
            translator_input_dim_rna=translator_input_dim_rna,
            translator_input_dim_atac=translator_input_dim_atac,
            translator_embed_act_list=translator_embed_act_list,
            discriminator_nlayer=discriminator_nlayer,
            discriminator_dim_list_rna=discriminator_dim_list_rna,
            discriminator_dim_list_atac=discriminator_dim_list_atac,
            discriminator_act_list=discriminator_act_list,
            dropout_rate=dropout_rate,
            R_noise_rate=R_noise_rate,
            A_noise_rate=A_noise_rate,
            chrom_list=chrom_list,
            RNA_data=self.RNA_data_p,
            ATAC_data=self.ATAC_data_p,
            other_data=self.other_data,
            mode=self.mode
        )
        print("Model successfully constructed.")

    def train_model(
        self,
        R_encoder_lr=0.001,
        A_encoder_lr=0.001,
        R_decoder_lr=0.001,
        A_decoder_lr=0.001,
        R_translator_lr=0.001,
        A_translator_lr=0.001,
        translator_lr=0.001,
        discriminator_lr=0.005,
        R2R_pretrain_epoch=100,
        A2A_pretrain_epoch=100,
        lock_encoder_and_decoder=False,
        translator_epoch=200,
        patience=50,
        batch_size=64,
        r_loss=nn.MSELoss(),
        a_loss=nn.BCELoss(),
        d_loss=nn.BCELoss(),
        loss_weight=[1, 2, 1],
        output_path=None,
        seed=52340,
        kl_mean=True,
        R_pretrain_kl_warmup=50,
        A_pretrain_kl_warmup=50,
        translation_kl_warmup=50,
        load_model=None,
    ):

        
        # Compute KL divergence weights
        R_kl_div = 1 / self.RNA_data_p.X.shape[1] * 20
        A_kl_div = 1 / self.ATAC_data_p.X.shape[1] * 20
        kl_div = R_kl_div + A_kl_div
        loss_weight.extend([R_kl_div, A_kl_div, kl_div])

        # Start training
        self.model.train(
            R_encoder_lr=R_encoder_lr,
            A_encoder_lr=A_encoder_lr,
            R_decoder_lr=R_decoder_lr,
            A_decoder_lr=A_decoder_lr,
            R_translator_lr=R_translator_lr,
            A_translator_lr=A_translator_lr,
            translator_lr=translator_lr,
            discriminator_lr=discriminator_lr,
            R2R_pretrain_epoch=R2R_pretrain_epoch,
            A2A_pretrain_epoch=A2A_pretrain_epoch,
            lock_encoder_and_decoder=lock_encoder_and_decoder,
            translator_epoch=translator_epoch,
            patience=patience,
            batch_size=batch_size,
            r_loss=r_loss,
            a_loss=a_loss,
            d_loss=d_loss,
            loss_weight=loss_weight,
            train_id_r=self.train_id_r,
            train_id_a=self.train_id_a,
            validation_id_r=self.validation_id_r, 
            validation_id_a=self.validation_id_a, 
            output_path=output_path,
            seed=seed,
            kl_mean=kl_mean,
            R_pretrain_kl_warmup=R_pretrain_kl_warmup,
            A_pretrain_kl_warmup=A_pretrain_kl_warmup,
            translation_kl_warmup=translation_kl_warmup,
            load_model=load_model,
        )
        print("Model training completed.")

    def test_model(
        self, 
        batch_size=64,
        model_path=None,
        load_model=False,
        output_path=None,
    ):
        
        predict = self.model.test(
            batch_size=batch_size,
            model_path=model_path,
            load_model=load_model,
            output_path=output_path,
            return_predict=True
        )
        return predict
    


