import os
import random
import time
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GCNConv
from SpaTranslator.utils import *

##ignore warnings
import warnings
warnings.filterwarnings("ignore")


class RNA_Encoder(nn.Module):
    """
    RNA Encoder with a mix of MLP and GCN layers for feature extraction.
    """
    def __init__(self, nlayer, dim_list, act_list, dropout_rate, noise_rate):
        super(RNA_Encoder, self).__init__()
        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(nlayer):
            if i == nlayer - 1:  # Use GCN for the final layer
                self.layers.append(GCNConv(dim_list[i], dim_list[i + 1]))
            else:
                linear_layer = nn.Linear(dim_list[i], dim_list[i + 1])
                nn.init.xavier_uniform_(linear_layer.weight)
                self.layers.append(linear_layer)

            self.batch_norms.append(nn.BatchNorm1d(dim_list[i + 1]))
            self.activations.append(act_list[i])
            
            if i != nlayer - 1:  
                self.dropouts.append(nn.Dropout(dropout_rate))

    def forward(self, x, edge_index):
        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            x = self.layers[i](x, edge_index) if i == self.nlayer - 1 else self.layers[i](x)
            x = self.batch_norms[i](x)
            x = self.activations[i](x)
            if i != self.nlayer - 1:
                x = self.dropouts[i](x)
        return x


class RNA_Decoder(nn.Module):
    """
    RNA Decoder using only MLP layers for reconstruction.
    """
    def __init__(self, nlayer, dim_list, act_list, dropout_rate, noise_rate):
        super(RNA_Decoder, self).__init__()
        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(nlayer):
            linear_layer = nn.Linear(dim_list[i], dim_list[i + 1])
            nn.init.xavier_uniform_(linear_layer.weight)
            self.layers.append(linear_layer)

            self.batch_norms.append(nn.BatchNorm1d(dim_list[i + 1]))
            self.activations.append(act_list[i])

            if i != nlayer - 1:
                self.dropouts.append(nn.Dropout(dropout_rate))

    def forward(self, x, edge_index):
        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            x = self.activations[i](x)
            if i != self.nlayer - 1:
                x = self.dropouts[i](x)
        return x


class Split_Chrom_Encoder_block(nn.Module):
    def __init__(self, nlayer, dim_list, act_list, chrom_list, dropout_rate, noise_rate):
        """
        Encoder block that processes input data split by chromosome.

        Parameters
        ----------
        nlayer : int
            Number of layers.
        dim_list : list
            List of dimensions for each layer.
        act_list : list
            List of activation functions.
        chrom_list : list
            Number of features per chromosome.
        dropout_rate : float
            Dropout probability.
        noise_rate : float
            Noise dropout probability.
        """
        super(Split_Chrom_Encoder_block, self).__init__()

        self.nlayer = nlayer
        self.chrom_list = chrom_list
        self.noise_dropout = nn.Dropout(noise_rate)

        # Define layer lists
        self.linear_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(nlayer):
            if i == 0:
                self.linear_layers.append(nn.ModuleList())
                self.batch_norms.append(nn.ModuleList())
                self.activations.append(nn.ModuleList())
                self.dropouts.append(nn.ModuleList())

                for chrom_size in chrom_list:
                    layer = nn.Linear(chrom_size, dim_list[i + 1] // len(chrom_list))
                    nn.init.xavier_uniform_(layer.weight)
                    
                    self.linear_layers[i].append(layer)
                    self.batch_norms[i].append(nn.BatchNorm1d(dim_list[i + 1] // len(chrom_list)))
                    self.activations[i].append(act_list[i])
                    self.dropouts[i].append(nn.Dropout(dropout_rate))
            else:
                if i == nlayer - 1:  # Use GCN for the final layer
                    self.linear_layers.append(GCNConv(dim_list[i], dim_list[i + 1]))
                else:
                    layer = nn.Linear(dim_list[i], dim_list[i + 1])
                    nn.init.xavier_uniform_(layer.weight)
                    self.linear_layers.append(layer)

                self.batch_norms.append(nn.BatchNorm1d(dim_list[i + 1]))
                self.activations.append(act_list[i])
                if i < nlayer - 1:
                    self.dropouts.append(nn.Dropout(dropout_rate))

    def forward(self, x, edge_index):
        x = self.noise_dropout(x)

        for i in range(self.nlayer):
            if i == 0:
                x_splits = torch.split(x, self.chrom_list, dim=1)
                x = torch.concat([
                    self.dropouts[0][j](
                        self.activations[0][j](
                            self.batch_norms[0][j](
                                self.linear_layers[0][j](x_splits[j])
                            )
                        )
                    )
                    for j in range(len(self.chrom_list))
                ], dim=1)
            else:
                x = self.linear_layers[i](x, edge_index) if i == self.nlayer - 1 else self.linear_layers[i](x)
                x = self.batch_norms[i](x)
                x = self.activations[i](x)
                if i < self.nlayer - 1:
                    x = self.dropouts[i](x)

        return x


class Split_Chrom_Decoder_block(nn.Module):
    def __init__(self, nlayer, dim_list, act_list, chrom_list, dropout_rate, noise_rate):
        """
        Decoder block that reconstructs input data split by chromosome.

        Parameters
        ----------
        nlayer : int
            Number of layers.
        dim_list : list
            List of dimensions for each layer.
        act_list : list
            List of activation functions.
        chrom_list : list
            Number of features per chromosome.
        dropout_rate : float
            Dropout probability.
        noise_rate : float
            Noise dropout probability.
        """
        super(Split_Chrom_Decoder_block, self).__init__()

        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        self.chrom_list = chrom_list

        # Define layer lists
        self.linear_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(nlayer):
            if i < nlayer - 1:
                layer = nn.Linear(dim_list[i], dim_list[i + 1])
                nn.init.xavier_uniform_(layer.weight)

                self.linear_layers.append(layer)
                self.batch_norms.append(nn.BatchNorm1d(dim_list[i + 1]))
                self.activations.append(act_list[i])
                self.dropouts.append(nn.Dropout(dropout_rate))
            else:
                # Last layer processes each chromosome separately
                self.linear_layers.append(nn.ModuleList())
                self.batch_norms.append(nn.ModuleList())
                self.activations.append(nn.ModuleList())

                for chrom_size in chrom_list:
                    layer = nn.Linear(dim_list[i] // len(chrom_list), chrom_size)
                    nn.init.xavier_uniform_(layer.weight)

                    self.linear_layers[i].append(layer)
                    self.batch_norms[i].append(nn.BatchNorm1d(chrom_size))
                    self.activations[i].append(act_list[i])

    def forward(self, x, edge_index):
        x = self.noise_dropout(x)

        for i in range(self.nlayer):
            if i < self.nlayer - 1:
                x = self.linear_layers[i](x)
                x = self.batch_norms[i](x)
                x = self.activations[i](x)
                x = self.dropouts[i](x)
            else:
                x_splits = torch.chunk(x, len(self.chrom_list), dim=1)
                x = torch.concat([
                    self.activations[i][j](
                        self.batch_norms[i][j](
                            self.linear_layers[i][j](x_splits[j])
                        )
                    )
                    for j in range(len(self.chrom_list))
                ], dim=1)

        return x

class NetBlock(nn.Module):
    """
    Fully connected neural network block with batch normalization and dropout.
    """
    def __init__(self, nlayer, dim_list, act_list, dropout_rate, noise_rate):
        super(NetBlock, self).__init__()
        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(nlayer):
            linear_layer = nn.Linear(dim_list[i], dim_list[i + 1])
            nn.init.xavier_uniform_(linear_layer.weight)
            self.layers.append(linear_layer)

            self.batch_norms.append(nn.BatchNorm1d(dim_list[i + 1]))
            self.activations.append(act_list[i])

            if i != nlayer - 1:
                self.dropouts.append(nn.Dropout(dropout_rate))

    def forward(self, x):
        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            x = self.activations[i](x)
            if i != self.nlayer - 1:
                x = self.dropouts[i](x)
        return x




class Translator(nn.Module):
    def __init__(
        self,
        input_dim_rna: int, 
        input_dim_atac: int, 
        latent_dim: int, 
        activation_funcs: list, 
    ):
        """
        Translator module for mapping RNA and ATAC modalities.

        Parameters:
        -----------
        input_dim_rna : int
            Feature dimension from RNA modality.
        input_dim_atac : int
            Feature dimension from ATAC modality.
        latent_dim : int
            Shared latent space dimensionality.
        activation_funcs : list
            List of activation functions for mean, log variance, and decoder layers.
        """
        super(Translator, self).__init__()

        act_mu, act_logvar, act_dec = activation_funcs

        # --- RNA Encoder ---
        self.rna_mu_layer = nn.Linear(input_dim_rna, latent_dim)
        self.rna_mu_norm = nn.BatchNorm1d(latent_dim)
        self.rna_mu_act = act_mu

        self.rna_logvar_layer = nn.Linear(input_dim_rna, latent_dim)
        self.rna_logvar_norm = nn.BatchNorm1d(latent_dim)
        self.rna_logvar_act = act_logvar

        # --- ATAC Encoder ---
        self.atac_mu_layer = nn.Linear(input_dim_atac, latent_dim)
        self.atac_mu_norm = nn.BatchNorm1d(latent_dim)
        self.atac_mu_act = act_mu

        self.atac_logvar_layer = nn.Linear(input_dim_atac, latent_dim)
        self.atac_logvar_norm = nn.BatchNorm1d(latent_dim)
        self.atac_logvar_act = act_logvar

        # --- Decoders ---
        self.rna_decoder = self._create_decoder(latent_dim, input_dim_rna, act_dec)
        self.atac_decoder = self._create_decoder(latent_dim, input_dim_atac, act_dec)

        # Weight Initialization
        self._initialize_weights()

    def _create_decoder(self, latent_dim, output_dim, activation):
        """Creates a decoder module."""
        return nn.Sequential(
            nn.Linear(latent_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            activation
        )

    def _initialize_weights(self):
        """Applies Xavier uniform initialization to all linear layers."""
        for layer in [
            self.rna_mu_layer, self.rna_logvar_layer,
            self.atac_mu_layer, self.atac_logvar_layer,
            self.rna_decoder[0], self.atac_decoder[0]
        ]:
            nn.init.xavier_uniform_(layer.weight)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from latent space."""
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, mu_layer, mu_norm, mu_act, logvar_layer, logvar_norm, logvar_act, mode):
        """Encodes input into latent space using mean and log variance."""
        mu = mu_act(mu_norm(mu_layer(x)))
        logvar = logvar_act(logvar_norm(logvar_layer(x)))
        latent = mu if mode == 'test' else self.reparameterize(mu, logvar)
        return latent, mu, logvar

    def forward_rna(self, x, mode):
        """Processes RNA input and returns decoded outputs."""
        latent, mu, logvar = self.encode(
            x, self.rna_mu_layer, self.rna_mu_norm, self.rna_mu_act,
            self.rna_logvar_layer, self.rna_logvar_norm, self.rna_logvar_act, mode
        )
        return self.rna_decoder(latent), self.atac_decoder(latent), mu, logvar

    def forward_atac(self, x, mode):
        """Processes ATAC input and returns decoded outputs."""
        latent, mu, logvar = self.encode(
            x, self.atac_mu_layer, self.atac_mu_norm, self.atac_mu_act,
            self.atac_logvar_layer, self.atac_logvar_norm, self.atac_logvar_act, mode
        )
        return self.rna_decoder(latent), self.atac_decoder(latent), mu, logvar

    def forward(self, x, modality, mode='train'):
        """
        General forward pass function.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor.
        modality : str
            Either 'RNA' or 'ATAC'.
        mode : str
            'train' for sampling, 'test' for direct latent mean output.
        """
        if modality == 'RNA':
            return self.forward_rna(x, mode)
        elif modality == 'ATAC':
            return self.forward_atac(x, mode)
        else:
            raise ValueError("Invalid modality. Choose 'RNA' or 'ATAC'.")
        
    def train_model(self, x, input_type):
        return self.forward(x, input_type, 'train')
    
    def test_model(self, x, input_type):
        return self.forward(x, input_type, 'test')

class ModalityPretrainer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        activation_fns: list,
        dropout_rate: float = 0.1,
    ):
        """
        Pretraining module for a single modality (RNA or ATAC).

        Parameters:
        -----------
        input_dim : int
            Feature dimensionality.
        latent_dim : int
            Embedding dimensionality.
        activation_fns : list
            Activation functions for mean, variance, and decoder.
        dropout_rate : float, optional
            Dropout rate for regularization (default: 0.1).
        """
        super(ModalityPretrainer, self).__init__()

        act_mean, act_var, act_decoder = activation_fns

        # Encoder layers
        self.encoder_mean = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Dropout(dropout_rate),
            act_mean
        )
        self.encoder_var = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Dropout(dropout_rate),
            act_var
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            act_decoder
        )

        # Xavier initialization
        for module in [self.encoder_mean, self.encoder_var, self.decoder]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)

    def reparametrize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mean + noise * std

    def forward(self, x, mode='train'):
        mu = self.encoder_mean(x)
        logvar = self.encoder_var(x)
        latent = mu if mode == 'test' else self.reparametrize(mu, logvar)
        output = self.decoder(latent)
        return output, mu, logvar


class Model:
    def __init__(
        self,
        RNA_data,
        ATAC_data,
        other_data,
        mode,
        chrom_list: list,
        R_encoder_dim_list: list,
        A_encoder_dim_list: list,
        R_decoder_dim_list: list,
        A_decoder_dim_list: list,
        R_encoder_nlayer: int = 2,
        A_encoder_nlayer: int = 2,
        R_decoder_nlayer: int = 2,
        A_decoder_nlayer: int = 2,
        R_encoder_act_list: list = [nn.LeakyReLU(), nn.LeakyReLU()],
        A_encoder_act_list: list = [nn.LeakyReLU(), nn.LeakyReLU()],
        R_decoder_act_list: list = [nn.LeakyReLU(), nn.LeakyReLU()],
        A_decoder_act_list: list = [nn.LeakyReLU(), nn.Sigmoid()],
        translator_embed_dim: int = 128,
        translator_input_dim_rna: int = 128,
        translator_input_dim_atac: int = 128,
        translator_embed_act_list: list = [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()],
        discriminator_nlayer: int = 1,
        discriminator_dim_list_rna: list = [128],
        discriminator_dim_list_atac: list = [128],
        discriminator_act_list: list = [nn.Sigmoid()],
        dropout_rate: float = 0.1,
        R_noise_rate: float = 0.5,
        A_noise_rate: float = 0.3,
    ):
        # Print model parameters
        print("\n------------------------------")
        print("Model Parameters")
        print(f"Mode: {mode}")
        print(f"R_encoder_nlayer: {R_encoder_nlayer}")
        print(f"A_encoder_nlayer: {A_encoder_nlayer}")
        print(f"R_decoder_nlayer: {R_decoder_nlayer}")
        print(f"A_decoder_nlayer: {A_decoder_nlayer}")
        print(f"R_encoder_dim_list: {R_encoder_dim_list}")
        print(f"A_encoder_dim_list: {A_encoder_dim_list}")
        print(f"R_decoder_dim_list: {R_decoder_dim_list}")
        print(f"A_decoder_dim_list: {A_decoder_dim_list}")
        print(f"R_encoder_act_list: {R_encoder_act_list}")
        print(f"A_encoder_act_list: {A_encoder_act_list}")
        print(f"R_decoder_act_list: {R_decoder_act_list}")
        print(f"A_decoder_act_list: {A_decoder_act_list}")
        print(f"Translator embed dim: {translator_embed_dim}")
        print(f"Translator input dim (RNA): {translator_input_dim_rna}")
        print(f"Translator input dim (ATAC): {translator_input_dim_atac}")
        print(f"Translator activation list: {translator_embed_act_list}")
        print(f"Discriminator layers: {discriminator_nlayer}")
        print(f"Discriminator dim list (RNA): {discriminator_dim_list_rna}")
        print(f"Discriminator dim list (ATAC): {discriminator_dim_list_atac}")
        print(f"Discriminator activation list: {discriminator_act_list}")
        print(f"Dropout rate: {dropout_rate}")
        print(f"R_noise_rate: {R_noise_rate}")
        print(f"A_noise_rate: {A_noise_rate}")
        print(f"Chromosome list: {chrom_list}")
        print("------------------------------\n")

        
        # Define the mode
        self.mode = mode
        # Define encoders
        self.RNA_encoder = RNA_Encoder(
            nlayer=R_encoder_nlayer,
            dim_list=R_encoder_dim_list,
            act_list=R_encoder_act_list,
            dropout_rate=dropout_rate,
            noise_rate=R_noise_rate,
        )

        self.ATAC_encoder = Split_Chrom_Encoder_block(
            nlayer=A_encoder_nlayer,
            dim_list=A_encoder_dim_list,
            act_list=A_encoder_act_list,
            chrom_list=chrom_list,
            dropout_rate=dropout_rate,
            noise_rate=A_noise_rate,
        )

        # Define decoders
        self.RNA_decoder = RNA_Decoder(
            nlayer=R_decoder_nlayer,
            dim_list=R_decoder_dim_list,
            act_list=R_decoder_act_list,
            dropout_rate=dropout_rate,
            noise_rate=0,
        )

        self.ATAC_decoder = Split_Chrom_Decoder_block(
            nlayer=A_decoder_nlayer,
            dim_list=A_decoder_dim_list,
            act_list=A_decoder_act_list,
            chrom_list=chrom_list,
            dropout_rate=dropout_rate,
            noise_rate=0,
        )

        # Define translators
        self.R_translator = ModalityPretrainer(
            input_dim=translator_input_dim_rna,
            latent_dim=translator_embed_dim,
            activation_fns=translator_embed_act_list,
        )

        self.A_translator = ModalityPretrainer(
            input_dim=translator_input_dim_atac,
            latent_dim=translator_embed_dim,
            activation_fns=translator_embed_act_list,
        )

        self.translator = Translator(
            input_dim_rna=translator_input_dim_rna,
            input_dim_atac=translator_input_dim_atac,
            latent_dim=translator_embed_dim,
            activation_funcs=translator_embed_act_list,
        )

        # Define discriminators
        discriminator_dim_list_rna.append(1)
        discriminator_dim_list_atac.append(1)

        self.discriminator_R = NetBlock(
            nlayer=discriminator_nlayer,
            dim_list=discriminator_dim_list_rna,
            act_list=discriminator_act_list,
            dropout_rate=0,
            noise_rate=0,
        )

        self.discriminator_A = NetBlock(
            nlayer=discriminator_nlayer,
            dim_list=discriminator_dim_list_atac,
            act_list=discriminator_act_list,
            dropout_rate=0,
            noise_rate=0,
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            self.RNA_encoder = self.RNA_encoder.cuda()
            self.RNA_decoder = self.RNA_decoder.cuda()
            self.ATAC_encoder = self.ATAC_encoder.cuda()
            self.ATAC_decoder = self.ATAC_decoder.cuda()
            self.R_translator = self.R_translator.cuda()
            self.A_translator = self.A_translator.cuda()
            self.translator = self.translator.cuda()
            self.discriminator_R = self.discriminator_R.cuda()
            self.discriminator_A = self.discriminator_A.cuda()

        # Store data
        self.is_train_finished = False
        self.RNA_data_obs = RNA_data.obs
        self.ATAC_data_obs = ATAC_data.obs
        self.RNA_data_var = RNA_data.var
        self.ATAC_data_var = ATAC_data.var
        self.RNA_data = RNA_data.X.toarray()
        self.ATAC_data = ATAC_data.X.toarray()
        self.RNA_data_spatial = torch.tensor(RNA_data.uns["adj_matrix"].toarray(), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.ATAC_data_spatial = torch.tensor(ATAC_data.uns["adj_matrix"].toarray(), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

        if self.mode == "R2A":
            self.RNA_other_data = other_data.X.toarray()
            self.RNA_other_data_obs = other_data.obs
            self.RNA_other_data_var = other_data.var
            self.RNA_AlignedEmbedding = (
                torch.from_numpy(RNA_data.obsm["AlignedEmbedding"]).cuda().to(torch.float32)
            )
            self.RNA_other_AlignedEmbedding = (
                torch.from_numpy(other_data.obsm["AlignedEmbedding"]).cuda().to(torch.float32)
            )
            
        elif self.mode == "A2R":
            
            self.ATAC_other_data = other_data.X.toarray()
            self.ATAC_other_data_obs = other_data.obs
            self.ATAC_other_data_var = other_data.var
            self.ATAC_AlignedEmbedding = (
                torch.from_numpy(ATAC_data.obsm["AlignedEmbedding"]).cuda().to(torch.float32)
            )
            self.ATAC_other_AlignedEmbedding = (
                torch.from_numpy(other_data.obsm["AlignedEmbedding"]).cuda().to(torch.float32)
            )
        
    def set_train(self):
            """Set all model components to training mode."""
            self.RNA_encoder.train()
            self.RNA_decoder.train()
            self.ATAC_encoder.train()
            self.ATAC_decoder.train()
            self.R_translator.train()
            self.A_translator.train()
            self.translator.train()
            self.discriminator_R.train()
            self.discriminator_A.train()

    def set_eval(self):
            """Set all model components to evaluation mode."""
            self.RNA_encoder.eval()
            self.RNA_decoder.eval()
            self.ATAC_encoder.eval()
            self.ATAC_decoder.eval()
            self.R_translator.eval()
            self.A_translator.eval()
            self.translator.eval()
            self.discriminator_R.eval()
            self.discriminator_A.eval()

    
    def forward_RNA(self, rna_data, rna_graph, rec_loss_fn, kl_weight, mode):
            """Forward pass for RNA-to-RNA transformation."""
            latent_repr, mean_rna, log_var_rna = self.R_translator(self.RNA_encoder(rna_data, rna_graph), mode)
            reconstructed_rna = self.RNA_decoder(latent_repr, rna_graph)
            
            reconstruction_err = rec_loss_fn(reconstructed_rna, rna_data)
            kl_divergence = 0.5 * torch.mean(mean_rna.pow(2) + log_var_rna.exp() - log_var_rna - 1)
            
            total_loss = reconstruction_err + kl_weight * kl_divergence
            return total_loss, reconstruction_err, kl_divergence

    
    def forward_ATAC(self, atac_data, atac_graph, rec_loss_fn, kl_weight, mode):
            """Forward pass for ATAC-to-ATAC transformation."""
            latent_repr, mean_atac, log_var_atac = self.A_translator(self.ATAC_encoder(atac_data, atac_graph), mode)
            reconstructed_atac = self.ATAC_decoder(latent_repr, atac_graph)
            
            reconstruction_err = rec_loss_fn(reconstructed_atac, atac_data)
            kl_divergence = torch.mean(0.5 * (mean_atac.pow(2) + log_var_atac.exp() - log_var_atac - 1))

            total_loss = reconstruction_err + kl_weight * kl_divergence
            return total_loss, reconstruction_err, kl_divergence



    def forward_translation(self, batch_samples, RNA_graph, ATAC_graph, RNA_input_dim, ATAC_input_dim, a_loss, r_loss, loss_weight, forward_type, previous_embedding, kl_div_mean=False):
            """Forward pass through the translator model."""
            RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)
            
            if self.mode == "R2A":
                R2 = previous_embedding
                A2 = self.ATAC_encoder(ATAC_input, ATAC_graph)
            elif self.mode == "A2R":
                R2 = self.RNA_encoder(RNA_input, RNA_graph)
                A2 = previous_embedding

            if forward_type == 'train':
                R2R, R2A, mu_r, sigma_r = self.translator.train_model(R2, 'RNA')
                A2R, A2A, mu_a, sigma_a = self.translator.train_model(A2, 'ATAC')
            elif forward_type == 'test':
                R2R, R2A, mu_r, sigma_r = self.translator.test_model(R2, 'RNA')
                A2R, A2A, mu_a, sigma_a = self.translator.test_model(A2, 'ATAC')

            R2R = self.RNA_decoder(R2R, RNA_graph)
            R2A = self.ATAC_decoder(R2A, ATAC_graph)
            A2R = self.RNA_decoder(A2R, RNA_graph)
            A2A = self.ATAC_decoder(A2A, ATAC_graph)

            lossR2R = r_loss(R2R, RNA_input)
            lossA2R = r_loss(A2R, RNA_input)
            lossR2A = a_loss(R2A, ATAC_input)
            lossA2A = a_loss(A2A, ATAC_input)

            kl_div_r = -0.5 * torch.mean(1 + sigma_r - mu_r.pow(2) - sigma_r.exp()) if kl_div_mean else torch.clamp(-0.5 * torch.sum(1 + sigma_r - mu_r.pow(2) - sigma_r.exp()), 0, 10000)
            kl_div_a = -0.5 * torch.mean(1 + sigma_a - mu_a.pow(2) - sigma_a.exp()) if kl_div_mean else torch.clamp(-0.5 * torch.sum(1 + sigma_a - mu_a.pow(2) - sigma_a.exp()), 0, 10000)

            r_loss_w, a_loss_w, d_loss_w, kl_div_R, kl_div_A, kl_div_w = loss_weight
            reconstruct_loss = r_loss_w * (lossR2R + lossA2R) + a_loss_w * (lossR2A + lossA2A)
            kl_div = kl_div_r + kl_div_a
            loss_g = kl_div_w * kl_div + reconstruct_loss

            return reconstruct_loss, kl_div, loss_g

    

  
    
    def forward_discriminator(self, batch_data, rna_graph, atac_graph, dim_rna, dim_atac, disc_loss_fn, mode, prev_embedding):
            """Forward pass through the discriminator network."""
            
            # Split batch into RNA and ATAC components
            rna_data, atac_data = torch.split(batch_data, [dim_rna, dim_atac], dim=1)
            
            if self.mode == "R2A":
                # Retrieve previous embeddings
                rna_latent = prev_embedding
                atac_latent = self.ATAC_encoder(atac_data, atac_graph)
            elif self.mode == "A2R":
                rna_latent = self.RNA_encoder(rna_data, rna_graph)
                atac_latent = prev_embedding

            # Get translated versions
            if mode == "train":
                rna_self, rna_to_atac, _, _ = self.translator.train_model(rna_latent, "RNA")
                atac_to_rna, atac_self, _, _ = self.translator.train_model(atac_latent, "ATAC")
            else:
                rna_self, rna_to_atac, _, _ = self.translator.test_model(rna_latent, "RNA")
                atac_to_rna, atac_self, _, _ = self.translator.test_model(atac_latent, "ATAC")

            batch_size = batch_data.shape[0]

            # Generate pseudo-labels
            rand_values = np.random.rand(batch_size)
            pseudo_labels = np.where(rand_values > 0.7, 1.0, np.where(rand_values > 0.4, 0.5, 0.0))

            # Choose inputs for discrimination
            input_atac = torch.stack([atac_latent[i] if pseudo_labels[i] > 0.5 else rna_to_atac[i] for i in range(batch_size)], dim=0)
            input_rna = torch.stack([rna_latent[i] if pseudo_labels[i] > 0.5 else atac_to_rna[i] for i in range(batch_size)], dim=0)

            # Discriminator predictions
            pred_atac = self.discriminator_A(input_atac)
            pred_rna = self.discriminator_R(input_rna)

            # Compute loss
            loss_atac = disc_loss_fn(pred_atac.reshape(batch_size), torch.tensor(pseudo_labels).cuda().float())
            loss_rna = disc_loss_fn(pred_rna.reshape(batch_size), torch.tensor(pseudo_labels).cuda().float())

            return loss_atac + loss_rna


    def save_model_dict(self, output_path):
            """Save model parameters."""
            torch.save(self.RNA_encoder.state_dict(), f"{output_path}/model/RNA_encoder.pt")
            torch.save(self.ATAC_encoder.state_dict(), f"{output_path}/model/ATAC_encoder.pt")
            torch.save(self.RNA_decoder.state_dict(), f"{output_path}/model/RNA_decoder.pt")
            torch.save(self.ATAC_decoder.state_dict(), f"{output_path}/model/ATAC_decoder.pt")
            torch.save(self.R_translator.state_dict(), f"{output_path}/model/R_translator.pt")
            torch.save(self.A_translator.state_dict(), f"{output_path}/model/A_translator.pt")
            torch.save(self.translator.state_dict(), f"{output_path}/model/translator.pt")
            torch.save(self.discriminator_A.state_dict(), f"{output_path}/model/discriminator_A.pt")
            torch.save(self.discriminator_R.state_dict(), f"{output_path}/model/discriminator_R.pt")

    def dense_to_edge_index(self, dense_matrix):
            """Convert a dense adjacency matrix to an edge index."""
            if isinstance(dense_matrix, np.ndarray):
                dense_matrix = torch.tensor(dense_matrix)  # Convert NumPy array to PyTorch tensor
            return dense_matrix.to_sparse().indices()
        
    def collate_fn(self,batch):
            samples, r_ids, a_ids = zip(*batch)
            samples = torch.stack(samples)
            r_ids = torch.tensor(r_ids)
            a_ids = torch.tensor(a_ids)


            return samples, r_ids, a_ids
    
    def collate_fn_test(self,batch):
            samples, ids = zip(*batch)
            samples = torch.stack(samples)
            ids = torch.tensor(ids)
        

            return samples, ids
        
    def train(
            self,
            loss_weight: list,
            train_id_r: list,
            train_id_a: list,
            validation_id_r: list,
            validation_id_a: list,
            R_encoder_lr: float = 0.001,
            A_encoder_lr: float = 0.001,
            R_decoder_lr: float = 0.001,
            A_decoder_lr: float = 0.001,
            R_translator_lr: float = 0.001,
            A_translator_lr: float = 0.001,
            translator_lr: float = 0.001,
            discriminator_lr: float = 0.005,
            R2R_pretrain_epoch: int = 100,
            A2A_pretrain_epoch: int = 100,
            lock_encoder_and_decoder: bool = False,
            translator_epoch: int = 200,
            patience: int = 50,
            batch_size: int = 64,
            r_loss = nn.MSELoss(reduction="mean"),
            a_loss = nn.BCELoss(reduction="mean"),
            d_loss = nn.BCELoss(reduction="mean"),
            output_path: str = None,
            seed: int = 52340,
            kl_mean: bool = True,
            R_pretrain_kl_warmup: int = 50,
            A_pretrain_kl_warmup: int = 50,
            translation_kl_warmup: int = 50,
            model_path: str = ".",
            load_model: bool = False,
        ):

            self.is_train_finished = False

            if output_path is None:
                output_path = '.'

            if load_model:
                self.RNA_encoder.load_state_dict(torch.load(f"{model_path}/model/RNA_encoder.pt"))
                self.ATAC_encoder.load_state_dict(torch.load(f"{model_path}/model/ATAC_encoder.pt"))
                self.RNA_decoder.load_state_dict(torch.load(f"{model_path}/model/RNA_decoder.pt"))
                self.ATAC_decoder.load_state_dict(torch.load(f"{model_path}/model/ATAC_decoder.pt"))
                self.R_translator.load_state_dict(torch.load(f"{model_path}/model/R_translator.pt"))
                self.A_translator.load_state_dict(torch.load(f"{model_path}/model/A_translator.pt"))
                self.discriminator_A.load_state_dict(torch.load(f"{model_path}/model/discriminator_A.pt"))
                self.discriminator_R.load_state_dict(torch.load(f"{model_path}/model/discriminator_R.pt"))
                self.translator.load_state_dict(torch.load(f"{model_path}/model/translator.pt"))

            if seed is not None:
                setup_seed(seed)

            RNA_input_dim = self.RNA_data.shape[1]
            ATAC_input_dim = self.ATAC_data.shape[1]
            cell_count = len(train_id_r)

            self.train_dataset = PairedOmicsDataset(
                self.RNA_data, self.ATAC_data, self.RNA_data_spatial, self.ATAC_data_spatial, train_id_r, train_id_a
            )
            self.validation_dataset = PairedOmicsDataset(
                self.RNA_data, self.ATAC_data, self.RNA_data_spatial, self.ATAC_data_spatial, validation_id_r, validation_id_a
            )

            drop_last = cell_count % batch_size == 1
            self.train_dataloader = DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=drop_last, collate_fn=self.collate_fn
            )
            self.validation_dataloader = DataLoader(
                self.validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=self.collate_fn
            )

            self.optimizer_R_encoder = torch.optim.Adam(self.RNA_encoder.parameters(), lr=R_encoder_lr)
            self.optimizer_A_encoder = torch.optim.Adam(self.ATAC_encoder.parameters(), lr=A_encoder_lr, weight_decay=0)
            self.optimizer_R_decoder = torch.optim.Adam(self.RNA_decoder.parameters(), lr=R_decoder_lr)
            self.optimizer_A_decoder = torch.optim.Adam(self.ATAC_decoder.parameters(), lr=A_decoder_lr, weight_decay=0)
            self.optimizer_R_translator = torch.optim.Adam(self.R_translator.parameters(), lr=R_translator_lr)
            self.optimizer_A_translator = torch.optim.Adam(self.A_translator.parameters(), lr=A_translator_lr)
            self.optimizer_translator = torch.optim.Adam(self.translator.parameters(), lr=translator_lr)
            self.optimizer_discriminator_A = torch.optim.SGD(self.discriminator_A.parameters(), lr=discriminator_lr)
            self.optimizer_discriminator_R = torch.optim.SGD(self.discriminator_R.parameters(), lr=discriminator_lr)

            self.early_stopping_R2R = EarlyStopping(patience=patience, verbose=False)
            self.early_stopping_A2A = EarlyStopping(patience=patience, verbose=False)
            self.early_stopping_all = EarlyStopping(patience=patience, verbose=False)

            os.makedirs(output_path + "/model", exist_ok=True)

            """ pretrain for RNA and ATAC """
            pretrain_r_loss, pretrain_r_kl, pretrain_r_loss_val, pretrain_r_kl_val = [], [], [], []
            with tqdm(total = R2R_pretrain_epoch, ncols=100) as pbar:
                pbar.set_description('RNA pretraining with VAE')
                for epoch in range(R2R_pretrain_epoch):
                    pretrain_r_loss_, pretrain_r_kl_, pretrain_r_loss_val_, pretrain_r_kl_val_ = [], [], [], []
                    self.set_train()
                    for idx, (batch_samples,  r_ids, a_ids ) in enumerate(self.train_dataloader): 
                        
                        if torch.cuda.is_available():
                            batch_samples = batch_samples.cuda().to(torch.float32)
                        
                        RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)
                        RNA_graph = self.RNA_data_spatial[r_ids[:, None], r_ids]
                        ATAC_graph = self.ATAC_data_spatial[a_ids[:,None],a_ids]
                        RNA_graph = self.dense_to_edge_index(RNA_graph)
                        ATAC_graph = self.dense_to_edge_index(ATAC_graph)

                        RNA_graph,_ = add_self_loops(RNA_graph)
                        ATAC_graph,_ = add_self_loops(ATAC_graph)

                        if torch.cuda.is_available():
                            RNA_graph = RNA_graph.cuda()
                            ATAC_graph = ATAC_graph.cuda()
                        """ pretrain for RNA """
                        weight_temp = loss_weight.copy()
                        if epoch < R_pretrain_kl_warmup:
                            weight_temp[3] = loss_weight[3] * epoch / R_pretrain_kl_warmup

                        loss, reconstruct_loss, kl_div_r = self.forward_RNA(RNA_input, RNA_graph,r_loss, weight_temp[3], 'train')
                        self.optimizer_R_encoder.zero_grad()
                        self.optimizer_R_decoder.zero_grad()
                        self.optimizer_R_translator.zero_grad()
                        loss.backward()
                        self.optimizer_R_encoder.step()
                        self.optimizer_R_decoder.step()
                        self.optimizer_R_translator.step()

                        pretrain_r_loss_.append(reconstruct_loss.item())
                        pretrain_r_kl_.append(kl_div_r.item())

                    self.set_eval()
                    for idx, (batch_samples,  r_ids, a_ids )  in enumerate(self.validation_dataloader):

                        if torch.cuda.is_available():
                            batch_samples = batch_samples.cuda().to(torch.float32)

                        RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)

                        RNA_graph = self.RNA_data_spatial[r_ids[:, None], r_ids].cuda().to(torch.float32)
                        ATAC_graph = self.ATAC_data_spatial[a_ids[:,None],a_ids].cuda().to(torch.float32)

                        RNA_graph = self.dense_to_edge_index(RNA_graph)
                        ATAC_graph = self.dense_to_edge_index(ATAC_graph)

                        RNA_graph,_ = add_self_loops(RNA_graph)
                        ATAC_graph,_ = add_self_loops(ATAC_graph)

                        loss, reconstruct_loss, kl_div_r = self.forward_RNA(RNA_input, RNA_graph,r_loss, weight_temp[3], 'test')

                        pretrain_r_loss_val_.append(reconstruct_loss.item())
                        pretrain_r_kl_val_.append(kl_div_r.item())

                    pretrain_r_loss.append(np.mean(pretrain_r_loss_))
                    pretrain_r_kl.append(np.mean(pretrain_r_kl_))
                    pretrain_r_loss_val.append(np.mean(pretrain_r_loss_val_))
                    pretrain_r_kl_val.append(np.mean(pretrain_r_kl_val_))

                    self.early_stopping_R2R(np.mean(pretrain_r_loss_val_), self, output_path)
                    time.sleep(0.01)
                    pbar.update(1)
                    pbar.set_postfix(
                        train='{:.4f}'.format(np.mean(pretrain_r_loss_val_)), 
                        val='{:.4f}'.format(np.mean(pretrain_r_loss_)))
                    
                    if self.early_stopping_R2R.early_stop:
                        print('RNA pretraining early stop, validation loss does not improve in '+str(patience)+' epoches!')
                        self.RNA_encoder.load_state_dict(torch.load(output_path + '/model/RNA_encoder.pt'))
                        self.RNA_decoder.load_state_dict(torch.load(output_path + '/model/RNA_decoder.pt'))
                        self.R_translator.load_state_dict(torch.load(output_path + '/model/R_translator.pt'))
                        break
                
  
            pretrain_a_loss, pretrain_a_kl, pretrain_a_loss_val, pretrain_a_kl_val = [], [], [], []
            with tqdm(total = A2A_pretrain_epoch, ncols=100) as pbar:
                pbar.set_description('ATAC pretraining with VAE')
                for epoch in range(A2A_pretrain_epoch):
                    pretrain_a_loss_, pretrain_a_kl_, pretrain_a_loss_val_, pretrain_a_kl_val_ = [], [], [], []
                    self.set_train()
                    for idx, (batch_samples,  r_ids, a_ids ) in enumerate(self.train_dataloader): 

                        if torch.cuda.is_available():
                            batch_samples = batch_samples.cuda().to(torch.float32)

                        RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)


                        RNA_graph = self.RNA_data_spatial[r_ids[:, None], r_ids].cuda().to(torch.float32)
                        ATAC_graph = self.ATAC_data_spatial[a_ids[:,None],a_ids].cuda().to(torch.float32)


                        RNA_graph = self.dense_to_edge_index(RNA_graph)
                        ATAC_graph = self.dense_to_edge_index(ATAC_graph)

                        RNA_graph,_ = add_self_loops(RNA_graph)
                        ATAC_graph,_ = add_self_loops(ATAC_graph)

                        """ pretrain for ATAC """
                        weight_temp = loss_weight.copy()
                        if epoch < A_pretrain_kl_warmup:
                            weight_temp[4] = loss_weight[4] * epoch / A_pretrain_kl_warmup

                        loss, reconstruct_loss, kl_div_a = self.forward_ATAC(ATAC_input, ATAC_graph, a_loss, weight_temp[4], 'train')
                        self.optimizer_A_encoder.zero_grad()
                        self.optimizer_A_decoder.zero_grad()
                        self.optimizer_A_translator.zero_grad()
                        loss.backward()
                        self.optimizer_A_encoder.step()
                        self.optimizer_A_decoder.step()
                        self.optimizer_A_translator.step()

                        pretrain_a_loss_.append(reconstruct_loss.item())
                        pretrain_a_kl_.append(kl_div_a.item())

                    self.set_eval()
                    for idx, (batch_samples,  r_ids, a_ids ) in enumerate(self.validation_dataloader):

                        if torch.cuda.is_available():
                            batch_samples = batch_samples.cuda().to(torch.float32)

                        RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)

                        RNA_graph = self.RNA_data_spatial[r_ids[:, None], r_ids].cuda().to(torch.float32)
                        ATAC_graph = self.ATAC_data_spatial[a_ids[:,None],a_ids].cuda().to(torch.float32)


                        RNA_graph = self.dense_to_edge_index(RNA_graph)
                        ATAC_graph = self.dense_to_edge_index(ATAC_graph)

                        RNA_graph,_ = add_self_loops(RNA_graph)
                        ATAC_graph,_ = add_self_loops(ATAC_graph)

                        loss, reconstruct_loss, kl_div_a = self.forward_ATAC(ATAC_input, ATAC_graph,a_loss, weight_temp[4], 'test')

                        pretrain_a_loss_val_.append(reconstruct_loss.item())
                        pretrain_a_kl_val_.append(kl_div_a.item())

                    pretrain_a_loss.append(np.mean(pretrain_a_loss_))
                    pretrain_a_kl.append(np.mean(pretrain_a_kl_))
                    pretrain_a_loss_val.append(np.mean(pretrain_a_loss_val_))
                    pretrain_a_kl_val.append(np.mean(pretrain_a_kl_val_))

                    self.early_stopping_A2A(np.mean(pretrain_a_loss_val_), self, output_path)
                    time.sleep(0.01)
                    pbar.update(1)
                    pbar.set_postfix(
                        train='{:.4f}'.format(np.mean(pretrain_a_loss_val_)), 
                        val='{:.4f}'.format(np.mean(pretrain_a_loss_)))
                    
                    if self.early_stopping_A2A.early_stop:
                        print('ATAC pretraining early stop, validation loss does not improve in '+str(patience)+' epoches!')
                        self.ATAC_encoder.load_state_dict(torch.load(output_path + '/model/ATAC_encoder.pt'))
                        self.ATAC_decoder.load_state_dict(torch.load(output_path + '/model/ATAC_decoder.pt'))
                        self.A_translator.load_state_dict(torch.load(output_path + '/model/A_translator.pt'))
                        break
            
    
            """ train for translator and discriminator """
            train_loss, train_kl, train_discriminator, train_loss_val, train_kl_val, train_discriminator_val = [], [], [], [], [], []
            with tqdm(total = translator_epoch, ncols=100) as pbar:
                pbar.set_description('Integrative training')
                for epoch in range(translator_epoch):
                    train_loss_, train_kl_, train_discriminator_, train_loss_val_, train_kl_val_, train_discriminator_val_ = [], [], [], [], [], []
                    self.set_train()
                    for idx, (batch_samples,  r_ids, a_ids ) in enumerate(self.train_dataloader):

                        if torch.cuda.is_available():
                            batch_samples = batch_samples.cuda().to(torch.float32)

                        RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)

                        RNA_graph = self.RNA_data_spatial[r_ids[:, None], r_ids].cuda().to(torch.float32)
                        ATAC_graph = self.ATAC_data_spatial[a_ids[:,None],a_ids].cuda().to(torch.float32)

                        RNA_graph = self.dense_to_edge_index(RNA_graph)
                        ATAC_graph = self.dense_to_edge_index(ATAC_graph)

                        RNA_graph,_ = add_self_loops(RNA_graph)
                        ATAC_graph,_ = add_self_loops(ATAC_graph)
                        if self.mode == "R2A":
                            RNA_AlignedEmbedding = self.RNA_AlignedEmbedding[r_ids,:]

                            """ train for discriminator """
                            loss_d = self.forward_discriminator(batch_samples, RNA_graph,ATAC_graph,RNA_input_dim, ATAC_input_dim, d_loss, 'train',RNA_AlignedEmbedding)
                            self.optimizer_discriminator_R.zero_grad()
                            self.optimizer_discriminator_A.zero_grad()
                            loss_d.backward()
                            self.optimizer_discriminator_R.step()
                            self.optimizer_discriminator_A.step()

                            """ train for generator """
                            weight_temp = loss_weight.copy()
                            if epoch < translation_kl_warmup:
                                weight_temp[5] = loss_weight[5] * epoch / translation_kl_warmup
                            loss_d = self.forward_discriminator(batch_samples, RNA_graph,ATAC_graph,RNA_input_dim, ATAC_input_dim, d_loss, 'train',RNA_AlignedEmbedding)
                            reconstruct_loss, kl_div, loss_g = self.forward_translation(batch_samples, RNA_graph,ATAC_graph,RNA_input_dim, ATAC_input_dim, a_loss, r_loss, weight_temp, 'train', RNA_AlignedEmbedding,kl_mean)
                        if self.mode == "A2R":
                            ATAC_AlignedEmbedding = self.ATAC_AlignedEmbedding[a_ids,:]
                            
                            """ train for discriminator """
                            loss_d = self.forward_discriminator(batch_samples, RNA_graph,ATAC_graph,RNA_input_dim, ATAC_input_dim, d_loss, 'train',ATAC_AlignedEmbedding)
                            self.optimizer_discriminator_R.zero_grad()
                            self.optimizer_discriminator_A.zero_grad()
                            loss_d.backward()
                            self.optimizer_discriminator_R.step()
                            self.optimizer_discriminator_A.step()
                            
                            
                            """ train for generator """
                            weight_temp = loss_weight.copy()
                            if epoch < translation_kl_warmup:
                                weight_temp[5] = loss_weight[5] * epoch / translation_kl_warmup
                            loss_d = self.forward_discriminator(batch_samples, RNA_graph,ATAC_graph,RNA_input_dim, ATAC_input_dim, d_loss, 'train',ATAC_AlignedEmbedding)
                            reconstruct_loss, kl_div, loss_g = self.forward_translation(batch_samples, RNA_graph,ATAC_graph,RNA_input_dim, ATAC_input_dim, a_loss, r_loss, weight_temp, 'train', ATAC_AlignedEmbedding,kl_mean)
                        
                        
                        if loss_d.item() < 1.35:
                            loss_g -= loss_weight[2] * loss_d

                        self.optimizer_translator.zero_grad()
                        if not lock_encoder_and_decoder:
                            self.optimizer_R_encoder.zero_grad()
                            self.optimizer_A_encoder.zero_grad()
                            self.optimizer_R_decoder.zero_grad()
                            self.optimizer_A_decoder.zero_grad()
                        loss_g.backward()
                        self.optimizer_translator.step()
                        if not lock_encoder_and_decoder:
                            self.optimizer_R_encoder.step()
                            self.optimizer_A_encoder.step()
                            self.optimizer_R_decoder.step()
                            self.optimizer_A_decoder.step()

                        train_loss_.append(reconstruct_loss.item())
                        train_kl_.append(kl_div.item()) 
                        train_discriminator_.append(loss_d.item())

                    self.set_eval()
                    for idx, (batch_samples,  r_ids, a_ids ) in enumerate(self.validation_dataloader):

                        if torch.cuda.is_available():
                            batch_samples = batch_samples.cuda().to(torch.float32)                    

                        RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)

                        RNA_graph = self.RNA_data_spatial[r_ids[:, None], r_ids].cuda().to(torch.float32)
                        ATAC_graph = self.ATAC_data_spatial[a_ids[:,None],a_ids].cuda().to(torch.float32)


                        RNA_graph = self.dense_to_edge_index(RNA_graph)
                        ATAC_graph = self.dense_to_edge_index(ATAC_graph)

                        RNA_graph,_ = add_self_loops(RNA_graph)
                        ATAC_graph,_ = add_self_loops(ATAC_graph)
                        if self.mode == "R2A":
                            RNA_AlignedEmbedding = self.RNA_AlignedEmbedding[r_ids,:]

                            """ test for discriminator """
                            loss_d = self.forward_discriminator(batch_samples,RNA_graph,ATAC_graph, RNA_input_dim, ATAC_input_dim, d_loss, 'test',RNA_AlignedEmbedding)

                            """ test for generator """
                            loss_d = self.forward_discriminator(batch_samples,RNA_graph,ATAC_graph, RNA_input_dim, ATAC_input_dim, d_loss, 'test',RNA_AlignedEmbedding)
                            reconstruct_loss, kl_div, loss_g = self.forward_translation(batch_samples,RNA_graph,ATAC_graph, RNA_input_dim, ATAC_input_dim, a_loss, r_loss, weight_temp, 'train',RNA_AlignedEmbedding, kl_mean)
                            loss_g -= loss_weight[2] * loss_d

                            train_loss_val_.append(reconstruct_loss.item())
                            train_kl_val_.append(kl_div.item()) 
                            train_discriminator_val_.append(loss_d.item())
                        if self.mode == "A2R":
                            ATAC_AlignedEmbedding = self.ATAC_AlignedEmbedding[a_ids,:]

                            """ test for discriminator """
                            loss_d = self.forward_discriminator(batch_samples,RNA_graph,ATAC_graph, RNA_input_dim, ATAC_input_dim, d_loss, 'test',ATAC_AlignedEmbedding)

                            """ test for generator """
                            loss_d = self.forward_discriminator(batch_samples,RNA_graph,ATAC_graph, RNA_input_dim, ATAC_input_dim, d_loss, 'test',ATAC_AlignedEmbedding)
                            reconstruct_loss, kl_div, loss_g = self.forward_translation(batch_samples,RNA_graph,ATAC_graph, RNA_input_dim, ATAC_input_dim, a_loss, r_loss, weight_temp, 'train',ATAC_AlignedEmbedding, kl_mean)
                            loss_g -= loss_weight[2] * loss_d

                            train_loss_val_.append(reconstruct_loss.item())
                            train_kl_val_.append(kl_div.item()) 
                            train_discriminator_val_.append(loss_d.item())

                    train_loss.append(np.mean(train_loss_))
                    train_kl.append(np.mean(train_kl_))
                    train_discriminator.append(np.mean(train_discriminator_))
                    train_loss_val.append(np.mean(train_loss_val_))
                    train_kl_val.append(np.mean(train_kl_val_))
                    train_discriminator_val.append(np.mean(train_discriminator_val_))
                    self.early_stopping_all(np.mean(train_loss_val_), self, output_path)
                    
                    time.sleep(0.01)
                    pbar.update(1)
                    pbar.set_postfix(
                        train='{:.4f}'.format(np.mean(train_loss_val_)), 
                        val='{:.4f}'.format(np.mean(train_loss_)))
                    
                    if self.early_stopping_all.early_stop:
                        print('Integrative training early stop, validation loss does not improve in '+str(patience)+' epoches!')
                        self.RNA_encoder.load_state_dict(torch.load(output_path + '/model/RNA_encoder.pt'))
                        self.ATAC_encoder.load_state_dict(torch.load(output_path + '/model/ATAC_encoder.pt'))
                        self.RNA_decoder.load_state_dict(torch.load(output_path + '/model/RNA_decoder.pt'))
                        self.ATAC_decoder.load_state_dict(torch.load(output_path + '/model/ATAC_decoder.pt'))
                        self.translator.load_state_dict(torch.load(output_path + '/model/translator.pt'))
                        self.discriminator_A.load_state_dict(torch.load(output_path + '/model/discriminator_A.pt'))
                        self.discriminator_R.load_state_dict(torch.load(output_path + '/model/discriminator_R.pt'))
                        break
            
            self.save_model_dict(output_path)
                
            self.is_train_finished = True
                
    def test(
        self,
        batch_size: int,
        model_path: str = ".",
        load_model: bool = False,
        output_path: str = None,
        return_predict: bool = False
    ):
    
        # Set default output path if not provided
        if output_path is None:
            output_path = '.'
        
        # Load the model if required
        if load_model:
            self.RNA_encoder.load_state_dict(torch.load(f"{model_path}/model/RNA_encoder.pt"))
            self.ATAC_encoder.load_state_dict(torch.load(f"{model_path}/model/ATAC_encoder.pt"))
            self.RNA_decoder.load_state_dict(torch.load(f"{model_path}/model/RNA_decoder.pt"))
            self.ATAC_decoder.load_state_dict(torch.load(f"{model_path}/model/ATAC_decoder.pt"))
            self.R_translator.load_state_dict(torch.load(f"{model_path}/model/R_translator.pt"))
            self.A_translator.load_state_dict(torch.load(f"{model_path}/model/A_translator.pt"))
            self.discriminator_A.load_state_dict(torch.load(f"{model_path}/model/discriminator_A.pt"))
            self.discriminator_R.load_state_dict(torch.load(f"{model_path}/model/discriminator_R.pt"))
            self.translator.load_state_dict(torch.load(f"{model_path}/model/translator.pt"))
            
        if self.mode == "R2A":
            # Create test dataset and dataloader
            self.R_test_dataset = Single_omics_dataset(self.RNA_other_data)
            self.R_test_dataloader = DataLoader(self.R_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=self.collate_fn_test)

            self.set_eval()

            # Prepare for storing the predictions
            R2A_embedding, R2A_predict = [], []

            # Iterate over test dataset to make predictions
            with torch.no_grad():
                with tqdm(total=len(self.R_test_dataloader), ncols=100) as pbar:
                    pbar.set_description('RNA to ATAC predicting...')
                    for idx, (batch_samples, ids) in enumerate(self.R_test_dataloader):
                        if torch.cuda.is_available():
                            batch_samples = batch_samples.cuda().to(torch.float32)
                        ATAC_graph = 1
                        # Prepare inputs for the translator model
                        RNA_other_AlignedEmbedding = self.RNA_other_AlignedEmbedding[ids, :]
                        R2 = RNA_other_AlignedEmbedding

                        # Make predictions using the model
                        R2R, R2A, mu_r, sigma_r = self.translator.test_model(R2, 'RNA')
                        R2A_final = self.ATAC_decoder(R2A, ATAC_graph)

                        # Store results
                        R2A_embedding.append(R2A.cpu())
                        R2A_predict.append(R2A_final.cpu())
                        
                        time.sleep(0.01)
                        pbar.update(1)

            # Convert predictions to AnnData format
            R2A_predict = convert_tensor_to_anndata(R2A_predict)
            R2A_predict.obs = self.RNA_other_data_obs
            R2A_predict.var = self.ATAC_data_var
            R2A_predict.uns['R2A_embedding'] = R2A_embedding

            # Return predictions if requested
            if return_predict:
                return R2A_predict
        
        if self.mode == "A2R":
            # Create test dataset and dataloader
            self.A_test_dataset = Single_omics_dataset(self.ATAC_other_data)
            self.A_test_dataloader = DataLoader(self.A_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=self.collate_fn_test)

            self.set_eval()

            # Prepare for storing the predictions
            A2R_embedding, A2R_predict = [], []

            # Iterate over test dataset to make predictions
            with torch.no_grad():
                with tqdm(total=len(self.A_test_dataloader), ncols=100) as pbar:
                    pbar.set_description('ATAC to RNA predicting...')
                    for idx, (batch_samples, ids) in enumerate(self.A_test_dataloader):
                        if torch.cuda.is_available():
                            batch_samples = batch_samples.cuda().to(torch.float32)
                        RNA_graph = 1
                        # Prepare inputs for the translator model
                        ATAC_other_AlignedEmbedding = self.ATAC_other_AlignedEmbedding[ids, :]
                        A2 = ATAC_other_AlignedEmbedding

                        # Make predictions using the model
                        A2R, A2A, mu_a, sigma_a = self.translator.test_model(A2, 'ATAC')
                        A2R_final = self.RNA_decoder(A2R, RNA_graph)

                        # Store results
                        A2R_embedding.append(A2R.cpu())
                        A2R_predict.append(A2R_final.cpu())
                        
                        time.sleep(0.01)
                        pbar.update(1)

            # Convert predictions to AnnData format
            A2R_predict = convert_tensor_to_anndata(A2R_predict)
            A2R_predict.obs = self.ATAC_other_data_obs
            A2R_predict.var = self.RNA_data_var
            A2R_predict.uns['A2R_embedding'] = A2R_embedding

            # Return predictions if requested
            if return_predict:
                return A2R_predict



