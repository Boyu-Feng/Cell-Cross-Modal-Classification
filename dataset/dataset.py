import pandas as pd
import torch
from torch.utils.data import Dataset
import scanpy as sc
import anndata as ad
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import issparse
import pickle
from collections import Counter

class GANDataset(Dataset):
    def __init__(self, sc_path, st_path):
        super(Dataset, self).__init__()
        self.sc_adata = sc.read_h5ad(sc_path)
        self.st_adata = sc.read_h5ad(st_path)
        sc.pp.log1p(self.st_adata)
        self.sc_adata.var.index = self.sc_adata.var.index.astype(str)
        self.st_adata.var.index = self.st_adata.var.index.astype(str)
        self.sc_adata.var_names_make_unique()
        self.st_adata.var_names_make_unique()
        genes = list(self.sc_adata.var_names.intersection(self.st_adata.var_names))
        self.sc_adata = self.sc_adata[:, genes].copy()
        self.st_adata = self.st_adata[:, genes].copy()
        print("Filtered shapes:", self.sc_adata.shape, self.st_adata.shape)
        sc.pp.scale(self.st_adata)
        sc.pp.scale(self.sc_adata)

        self.cell_type_c = self.sc_adata.obs['cell_type']
        self.cell_type_t = self.st_adata.obs['cell_type']
        self.cell_type = set(self.cell_type_c.unique()).intersection(set(self.cell_type_t.unique()))
        self.sc_adata = self.sc_adata[self.sc_adata.obs['cell_type'].isin(self.cell_type)]

        self.st_adata = self.st_adata[self.st_adata.obs['cell_type'].isin(self.cell_type)]

        self.cell_type_union = self.sc_adata.obs['cell_type']
        labels_count = Counter(self.cell_type_union)
        valid_label = {label for label, count in labels_count.items() if count >= 2}
        self.sc_adata = self.sc_adata[self.sc_adata.obs['cell_type'].isin(valid_label)]
        self.st_adata = self.st_adata[self.st_adata.obs['cell_type'].isin(valid_label)]
        self.cell_type_union = self.sc_adata.obs['cell_type']

        self.balance_data()
        print("new sc data:", self.sc_adata.X.shape)
        print("new st data:", self.st_adata.X.shape)
        print("union cell types:", self.cell_type_union.value_counts())

        if issparse(self.sc_adata.X):
            self.sc_data = self.sc_adata.X.toarray()
        else:
            self.sc_data = self.sc_adata.X
        self.cell_type_c = self.sc_adata.obs['cell_type']

        if issparse(self.st_adata.X):
            self.st_data = self.st_adata.X.toarray()
        else:
            self.st_data = self.st_adata.X

        # one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        # self.cell_type = one_hot_encoder.fit_transform(self.cell_type_union.values.reshape(-1, 1))
        # with open('one_hot_encoder.pkl', 'wb') as f:
        #     pickle.dump(one_hot_encoder, f)
        self.cell_type_dict = dict(zip(self.sc_adata.obs['cell_type'], self.sc_adata.obs['cell_type'].cat.codes))
        print(self.cell_type_dict)

        with open('label_dict.pkl', 'wb') as f:
            pickle.dump(self.cell_type_dict , f)

        self.cell_type = [self.cell_type_dict[celltype] for celltype in self.sc_adata.obs['cell_type']]
        self.cell_type_t = [self.cell_type_dict[celltype] for celltype in self.st_adata.obs['cell_type']]
        # self.type_dim = self.cell_type.shape[1]

        print("Cell type shape", len(self.cell_type))
        print("sc_data shape:", self.sc_data.shape)
        print("st_data shape:", self.st_data.shape)

        """""
        self.sc_adata = sc.read_h5ad(sc_path)
        self.st_adata = sc.read_h5ad(st_path)
        self.sc_data = self.sc_adata.X.toarray()
        self.st_data = self.st_adata.X
        print('sc_data.shape', self.sc_data.shape)
        print('st_data.shape', self.st_data.shape)
        print('Dataset loaded')
        """""

    def balance_data(self):
        max_len = max(len(self.sc_adata), len(self.st_adata))
        if len(self.st_adata) < max_len:
            indices = np.arange(len(self.st_adata))
            sampled_indices = np.random.choice(indices, max_len-len(self.st_adata), replace=False)
            additional_sampled = self.st_adata[sampled_indices]
            self.st_adata = sc.concat([self.st_adata, additional_sampled])

        if len(self.sc_adata) < max_len:
            indices = np.arange(len(self.sc_adata))
            sampled_indices = np.random.choice(indices, max_len - len(self.sc_adata), replace=False)
            additional_sampled = self.st_adata[sampled_indices]
            self.sc_adata = sc.concat([self.sc_adata, additional_sampled])

    def __len__(self):

        return self.sc_data.shape[0]

    def __getitem__(self, idx):

        sc_sample = self.sc_data[idx]
        st_sample = self.st_data[idx]
        cell_type = self.cell_type[idx]
        return sc_sample, st_sample, cell_type

    def Preprocess(self, sc_path, st_path, cca_dim=20):

        sc_adata = sc.read_h5ad(sc_path)
        st_adata = sc.read_h5ad(st_path)
        #single_cell = sc.read_h5ad(sc_path)
        #sc_adata = sc_adata[:512]
        #st_adata = st_adata[:512]
        sc.pp.log1p(st_adata)
        sc_adata.var.index = sc_adata.var.index.astype(str)
        st_adata.var.index = st_adata.var.index.astype(str)
        sc_adata.var_names_make_unique()
        st_adata.var_names_make_unique()

        genes = list(sc_adata.var_names.intersection(st_adata.var_names))
        sc_adata = sc_adata[:, genes].copy()
        st_adata = st_adata[:, genes].copy()
        print("Filtered shapes:", sc_adata.shape, st_adata.shape)

        sc.pp.scale(st_adata)
        sc.pp.scale(sc_adata)

        cca = CCA(n_components=cca_dim)
        cca.fit(sc_adata.X.T, st_adata.X.T)
        sc_data, st_data = cca.transform(sc_adata.X.T, st_adata.X.T)

        #correlations = [np.corrcoef(sc_data[:, i], st_data[:, i])[0, 1] for i in range(cca.n_components)]
        #print("correlations:", correlations)

        self.sc_data = np.dot(sc_adata.X, sc_data)
        self.st_data = np.dot(st_adata.X, st_data)
        self.cell_type_index = sc_adata.obs['cell_type']
        print("Cell type index_unique", self.cell_type_index.value_counts())
        print("cell_type_index", self.cell_type_index.shape)
        one_hot_encoder = OneHotEncoder(sparse=False)
        self.cell_type = one_hot_encoder.fit_transform(self.cell_type_index.values.reshape(-1, 1))
        print(self.cell_type.shape)
        self.type_dim = self.cell_type.shape[1]

        print("Cell type shape", self.cell_type.shape)
        print("sc_adata.X shape:", sc_adata.X.shape)
        print("sc_data shape:", self.sc_data.shape)
        print("st_adata.X shape:", st_adata.X.shape)
        print("st_data shape:", self.st_data.shape)

    def read_data(self, data_paths):
        pass

