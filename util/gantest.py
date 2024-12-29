import scanpy as sc
import torch
from matplotlib import pyplot as plt

from ganbo.model.CycleGAN import Model
import argparse
from ganbo.dataset.dataset import GANDataset
import pickle
import numpy as np
from sklearn.metrics import silhouette_score
import umap
import pandas as pd

class Test(object):
    def __init__(self, sc_path, st_path, generator_c_path, generator_t_path, classifier_path, device="cuda"):
        self.sc_path = sc_path
        self.st_path = st_path
        self.device = device
        self.dataset = GANDataset(self.sc_path, self.st_path)
        self.sc_adata = self.dataset.sc_adata
        self.st_adata = self.dataset.st_adata
        self.sc_data = self.dataset.sc_data
        self.st_data = self.dataset.st_data
        self.feature_dim = self.sc_data.shape[1]
        self.cell_type = self.dataset.cell_type
        self.cell_type_t = self.dataset.cell_type_t
        print('cell type shape', len(self.cell_type))
        self.type_dim = len(set(self.cell_type))
        print('type_dim', self.type_dim)

        self.model = Model(feature_dim=self.feature_dim, type_dim=self.type_dim)
        self.model.generator_c = self.model.generator_c.to(self.device)
        self.model.generator_t = self.model.generator_t.to(self.device)
        self.model.classifier = self.model.classifier.to(self.device)
        self.model.generator_c.load_state_dict(torch.load(generator_c_path))
        self.model.generator_t.load_state_dict(torch.load(generator_t_path))
        self.model.classifier.load_state_dict(torch.load(classifier_path))
        print(self.model.generator_c)
        print("All model loaded")

    def test(self):
        self.model.generator_c.eval()
        self.model.generator_t.eval()
        self.model.classifier.eval()
        print("eval begin")
        with torch.no_grad():
            c_to_t = self.model.generator_c(torch.tensor(self.dataset.sc_data).to(self.device).to(torch.float32))
            t_to_c = self.model.generator_t(torch.tensor(self.dataset.st_data).to(self.device).to(torch.float32))
            c_original = self.dataset.sc_data
            t_original = self.dataset.st_data
            ori_data = np.concatenate([c_original, t_original])
            ori_label = [1] * c_original.shape[0] + [0] * t_original.shape[0]
            ori_sil = silhouette_score(ori_data, ori_label)
            print("ori_sil:", ori_sil)
            share_c_to_t = (self.dataset.sc_data + c_to_t.cpu().numpy()) / 2
            share_t_to_c = (self.dataset.st_data + t_to_c.cpu().numpy()) / 2
            c_data = np.concatenate([self.dataset.sc_data, t_to_c.cpu().numpy()])
            t_data = np.concatenate([self.dataset.st_data, c_to_t.cpu().numpy()])
            c_label = [1] * self.dataset.sc_data.shape[0] + [0] * t_to_c.cpu().numpy().shape[0]
            t_label = [1] * self.dataset.st_data.shape[0] + [0] * c_to_t.cpu().numpy().shape[0]
            c_sil_score = silhouette_score(c_data, c_label)
            t_sil_score = silhouette_score(t_data, t_label)
            print(f"c_sil_score:{c_sil_score} ,t_sil_score:{t_sil_score}" )
            sc_adata = self.sc_adata.copy()
            st_adata = self.st_adata.copy()
            sc_var_valid_idx = ~sc_adata.var.index.duplicated()
            st_var_valid_idx = ~st_adata.var.index.duplicated()

            sc_adata = sc_adata[:, sc_var_valid_idx]
            st_adata = st_adata[:, st_var_valid_idx]
            adata = sc.concat([self.sc_adata, self.st_adata])
            print(f"c_real.shape: {self.dataset.sc_data.shape}")
            print(f"t_real.shape: {self.dataset.st_data.shape}")
            print(f"c_fake.shape: {t_to_c.cpu().numpy().shape}")
            print(f"t_fake.shape: {c_to_t.cpu().numpy().shape}")
            """""
            all_data = np.vstack([self.dataset.sc_data, t_to_c.cpu().numpy(), self.dataset.st_data, c_to_t.cpu().numpy()])
            all_label = ['sc_real']*len(self.dataset.sc_data) + ['sc_fake']*len(t_to_c.cpu().numpy()) + ['st_real']*len(self.dataset.st_data) + ['st_fake']*len(c_to_t.cpu().numpy())
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            emb = reducer.fit_transform(all_data)
            print("emb finished")
            emb_df = pd.DataFrame(emb, columns=['umap1', 'umap2'])
            emb_df['label'] = all_label
            plt.figure(figsize=(16, 8))
            for label in set(all_label):
                subset = emb_df[emb_df["label"] == label]
                plt.scatter(subset["umap1"], subset["umap2"], label=label, alpha=0.7)
            plt.legend()
            plt.title("UMAP Visualization of Class Features")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.savefig('all_data_figures.pdf')
            print("UMAP Visualization of Class Features")

            adata.obs['batch'] = ["sc"] * len(sc_adata) + ["st"] * len(st_adata)
            adata.obsm['emb'] = np.concatenate([share_t_to_c, share_c_to_t])

            sc.pp.neighbors(adata, use_rep="emb")
            sc.tl.umap(adata)

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            sc.pl.umap(adata, color="cell_type", ax=axes[0], show=False)
            sc.pl.umap(adata, color="batch", ax=axes[1], show=False)

            plt.savefig("gan_res.pdf")
            print("result saved")
            """""
            predict = self.model.classifier(t_to_c)
            predict_class = torch.argmax(predict, dim=1)
            self.cell_type_t = torch.tensor(self.cell_type_t, dtype=torch.long).to(self.device)
            correct = (predict_class == self.cell_type_t).sum().item()
            accuracy = correct / len(self.cell_type_t)
            print(accuracy)

    def try1(self):
        label = torch.tensor(self.sc_data).to(self.device)
        predict = self.model.classifier(label)
        predict_class = torch.argmax(predict, dim=1)

        if not isinstance(self.cell_type, torch.Tensor):
            self.cell_type = torch.tensor(self.cell_type, dtype=torch.long).to(self.device)

        correct = (predict_class == self.cell_type).sum().item()
        accuracy = correct / len(self.cell_type)

        print(f"Test Accuracy: {accuracy:.2%}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Different type data.')
    parser.add_argument('--sc_dir', type=str, required=True, help='Directory containing the sc data files.')
    parser.add_argument('--st_dir', type=str, required=True, help='Directory containing the st data files..')
    parser.add_argument('--gc_dir', type=str, required=True, help='Directory containing the gc')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory containing the')
    parser.add_argument('--class_dir', type=str, required=True, help='Directory containing the')
    args = parser.parse_args()

    sc_path = args.sc_dir
    st_path = args.st_dir
    generator_c_path = args.gc_dir
    generator_t_path = args.gt_dir
    classifier_path = args.class_dir

    Test = Test(sc_path=sc_path, st_path=st_path, generator_c_path=generator_c_path,
                generator_t_path=generator_t_path, classifier_path=classifier_path)
    Test.test()
