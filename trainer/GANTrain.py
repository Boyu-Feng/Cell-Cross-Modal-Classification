"""""
Author: BoyuFeng
Email: fengboyu@genomics.cn
Date: 2024-12
Description:cell annotation classifier based on GAN

"""""

import torch
import torch.optim as optim
import torch.nn.functional as F
import hnswlib
from ganbo.model.CycleGAN import Model
from torch.utils.data import DataLoader
from ganbo.dataset.dataset import GANDataset
import matplotlib.pyplot as plt
import scanpy as sc
import argparse
import numpy as np

class GANTrainer(object):
    def __init__(self, batch_size=512, num_epochs=50, learning_rate=0.01,
                 sc_path=None, st_path=None, device="cuda"):
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.sc_path = sc_path
        self.st_path = st_path
        self.sc_adata = sc.read_h5ad(self.sc_path)
        self.st_adata = sc.read_h5ad(self.st_path)
        self.dataset = GANDataset(sc_path, st_path)
        self.sc_data = self.dataset.sc_data
        self.st_data = self.dataset.st_data
        self.feature_dim = self.sc_data.shape[1]
        self.cell_type = self.dataset.cell_type
        self.type_dim = len(set(self.cell_type))
        self.epsilon = 0.1
        self.real_labels = torch.ones(self.batch_size, 1).to(self.device) * (1-self.epsilon)
        self.fake_labels = torch.zeros(self.batch_size, 1).to(self.device) + self.epsilon
        self.model = Model(feature_dim=self.feature_dim, type_dim=self.type_dim)
        self.model = self.model.to(self.device)
        print("Model load")
        self.lam_l1 = 0.01
        self.lam = 0.3
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.gradient_penalty_weight = 10
        self.g_optimizer_t = optim.Adam(self.model.generator_t.parameters(), lr=self.learning_rate)
        self.d_optimizer_t = optim.Adam(self.model.discriminator_t.parameters(), lr=self.learning_rate)
        self.g_optimizer_c = optim.Adam(self.model.generator_c.parameters(), lr=self.learning_rate)
        self.d_optimizer_c = optim.Adam(self.model.discriminator_c.parameters(), lr=self.learning_rate)
        self.cla_optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)
        """""
        self.num_node = 16
        self.num_edge = 16
        self.node_features = torch.randn(num_nodes, 3)
        self.edge_index = torch.randint(0, num_nodes, (2, num_edges))
        """""

        print("Four optimizer loaded")

    def build_hnsw_index(self, data, space='l2', ef_construction=200, M=16):

        dim = data.shape[1]
        num_elements = data.shape[0]

        p = hnswlib.Index(space=space, dim=dim)
        p.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)

        data = data.cpu().numpy() if isinstance(data, torch.Tensor) else data

        p.add_items(data)

        p.set_ef(50)

        return p

    def get_nearest_neighbor_in_cca(self, sc_data, st_data, sc_hnsw_index, st_hnsw_index, k=1):
        sc_data = sc_data.detach().cpu().numpy() if isinstance(sc_data, torch.Tensor) else sc_data
        st_data = st_data.detach().cpu().numpy() if isinstance(st_data, torch.Tensor) else st_data

        sc_to_st_neighbors, _ = st_hnsw_index.knn_query(sc_data, k=k)

        st_to_sc_neighbors, _ = sc_hnsw_index.knn_query(st_data, k=k)

        return sc_to_st_neighbors, st_to_sc_neighbors

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, device=self.device)
        alpha = alpha.expand(real_samples.size(0), real_samples.size(1))
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones(d_interpolates.size(), requires_grad=False, device=self.device)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def intra_class_loss(self, sample, cell_type):
        """""
        args:sample: batch_size*cca_dim
        args:cell_type:after softmax, batch_size*type_dim
        """""
        loss = 0.0
        predict_class = torch.argmax(cell_type, dim=1)
        predict = torch.zeros_like(cell_type).scatter_(1, predict_class.unsqueeze(1), 1)
        for type in range(self.type_dim):
            class_indices = (predict_class == type).nonzero(as_tuple=True)
            if len(class_indices) > 1:
                class_features = sample[class_indices]
                difference = F.pairwise_distance(class_features[:, None, :], class_features[None, :, :])
                loss += difference.mean()
        return loss/self.type_dim

    def center_loss(self, sample, cell_type):
        """""
        args: sample: batch_size * cca_dim
        args: cell_type: after softmax, batch_size * type_dim
        """""
        predict_class = torch.argmax(cell_type, dim=1)
        total_difference = 0
        for type in range(self.type_dim):
            class_indices = (predict_class == type).nonzero(as_tuple=True)

            if len(class_indices[0]) > 0:
                class_features = sample[class_indices]
                center_point = torch.mean(class_features, dim=0)
                cos_similarity = F.cosine_similarity(class_features, center_point.unsqueeze(0).expand_as(class_features), dim=1)
                difference = 1 - cos_similarity
                total_difference += difference.mean()

        return total_difference

    def train(self):
        self.sc_data = torch.tensor(self.sc_data, dtype=torch.float32).to(self.device)
        self.st_data = torch.tensor(self.st_data, dtype=torch.float32).to(self.device)
        self.cell_type = torch.tensor(self.cell_type).to(self.device)
        """""
        train_indices, val_indices = train_test_split(
            np.arange(len(self.sc_data)),
            test_size=0.1,
            random_state=42,
            stratify=self.cell_type.cpu().numpy()
        )
        
        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)
        val_data = [val_dataset[i][0] for i in range(len(val_dataset))]
        val_labels = [val_dataset[i][1] for i in range(len(val_dataset))]
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        """""
        for epoch in range(self.num_epochs):
            print("epoch", epoch+1)
            for sc_data_batch, st_data_batch, cell_type_batch in self.dataloader:

                sc_data_batch = sc_data_batch.to(self.device).to(torch.float32)
                st_data_batch = st_data_batch.to(self.device).to(torch.float32)
                cell_type_batch = torch.tensor(cell_type_batch).to(self.device)

                sc_data_batch_numpy = sc_data_batch.cpu().numpy()
                st_data_batch_numpy = st_data_batch.cpu().numpy()

                hnsw_index_st = self.build_hnsw_index(st_data_batch_numpy)
                hnsw_index_sc = self.build_hnsw_index(sc_data_batch_numpy)

                generated_sample_t_to_c = self.model.generator_t(st_data_batch)  # 从 st 到 sc 的生成样本

                generated_sample_c_to_t = self.model.generator_c(sc_data_batch)  # 从 sc 到 st 的生成样本


                nearest_neighbors_st_to_sc, nearest_neighbors_sc_to_st = self.get_nearest_neighbor_in_cca(
                    generated_sample_t_to_c, generated_sample_c_to_t, hnsw_index_sc, hnsw_index_st
                )

                nearest_real_sample_sc_to_st = torch.tensor(st_data_batch_numpy[nearest_neighbors_sc_to_st].squeeze(1)).to(self.device)
                nearest_real_sample_st_to_sc = torch.tensor(sc_data_batch_numpy[nearest_neighbors_st_to_sc].squeeze(1)).to(self.device)

                nearest_real_sample_sc_to_st = nearest_real_sample_sc_to_st.to(torch.float32)
                nearest_real_sample_st_to_sc = nearest_real_sample_st_to_sc.to(torch.float32)

                self.cla_optimizer.zero_grad()

                pret_c = self.model.classifier(sc_data_batch)
                # data_t_to_c = self.model.generator_t(st_data_batch)
                # pret_t_to_c = self.model.classifier(data_t_to_c).detach() #对生成的假sc样本的细胞类型预测值 n_sample*type_dim
                # l1_loss = sum(torch.sum(torch.abs(param)) for param in self.model.classifier.parameters())
                class_loss = F.cross_entropy(pret_c, cell_type_batch)
                #class_loss_t_to_c = self.center_loss(st_data_batch, pret_t_to_c)
                # class_loss = class_loss_c + self.lam_l1 * l1_loss
                
                class_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.classifier.parameters(), max_norm=1.0)
                """""
                for name, param in self.model.classifier.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: {param.grad.norm().item()}")
                """""
                self.cla_optimizer.step()


                self.d_optimizer_t.zero_grad()

                real_pred_t = self.model.discriminator_t(nearest_real_sample_sc_to_st)
                fake_pred_t = self.model.discriminator_t(generated_sample_c_to_t.detach())

                gradient_penalty = self.compute_gradient_penalty(
                    self.model.discriminator_t,
                    nearest_real_sample_sc_to_st,
                    generated_sample_t_to_c.detach()
                )

                d_loss_t = (F.binary_cross_entropy(real_pred_t, self.real_labels)
                            + F.binary_cross_entropy(fake_pred_t, self.fake_labels)) + self.lam * gradient_penalty

                d_loss_t.backward()
                self.d_optimizer_t.step()

                for _ in range(2):
                    self.g_optimizer_c.zero_grad()
                    generated_sample_t_to_c = self.model.generator_t(st_data_batch)
                    fake_pred_c = self.model.discriminator_c(generated_sample_t_to_c)
                    g_loss_t = F.binary_cross_entropy(fake_pred_c, self.real_labels)
                    g_loss_t.backward(retain_graph=True)
                    self.g_optimizer_t.step()


                self.d_optimizer_c.zero_grad()
                real_pred_c = self.model.discriminator_c(nearest_real_sample_st_to_sc)
                fake_pred_c = self.model.discriminator_c(generated_sample_t_to_c.detach())

                gradient_penalty = self.compute_gradient_penalty(
                    self.model.discriminator_c,
                    nearest_real_sample_st_to_sc,
                    generated_sample_c_to_t.detach()
                )

                d_loss_c = (F.binary_cross_entropy(real_pred_c, self.real_labels)
                            + F.binary_cross_entropy(fake_pred_c, self.fake_labels)) + self.lam * gradient_penalty

                d_loss_c.backward()
                self.d_optimizer_c.step()

                for _ in range(2):
                    self.g_optimizer_t.zero_grad()
                    generated_sample_c_to_t = self.model.generator_c(sc_data_batch)
                    fake_pred_t = self.model.discriminator_t(generated_sample_c_to_t)
                    g_loss_c = F.binary_cross_entropy(fake_pred_t, self.real_labels)
                    g_loss_c.backward(retain_graph=True)
                    self.g_optimizer_c.step()

                del hnsw_index_st
                del hnsw_index_sc
                torch.cuda.empty_cache()

            print(
                f"Epoch: {epoch+1}/{self.num_epochs},classifier loss: {class_loss.item():4f}")

            """""
            t_to_c = self.model.generator_t(val_data)
            predict = torch.tensor(self.model.classifier(t_to_c))
            predict = torch.argmax(predict, dim=1).to(self.device)
            val_labels = torch.argmax(torch.tensor(val_labels), dim=1).to(self.device)
            correct = (predict == val_labels.to(self.device)).sum().item()
            accuracy = correct / len(val_labels)
            print(f"Accuracy:{accuracy}")
            """""

            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], D Loss (T): {d_loss_t.item():.4f}, G Loss (T): {g_loss_t.item():.4f}")
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], D Loss (C): {d_loss_c.item():.4f}, G Loss (C): {g_loss_c.item():.4f}")


        torch.save(self.model.discriminator_t.state_dict(),
                   '/home/share/huadjyin/home/s_huluni/fengboyu/bio_tools/CATree/sctGAN/model/discriminator_t.pth')
        torch.save(self.model.generator_t.state_dict(),
                   '/home/share/huadjyin/home/s_huluni/fengboyu/bio_tools/CATree/sctGAN/model/generator_t.pth')
        torch.save(self.model.discriminator_c.state_dict(),
                   '/home/share/huadjyin/home/s_huluni/fengboyu/bio_tools/CATree/sctGAN/model/discriminator_c.pth')
        torch.save(self.model.generator_c.state_dict(),
                   '/home/share/huadjyin/home/s_huluni/fengboyu/bio_tools/CATree/sctGAN/model/generator_c.pth')
        torch.save(self.model.classifier.state_dict(),
                   '/home/share/huadjyin/home/s_huluni/fengboyu/bio_tools/CATree/sctGAN/model/classifier.pth')

        print("All models saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Different type data.')
    parser.add_argument('--sc_dir', type=str, required=True, help='Directory containing the sc data files.')
    parser.add_argument('--st_dir', type=str, required=True, help='Directory containing the st data files..')

    args = parser.parse_args()

    sc_path = args.sc_dir
    st_path = args.st_dir

    Train = GANTrainer(sc_path=sc_path, st_path=st_path)
    Train.train()