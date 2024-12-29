import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels)
        )

    def forward(self, x):
        return x + self.layer(x)

class Generator(nn.Module):
    def __init__(self, feature_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(p=0.3)
        self.res_block1 = ResidualBlock(128)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, feature_dim)

        self.apply(init_weights)

    def forward(self, z):
        x = F.leaky_relu(self.bn1(self.fc1(z)))
        x = self.dropout1(x)
        x = self.res_block1(x)
        generated_sample = self.fc2(x)
        return generated_sample

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.res_block1 = ResidualBlock(128)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.res_block2 = ResidualBlock(32)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):

        x = self.bn1(self.fc1(x))
        x = F.leaky_relu(x)
        x = self.dropout1(x)
        x = self.res_block1(x)
        x = self.bn2(self.fc2(x))
        x = F.leaky_relu(x)
        x = self.dropout2(x)
        x = self.res_block2(x)
        return torch.sigmoid(self.fc3(x))

class Classifier(nn.Module):
    def __init__(self, input_dim, type_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.res_block1 = ResidualBlock(128)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128,64)
        self.bn2 = nn.BatchNorm1d(64)
        self.res_block2 = ResidualBlock(64)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(64, type_dim)

    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = F.relu(self.res_block1(x))
        x = self.dropout1(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(self.res_block2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class Model(nn.Module):
    def __init__(self, feature_dim, type_dim=None):
        super(Model, self).__init__()
        self.feature_dim = feature_dim
        self.type_dim = type_dim
        self.generator_t = Generator(feature_dim)
        self.discriminator_t = Discriminator(feature_dim)
        self.generator_c = Generator(feature_dim)
        self.discriminator_c = Discriminator(feature_dim)
        self.classifier = Classifier(feature_dim, type_dim)

    def forward(self, z):
        generated_sample_t = self.generator_t(z)
        generated_sample_c = self.generator_c(z)

        return generated_sample_t, generated_sample_c
