import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random


class TripletEmbeddingNet(nn.Module):
    def __init__(self, dim=16):
        super(TripletEmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(dim, dim*2)
        self.fc2 = nn.Linear(dim*2, dim)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return x

    def get_embedding(self, x):
        return self.forward(x)

class TripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """
    def __init__(self, df, key_name='user_id'):
        # user_id as group ip, using the average embedding as the anchor
        key2embedding = {}
        key2label = {}
        df_groups = df.groupby(key_name)
        for key, this_df in df_groups:
            embeddings = [list(x) for x in this_df['embedding'].values]
            embedding = list(np.mean(embeddings, axis=0)) # average 
            key2embedding[key] = embedding
            key2label[key] = this_df['label'].values[0]
        keys = [x for x in key2embedding.keys()]
        key_to_index = {keys[i]: i for i in range(len(keys))}
        index_to_key = {v: k for k, v in key_to_index.items()}
        self.key2embedding = key2embedding
        self.key2label = key2label
        self.index_to_key = index_to_key

        # random select positive samples and negative candidatea

        key2pos_embedding = {}
        key2neg_embedding = {}
        for key, emb in key2embedding.items():
            pos = list(df[df[key_name] == key]['embedding'].values)
            label = key2label[key]
            neg = list(df[df['label'] != label]['embedding'].values)
            neg_sample = random.sample(neg, 100) # random select 100 negative samples
            key2pos_embedding[key] = pos
            key2neg_embedding[key] = neg_sample
        self.key2pos_embedding = key2pos_embedding
        self.key2neg_embedding = key2neg_embedding

    def __getitem__(self, index):
        key = self.index_to_key[index]
        emb1 = torch.Tensor(self.key2embedding[key])
        emb2 = torch.Tensor(random.sample(self.key2pos_embedding[key], 1)[0])
        emb3 = torch.Tensor(random.sample(self.key2neg_embedding[key], 1)[0])
        return (emb1, emb2, emb3), [] 

    def __len__(self):
        return len(self.key2embedding)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  
        distance_negative = (anchor - negative).pow(2).sum(1)  
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()