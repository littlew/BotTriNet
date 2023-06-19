from util.data_io import read_dataset
from util.embedding_io import get_setence_Xy, get_account_Xy

from src.word_vector import get_word_embedding_dict
from src.avg_pooling import sentence_embedding_baseline
from src.triplet_network import TripletDataset
from src.triplet_network import TripletNet, TripletLoss, TripletEmbeddingNet
from src.trainer import fit
from src.evalate import predict

from torch.optim import lr_scheduler
import torch.optim as optim
import torch

import random

def random_split_instance_by_user_id(data, frac=0.3):
    user_ids = list(set(data['user_id'].values))
    n = len(user_ids)
    test_uids = random.sample(user_ids, int(n * frac))
    train_uids = [uid for uid in user_ids if uid not in test_uids]
    df_train = data.loc[data['user_id'].isin(train_uids)]
    df_test = data.loc[data['user_id'].isin(test_uids)]
    return df_train, df_test

def get_embedding_from_model(xs):
    return list(model.get_embedding(torch.Tensor(xs)).detach().numpy())

# Load tweet logs
files = [#'social_spambots_1.csv',
     'social_spambots_2.csv',
    #'social_spambots_3.csv',
     #'traditional_spambots_1.csv',
    #'fake_followers.csv',
    'genuine_accounts.csv']
data = read_dataset(files, path = 'dataset/cresci/sample')
df_train, df_test = random_split_instance_by_user_id(data, frac=0.3)

# Baseline Approach
print('[BaseLine]: Performance of Account Embedding')
# construct word embedding (raw)
word2vec = get_word_embedding_dict(df_train['text'].values, dim=16)
embedding_tools = {'word2vec': word2vec}
# construct sentence embedding (raw)
df_train['embedding'] = df_train['text'].apply(lambda t: sentence_embedding_baseline(t, embedding_tools))
df_test['embedding'] = df_test['text'].apply(lambda t: sentence_embedding_baseline(t, embedding_tools))
# construct account embedding (raw)
train_X, train_y = get_account_Xy(df_train, 'embedding')
test_X, test_y = get_account_Xy(df_test, 'embedding')
# evalute on test set
predict(train_X, train_y, test_X, test_y, algo='rf')


# BotTriNet Approach
print('[BotTriNet]: Performance of Account Embedding')
# Set up Triple Selector (via DataLoader Method in Pytorch)
cuda = torch.cuda.is_available()
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
triplet_train_dataset = TripletDataset(df_train) # Returns triplets of images
triplet_test_dataset = TripletDataset(df_test)
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset,
                                                   batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset,
                                                  batch_size=batch_size, shuffle=False, **kwargs)

# Set up Embedding Network
embedding_net = TripletEmbeddingNet(dim=16)
model = TripletNet(embedding_net)
if cuda: model.cuda()

# Set up  Loss function
loss_fn = TripletLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

# Training Triplet Network
n_epochs = 50
log_interval = 100
fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

# Get Predicted Embeddings on Triplet Network
df_train['triplet_embedding'] = [get_embedding_from_model(x) for x in df_train['embedding'].values]
df_test['triplet_embedding'] = [get_embedding_from_model(x) for x in df_test['embedding'].values]

# evaluate
train_X, train_y = get_account_Xy(df_train, 'triplet_embedding')
test_X, test_y = get_account_Xy(df_test, 'triplet_embedding')
predict(train_X, train_y, test_X, test_y, algo='rf')
