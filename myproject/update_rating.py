import sqlite3
from sqlite3 import Error
import pandas as pd
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from box import Box
from copy import deepcopy
import warnings

# warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=True)


def connection():
    try:
        con = sqlite3.connect('db.sqlite3')
        print("db connection success")
        return con
    except Error:
        print(Error)

def get_valid_user(df):
    usr_counts = df['log_time'].value_counts()
    usr_list = usr_counts[usr_counts >= 10].index.tolist()
    zip_data_df = df[df['log_time'].isin(usr_list)]
    return zip_data_df

def update_user_rating(valid_user_df):
    valid_user_df = valid_user_df.drop('user', axis=1)
    valid_user_df.rename(columns={'log_time': 'user'}, inplace=True)
    og_df = pd.read_csv("user_rating_10.csv", encoding="euc-kr")
    og_user_list = list(og_df['user'].unique())

    tobedeleted = []
    for idx,row in valid_user_df.iterrows():
        if row['user'] in og_user_list:
            print(row['user'],idx,"@@@")
            tobedeleted.append(1)
        else:
            tobedeleted.append(0)
    valid_user_df['checker'] = tobedeleted
    valid_user_df = valid_user_df.drop(valid_user_df[valid_user_df.checker == 1].index)
    valid_user_df = valid_user_df.drop('checker', axis=1)
    print(valid_user_df)
    frames = [og_df, valid_user_df]
    result_df = pd.concat(frames)
    result_df.to_csv("user_rating_10.csv",encoding="euc-kr",index=False)

#model_update


config = {
    'data_path': "/content/drive/MyDrive/NaverWebtoon/data",

    'submission_path': "/content/drive/MyDrive/NaverWebtoon/submission",
    'submission_name': 'RecVAE_v2_submission.csv',

    'model_path': "/content/drive/MyDrive/NaverWebtoon/model",
    'model_name': 'oof0_RecVAEv3.pt',

    'hidden_dim': 600,
    'latent_dim': 200,
    'dropout_rate': 0.7,
    'gamma': 0.0005,
    'beta': None,
    'not_alternating': True,
    'e_num_epochs': 2,
    'd_num_epochs': 1,

    'lr': 5e-4,
    'batch_size': 500,
    'num_epochs': 200,
    'num_workers': 2,

    'valid_samples': 5,
    'seed': 22,
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MakeMatrixDataSet():

    def __init__(self, config):
        self.config = config
        self.df = pd.read_csv(('user_rating_10.csv'), encoding="euc-kr")
        self.df = self.df.rename(columns={'title': 'item'})
        self.item_encoder, self.item_decoder = self.generate_encoder_decoder('item')
        self.user_encoder, self.user_decoder = self.generate_encoder_decoder('user')
        self.num_item, self.num_user = len(self.item_encoder), len(self.user_encoder)

        self.df['item_idx'] = self.df['item'].apply(lambda x: self.item_encoder[x])
        self.df['user_idx'] = self.df['user'].apply(lambda x: self.user_encoder[x])

        self.user_train, self.user_valid = self.generate_sequence_data()

    def generate_encoder_decoder(self, col: str):


        encoder = {}
        decoder = {}
        ids = self.df[col].unique()

        for idx, _id in enumerate(ids):
            encoder[_id] = idx
            decoder[idx] = _id

        return encoder, decoder

    def generate_sequence_data(self):

        users = defaultdict(list)
        user_train = {}
        user_valid = {}
        for user, item in zip(self.df['user_idx'], self.df['item_idx']):
            users[user].append(item)

        for user in users:
            np.random.seed(config['seed'])

            user_total = users[user]
            valid = np.random.choice(user_total, size=self.config['valid_samples']//2, replace=False).tolist()
            train = list(set(user_total) - set(valid))

            user_train[user] = train
            user_valid[user] = valid

        return user_train, user_valid

    def get_train_valid_data(self):
        return self.user_train, self.user_valid

    def oof_make_dataset(self, seed):
        user_train = {}
        user_valid = {}

        users = defaultdict(list)
        group_df = self.df.groupby('user_idx')
        for user, items in group_df:
            users[user].extend(items['item_idx'].tolist())

        for user in users:
            np.random.seed(seed)

            user_total = users[user]
            valid = np.random.choice(user_total, size=self.config['valid_samples'], replace=False//2).tolist()
            train = list(set(user_total) - set(valid))

            user_train[user] = train
            user_valid[user] = valid

        return user_train, user_valid

    def make_matrix(self, user_list, user_train, user_valid, train=True):

        mat = torch.zeros(size=(user_list.size(0), self.num_item))
        for idx, user in enumerate(user_list):
            if train:
                # print("train",train)
                mat[idx, user_train[user.item()]] = 1
            else:
                mat[idx, user_train[user.item()] + user_valid[user.item()]] = 1
        return mat
class AEDataSet(Dataset):
    def __init__(self, num_user):
        self.num_user = num_user
        self.users = [i for i in range(num_user)]

    def __len__(self):
        return self.num_user

    def __getitem__(self, idx):
        user = self.users[idx]
        return torch.LongTensor([user])


def swish(x):
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3 / 20, 3 / 4, 1 / 10]):
        super(CompositePrior, self).__init__()

        self.mixture_weights = mixture_weights

        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)

        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)

        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)

        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)

    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, dropout_rate=0)

        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)

        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]

        density_per_gaussian = torch.stack(gaussians, dim=-1)

        return torch.logsumexp(density_per_gaussian, dim=-1)


class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]

        x = F.dropout(x, p=dropout_rate, training=self.training)

        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)


class RecVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=600, latent_dim=200):
        super(RecVAE, self).__init__()

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, user_ratings, beta=None, gamma=0.005, dropout_rate=0.5, calculate_loss=True):
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)

        if calculate_loss:
            if gamma:
                norm = user_ratings.sum(dim=-1)
                kl_weight = gamma * norm
            elif beta:
                kl_weight = beta

            mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
            negative_elbo = -(mll - kld)

            return (mll, kld), negative_elbo

        else:
            return x_pred

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))


def train(model, optimizer, data_loader, user_train, user_valid, make_matrix_data_set, beta, gamma, dropout_rate):
    model.train()
    loss_val = 0
    for users in data_loader:
        mat = make_matrix_data_set.make_matrix(user_list=users, user_train=user_train, user_valid=user_valid,
                                               train=True)
        mat = mat.to(device)
        _, loss = model(user_ratings=mat, beta=beta, gamma=gamma, dropout_rate=dropout_rate)

        optimizer.zero_grad()
        loss_val += loss.item()
        loss.backward()
        optimizer.step()

    loss_val /= len(data_loader)

    return loss_val


def get_ndcg(pred_list, true_list):
    idcg = sum((1 / np.log2(rank + 2) for rank in range(1, len(pred_list))))
    dcg = 0
    for rank, pred in enumerate(pred_list):
        if pred in true_list:
            dcg += 1 / np.log2(rank + 2)
    ndcg = dcg / idcg
    return ndcg


# hit == recall == precision
def get_hit(pred_list, true_list):
    hit_list = set(true_list) & set(pred_list)
    hit = len(hit_list) / len(true_list)
    return hit


def evaluate(model, data_loader, user_train, user_valid, make_matrix_data_set):
    # print(type(model))
    model.eval()

    NDCG = 0.0  # NDCG@10
    HIT = 0.0  # HIT@10

    with torch.no_grad():
        for users in data_loader:
            mat = make_matrix_data_set.make_matrix(user_list=users, user_train=user_train, user_valid=user_valid,
                                                   train=True)
            mat = mat.to(device)

            recon_mat = model(mat, calculate_loss=False)
            recon_mat[mat == 1] = -np.inf
            rec_list = recon_mat.argsort(dim=1)

            for user, rec in zip(users, rec_list):
                uv = user_valid[user.item()]
                up = rec[-10:].cpu().numpy().tolist()[::-1]
                NDCG += get_ndcg(pred_list=up, true_list=uv)
                HIT += get_hit(pred_list=up, true_list=uv)

    NDCG /= len(data_loader.dataset)
    HIT /= len(data_loader.dataset)

    return NDCG, HIT


def predict(model, data_loader, user_train, user_valid, make_matrix_data_set):
    # print(type(model))
    model.eval()

    user2rec_list = {}
    with torch.no_grad():
        for users in data_loader:
            print(users, "@@@")
            mat = make_matrix_data_set.make_matrix(users, user_train, user_valid, train=False)

            mat = mat.to(device)
            print(mat.shape, "!@#")
            print(mat, "#@!")
            recon_mat = model(mat, calculate_loss=False)
            print(recon_mat.shape, "$%^")
            print(recon_mat, "^%$")
            recon_mat = recon_mat.softmax(dim=1)
            recon_mat[mat == 1] = -1.
            rec_list = recon_mat.argsort(dim=1)
            print(rec_list[0])

            for user, rec in zip(users, rec_list):
                up = rec[-10:].cpu().numpy().tolist()
                user2rec_list[user.item()] = up

    return user2rec_list

def RecVAE_train():
    make_matrix_data_set = MakeMatrixDataSet(config = config)
    best_hit = 0
    for oof in range(1):

        ae_dataset = AEDataSet(
            num_user=make_matrix_data_set.num_user,
        )

        data_loader = DataLoader(
            ae_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            pin_memory=True,
            num_workers=config['num_workers'],
        )

        model = RecVAE(
            input_dim=make_matrix_data_set.num_item,
            hidden_dim=config['hidden_dim'],
            latent_dim=config['latent_dim']).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        user_train, user_valid = make_matrix_data_set.oof_make_dataset(seed=config['seed'] + oof)
        for epoch in range(1, config['num_epochs'] + 1):
            tbar = tqdm(range(1))
            for _ in tbar:
                train_loss = train(
                    model=model,
                    optimizer=optimizer,
                    data_loader=data_loader,
                    user_train=user_train,
                    user_valid=user_valid,
                    make_matrix_data_set=make_matrix_data_set,
                    beta=config['beta'],
                    gamma=config['gamma'],
                    dropout_rate=config['dropout_rate'],
                )

                ndcg, hit = evaluate(
                    model=model,
                    data_loader=data_loader,
                    user_train=user_train,
                    user_valid=user_valid,
                    make_matrix_data_set=make_matrix_data_set,
                )

                if best_hit < hit:
                    best_epoch = epoch
                    best_hit = hit
                    best_ndcg = ndcg
                    best_loss = train_loss
                    print(type(model))
                    torch.save(model.state_dict(), 'RecVae_LSD.pt')
                tbar.set_description(
                    f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')

        print(f'BEST Epoch: {best_epoch:3d}| Train loss: {best_loss:.5f}| NDCG@10: {best_ndcg:.5f}| HIT@10: {best_hit:.5f}')


if __name__ == '__main__':
    con = connection()
    df = pd.read_sql_query("SELECT * FROM 'user_rating'", con)
    valid_user_df = get_valid_user(df)
    update_user_rating(valid_user_df)
    RecVAE_train()
