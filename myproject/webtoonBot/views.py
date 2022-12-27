from django.shortcuts import render
import pandas as pd
import math
import numpy as np
import scipy.sparse as sp
import pandas as pd
# from tqdm import tqdm
from collections import defaultdict
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from box import Box
from copy import deepcopy

import ast

# from MakeMatrixDataSet import MakeMatrixDataSet
# import AEDataSet
# import EASE

import warnings

# warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=True)

class EASE():
    def __init__(self, X, reg):
        self.X = self._convert_sp_mat_to_sp_tensor(X)
        self.reg = reg

    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        Convert scipy sparse matrix to PyTorch sparse matrix

        Arguments:
        ----------
        X = Adjacency matrix, scipy sparse matrix
        """
        coo = X.tocoo().astype(np.float32)
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        res = torch.sparse.FloatTensor(i, v, coo.shape).to(device)
        return res

    def fit(self):
        '''

        진짜 정말 간단한 식으로 모델을 만듬

        '''
        G = self.X.to_dense().t() @ self.X.to_dense()
        diagIndices = torch.eye(G.shape[0]) == 1
        G[diagIndices] += self.reg

        P = G.inverse()
        B = P / (-1 * P.diag())
        B[diagIndices] = 0
        self.B = B
        self.pred = self.X.to_dense() @ B

class AEDataSet(Dataset):
    def __init__(self, num_user):
        self.num_user = num_user
        self.users = [i for i in range(num_user)]

    def __len__(self):
        return self.num_user

    def __getitem__(self, idx):
        user = self.users[idx]
        return torch.LongTensor([user])

class MakeMatrixDataSet():
    """
    MatrixDataSet 생성
    """

    def __init__(self, config, df):
        self.config = config
        self.df = df
        self.df = df.rename(columns={'title': 'item'})
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
        """
        sequence_data 생성

        Returns:
            dict: train user sequence / valid user sequence
        """
        users = defaultdict(list)
        user_train = {}
        user_valid = {}
        for user, item in zip(self.df['user_idx'], self.df['item_idx']):
            users[user].append(item)

        for user in users:
            np.random.seed(self.config.seed)

            user_total = users[user]
            valid = np.random.choice(user_total, size=self.config.valid_samples, replace=False).tolist()
            train = list(set(user_total) - set(valid))

            user_train[user] = train
            user_valid[user] = valid

        return user_train, user_valid

    def get_train_valid_data(self):
        return self.user_train, self.user_valid

    def make_sparse_matrix(self, test=False):
        X = sp.dok_matrix((self.num_user, self.num_item), dtype=np.float32)

        for user in self.user_train.keys():
            item_list = self.user_train[user]
            X[user, item_list] = 1.0

        if test:
            for user in self.user_valid.keys():
                item_list = self.user_valid[user]
                X[user, item_list] = 1.0

        return X.tocsr()

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
            valid = np.random.choice(user_total, size=self.config.valid_samples, replace=False).tolist()
            train = list(set(user_total) - set(valid))

            user_train[user] = train
            user_valid[user] = valid

        return user_train, user_valid

    def make_matrix(self, user_list, user_train, user_valid, train=True):
        """
        user_item_dict를 바탕으로 행렬 생성
        """
        mat = torch.zeros(size=(user_list.size(0), self.num_item))
        for idx, user in enumerate(user_list):
            if train:
                mat[idx, user_train[user.item()]] = 1
            else:
                mat[idx, user_train[user.item()] + user_valid[user.item()]] = 1
        return mat

def toFinalForm(submission):
  d={}
  for idx,row in submission.iterrows():
    user = (row['user'])
    item = (row['item'])
    if user in d:
      d[user] = d[user]+[item]
    else:
      d[user] = [item]
  df = pd.DataFrame(list(d.items()),columns = ['user','item'])
  return df


def recvae_predict(model, data_loader, user_train, user_valid, make_matrix_data_set,device):
    model.eval()

    user2rec_list = {}
    with torch.no_grad():
        for users in data_loader:
            mat = make_matrix_data_set.make_matrix(users, user_train, user_valid, train=False)
            mat = mat.to(device)
            recon_mat = model(mat, calculate_loss=False)
            recon_mat = recon_mat.softmax(dim=1)
            recon_mat[mat == 1] = -1.
            rec_list = recon_mat.argsort(dim=1)

            for user, rec in zip(users, rec_list):
                up = rec[-10:].cpu().numpy().tolist()
                user2rec_list[user.item()] = up

    return user2rec_list


def getConfig(new_item_list):
  valid_samples=0

  if len(new_item_list) == 1:
    valid_samples = len(new_item_list)//2
  else:
    valid_samples=2

  recvae_config = {
      'batch_size' : 500, #
      'num_workers' : 2, #
      'valid_samples' : valid_samples,  #
      'seed' : 22, #
  }

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  recvae_config = Box(recvae_config)

  EASE_config = {
      'candidate_item_num' : 3,
      'valid_samples' : valid_samples, # 검증에 사용할 sample 수
      'seed' : 22,
      'reg' : 750,
  }

  EASE_config = Box(EASE_config)
  return recvae_config,EASE_config


def RecVae_get_recomendation(new_user_name, new_item_list,recvae_config,model3):

    new_df = pd.DataFrame()
    new_df['user'] = [new_user_name for i in range(len(new_item_list))]
    new_df['title'] = new_item_list
    new_df.to_csv("new_user_df.csv", encoding="euc-kr")
    og_df = pd.read_csv("user_rating_10.csv", encoding="euc-kr")
    frames = [og_df, new_df]
    result_df = pd.concat(frames)

    make_matrix_data_set_new = MakeMatrixDataSet(config=recvae_config, df=result_df)
    ae_dataset_new = AEDataSet(num_user=make_matrix_data_set_new.num_user, )
    user_train_new, user_valid_new = make_matrix_data_set_new.oof_make_dataset(seed=recvae_config.seed)

    submission_data_loader_new = DataLoader(
        ae_dataset_new,
        batch_size=recvae_config.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=recvae_config.num_workers,
    )

    user2rec_list2 = recvae_predict(
        model=model3,
        data_loader=submission_data_loader_new,
        user_train=user_train_new,
        user_valid=user_valid_new,
        make_matrix_data_set=make_matrix_data_set_new,
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    )

    submission_new = []
    users = [i for i in range(0, make_matrix_data_set_new.num_user)]
    users = users[-1:]
    for user in users:
        rec_item_list = user2rec_list2[user]
        for item in rec_item_list:
            submission_new.append(
                {
                    'user': make_matrix_data_set_new.user_decoder[user],
                    'item': make_matrix_data_set_new.item_decoder[item],
                }
            )
    submission_new = pd.DataFrame(submission_new)
    final = toFinalForm(submission_new)
    final_list = (final['item'][0])
    # for i in final_list:
    # print(i)
    return final_list

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

def evaluate(model, X, user_train, user_valid, user_list):

    mat = torch.from_numpy(X)

    NDCG = 0.0 # NDCG@10
    HIT = 0.0 # HIT@10

    recon_mat = model.pred.cpu()
    recon_mat[mat == 1] = -np.inf
    rec_list = recon_mat.argsort(dim = 1)

    for user, rec in enumerate(rec_list):
        if user in user_list:
            uv = user_valid[user]
            up = rec[-10:].cpu().numpy().tolist()[::-1]
            NDCG += get_ndcg(pred_list = up, true_list = uv)
            HIT += get_hit(pred_list = up, true_list = uv)

    NDCG /= len(user_list)
    HIT /= len(user_list)

    return NDCG, HIT


def EASE_predict(model, X):
    user2rec = {}
    recon_mat = model.pred.cpu()
    score = recon_mat * torch.from_numpy(1 - X)
    rec_list = score.argsort(dim=1)

    for user, rec in enumerate(rec_list):
        up = rec[-10:].cpu().numpy().tolist()
        user2rec[user] = up

    return user2rec

def EASE_get_recomendation(new_user_name,new_item_list,EASE_config):
  new_df = pd.DataFrame()
  new_df['user'] = [new_user_name for i in range(len(new_item_list))]
  new_df['title'] = new_item_list
  og_df = pd.read_csv("user_rating_10.csv",encoding="euc-kr")
  frames = [og_df, new_df]
  result_df = pd.concat(frames)
  make_matrix_data_set = MakeMatrixDataSet(config = EASE_config,df = result_df)
  user_train, user_valid = make_matrix_data_set.get_train_valid_data()
  X = make_matrix_data_set.make_sparse_matrix()
  model = EASE(X = X, reg = 680)
  model.fit()
  # ndcg, hit = evaluate(model = model, X = X.todense(), user_train = user_train, user_valid = user_valid,user_list = user_valid)
  # print(f'NDCG@10: {ndcg:.5f}| HIT@10: {hit:.5f}')
  make_matrix_data_set = MakeMatrixDataSet(config = EASE_config,df=result_df)
  X_test = make_matrix_data_set.make_sparse_matrix(test = True)
  model = EASE(X = X_test, reg = EASE_config.reg)
  model.fit()

  user2rec_list = EASE_predict(
    model = model,
    X = X_test.todense(),
    )
  submission = []
  users = [i for i in range(0, make_matrix_data_set.num_user)]
  users = users[-1:]
  for user in users:
      rec_item_list = user2rec_list[user]
      for item in rec_item_list:
          submission.append(
              {
                  'user' : make_matrix_data_set.user_decoder[user],
                  'item' : make_matrix_data_set.item_decoder[item],
              }
          )

  submission = pd.DataFrame(submission)
  final = toFinalForm(submission)
  final_list = (final['item'][0])
  # for i in final_list:
  #   print(i)
  return final_list

def getFinalList(recvae_list,ease_list):
  d={}
  for i in range(len(recvae_list)):
    r = recvae_list[i]
    e = ease_list[i]
    if r in d:
      d[r] +=1
    else:
      d[r]=1
    if e in d:
      d[e] +=1
    else:
      d[e] = 1
  total_list = []
  for i in d:
    total_list.append([i,d[i]])
  total_list = sorted(total_list,key=lambda x:x[1],reverse=True)

  two_votes=[]
  one_vote=[]

  for i in total_list:
    if i[1] == 2:
      two_votes.append(i[0])
    else:
      one_vote.append(i[0])

  total_list = [i[0] for i in total_list]
  # print(total_list)
  print("총 "+str(len(total_list))+"개의 웹툰이 추천되었습니다 :)")
  for i in two_votes:
    print(i)
  for i in one_vote:
    print(i)

  return two_votes+one_vote

def index(request):
    return render(request, 'webtoonBot/index.html')

def ver3(request):
    og_list = pd.read_csv("user_rating_10.csv", encoding="euc-kr")
    webtoon_list = list(og_list['title'].unique())
    thumbnail_list = list(og_list['thumbnail'].unique())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model3 = torch.load("recVae_model_test.pt",map_location=torch.device('cpu'))
    item_encoder = getCoder("item_encoder")
    item_decoder = getCoder("item_decoder")
    user_encoder = getCoder("user_encoder")
    user_decoder = getCoder("user_decoder")

    new_item_list = []
    if request.method == 'POST':
        new_item_list = request.POST.getlist('user_webtoon_list')
        new_user_name = "vincenzodh"
        print(new_item_list,"$")

        recvae_config, EASE_config = getConfig(new_item_list)
        recvae_list = RecVae_get_recomendation(new_user_name, new_item_list, recvae_config, model3)
        ease_list = EASE_get_recomendation(new_user_name, new_item_list, EASE_config)
        final_list = getFinalList(recvae_list, ease_list)


        return render(request, 'webtoonBot/ver3_result.html',
                      {'final_list': final_list})
    else:
        return render(request, 'webtoonBot/ver3.html', {'webtoon_list': webtoon_list,'thumbnail_list':thumbnail_list})

def ver4 (request):
    og_list = pd.read_csv("user_rating_10.csv", encoding="euc-kr")
    webtoon_list = list(og_list['title'].unique())
    thumbnail_list = list(og_list['thumbnail'].unique())
    actual_url_df = pd.read_csv("actual_NW_url_with_thumbnails_ansi.csv",encoding="euc-kr")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model3 = torch.load("recVae_model_test.pt",map_location=torch.device('cpu'))
    item_encoder = getCoder("item_encoder")
    item_decoder = getCoder("item_decoder")
    user_encoder = getCoder("user_encoder")
    user_decoder = getCoder("user_decoder")

    new_item_list = []
    if request.method == 'POST':
        new_item_list = request.POST.getlist('user_webtoon_list')
        new_user_name = "vincenzodh"
        print(new_item_list,"$")

        recvae_config, EASE_config = getConfig(new_item_list)
        recvae_list = RecVae_get_recomendation(new_user_name, new_item_list, recvae_config, model3)
        ease_list = EASE_get_recomendation(new_user_name, new_item_list, EASE_config)
        final_list = getFinalList(recvae_list, ease_list)
        actual_url=[]
        combined_list = []
        for i in final_list:
            url = actual_url_df.loc[actual_url_df['title'] == i, 'url']
            thumb = og_list.loc[og_list['title']==i,'thumbnail']
            url = url.values[0]
            actual_url.append(url)
            combined_list.append([i,url])



        return render(request, 'webtoonBot/ver4_result.html',
                      {'final_list': final_list,'new_item_list':new_item_list,'actual_url':actual_url,'combined_list':combined_list})
    else:
        return render(request, 'webtoonBot/ver4.html', {'webtoon_list': webtoon_list,'thumbnail_list':thumbnail_list})

def ver1(request):
    return render(request, 'webtoonBot/ver1.html')

def ver2(request):
    webtoon_list = pd.read_csv("user_rating_10.csv",encoding="euc-kr")
    webtoon_list = list(webtoon_list['title'].unique())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model3 = torch.load("recVae_model_test.pt",
                        map_location=torch.device('cpu'))
    item_encoder = getCoder("item_encoder")
    item_decoder = getCoder("item_decoder")
    user_encoder = getCoder("user_encoder")
    user_decoder = getCoder("user_decoder")


    if request.method == 'POST':
        new_user_name = "vincenzodh"
        new_item_list = ["헬퍼", "두근두근두근거려", "목욕의 신", "병의 맛", "삼봉이발소", "스퍼맨 시즌1", "안나라수마나라", "호랑이형님", "입시명문사립 정글고등학교",
                         "무한동력"]
        new_item_list = []
        for i in range(1,6):
            new_item_list.append(request.POST.get('beer' + str(i), ''))
        print(new_item_list,"@#@#@#@@#@")
        recvae_config, EASE_config = getConfig(new_item_list)

        recvae_list = RecVae_get_recomendation(new_user_name, new_item_list,recvae_config,model3)
        ease_list = EASE_get_recomendation(new_user_name, new_item_list,EASE_config)
        final_list = getFinalList(recvae_list, ease_list)
        return render(request, 'webtoonBot/ver2_result.html',
                      {'final_list': final_list})
    else:
        return render(request, 'webtoonBot/ver2.html', {'webtoon_list': webtoon_list})

def getCoder(coder_name):
  filename = str(coder_name)+".txt"
  with open(filename,encoding="utf-8") as f:
    lines = f.readlines()
  coder = ast.literal_eval(lines[0])
  return coder
