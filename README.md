# 네이버 웹툰 추천 시스템

![Nwebtoon](https://user-images.githubusercontent.com/17634399/211155655-13b02318-0a1d-4463-8eee-bb5f4bc8503f.gif)

https://www.webtoonbot.com

딥러닝을 활용한 추천시스템 구현 및 서빙 경험을 위해 진행한 개인 프로젝트 입니다. 개인화된 웹툰 추천시스템 개발을 위해 진행한 작업과정은 다음과 같았습니다.

웹툰 정보+감상이력 크롤링 → SOTA 추천 모델 학습 → 웹페이지 개발 → 배포

# 목차
- [1. 데이터 수집 (Python)](#1데이터-수집-Python)
- [2. 추천시스템 모델 개발 (Pytorch)](#2-추천시스템-모델-개발-pytorch)
- [3. 웹페이지 개발 (Django)](#3-웹페이지-개발-django)
- [4. 배치 프로세싱 (AirFlow)](#4-배치-프로세싱-airflow)
- [5. 추가 고민사항 / To-Do List](#추가-고민사항)


# 1.데이터 수집 (Python)
기획단계에서 추천 모델의 두 주축인 협업 필터링(CF) 와 콘텐츠 기반 필터링(CB) 중에서 고민을 했습니다. 

[CF 모델로 진행]
1. User ↔ Item 간의 데이터를 수집할 수 있을까? 
2. CF 모델의 단점인 신규 아이템에 대해서는 어떻게 처리할까? 

[CB 모델로 진행]
1. 웹툰의 Cosine Similarity 또는 Mean Squared Difference를 어떻게 구현할 것인가?

일반적으로 CB모델보다는 CF 모델이 좋은 성능을 내기에 CF 모델로 진행하기로 하였고, 총 Item의 개수를 제한하여 신규 아이템에 대한 가능성을 제외 하였습니다. 

웹툰에 대한 유저 개인의 취향은 비슷할 것이라 가정하고, 리뷰수가 충분하다면 성능이 괜찮은 추천 모델을 구현할 수 있다고 판단했습니다.

데이터를 알아보는 과정에서 왓챠피디아(https://pedia.watcha.com/ko-KR/) 의 데이터를 크롤링하기에 적합하다고 판단하였고 python의 Selenium 라이브러리를 활용하여 크롤링을 진행하였습니다. 

#### <타임인조선의 유저 평가 예시>
![왓챠피디아1](https://user-images.githubusercontent.com/17634399/215336912-30400c93-052d-4238-84b4-02887cf1d51f.png)

#### <왓챠피디아 유저A가 평가한 웹툰들 예시>
![왓챠피디아2](https://user-images.githubusercontent.com/17634399/215337333-6e389443-7f2d-4ec9-a800-b82f8d6f3b4a.png)

웹툰의 플랫폼이 네이버웹툰인 웹툰만 크롤링을 진행하고 사용자들의 explicit feedback (별점) 을 크롤링하여 생각보다 직관적인 데이터를 수집할 수 있었습니다.

평균과 표준 편차 계산을 통해 별점이 3점이상인 웹툰들만 유저가 해당 웹툰을 "즐겨봤다"라고 판단하여 별점이 낮은 기록은 삭제했습니다. (별점이 낮은 리뷰들은 추후 negative review 정보를 추천 모델링에 함께 사용 예정)

![3점미만제거](https://user-images.githubusercontent.com/17634399/215338465-38a522f7-e814-4610-a4a3-096fe1ea8d79.png)

총 수집한 124,703개의 history data 중에서 110,562 개의 데이터를 남기게 되었습니다.

(데이터 제거 전 별점의 평균:3.453, 표준편차: 1.004 / 데이터 제거 후 별점의 평균: 3.707, 표준편차: 0.728)

수집한 웹툰 데이터에는 리뷰가 1개인 웹툰들이 존재하고, 리뷰를 남긴 유저들도 만찬가지였습니다. 

이는 추천시스템의 성능을 저하시킬 수 있으므로 최소 10개 이상의 리뷰를 가진 웹툰과 유저들만으로 이루어진 데이터셋을 구성하였습니다. 

```python
#n개 이상의 데이터 리뷰를 남긴 유저, 웹툰을 필터링 하는 함수
def preprocessing(data,n):
    min_id = data['user'].value_counts() >=n
    min_id = min_id[min_id].index.to_list()
    data = data[data['user'].isin(min_id)]
    
    min_webtoon = data['title'].value_counts() >= n
    min_webtoon = min_webtoon[min_webtoon].index.to_list()
    data = data[data['title'].isin(min_webtoon)]
    
    return data
```



![EDA결과](https://user-images.githubusercontent.com/17634399/215339547-3fcda472-64df-4fbd-b0e7-da2b41351a2f.png)

최종적으로 총 972개의 웹툰과 1,759명의 사용자로 이루어진 104,660 개의 데이터를 정제했습니다.

# 2. 추천시스템 모델 개발 (Pytorch)

현재 발표된 SOTA 논문 중에서, 웹툰 프로젝트와의 적합성, 학습시간 등을 고려하여 여러 실험을 진행하였습니다. 실험을 통해 좋은 성능을 보이는 모델들을 선정하여 Hard-Voting방식의 Ensemble을 통해 최대 20개의 웹툰을 추천하는 시스템을 기획하였습니다.

1. Bert4Rec (ACM, 2019) - Transformer 기반의 모델은 user rating을 임베딩하여 추후에 선호할 영화를 예측합니다. NLP 분야에서 좋은 성능을 보이는 Transformer지만, 해당 프로젝트의 크롤링 된 데이터 특성상 sequential dependency를 가지지 못합니다. e.g. 유저가 읽은 순서, 웹툰 회차별 순서 등. 
따라서 Bert4REC 실험을 진행하였을 때, HIT@10 또는 NDCG@10의 결과가 좋지 못하였고, 실사용 모델로는 선정하지 않았습니다.

    ```python
    class Bert4Rec(nn.Module):
        def __init__(self, num_classes):
            super(Bert4Rec, self).__init__()
            self.num_classes = num_classes
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.fc = nn.Linear(768, num_classes)

        def forward(self, input_ids, attention_mask):
            _, pooled_output = self.bert(input_ids, attention_mask=attention_mask)
            logits = self.fc(pooled_output)
            return logits
    ```


2. RecVAE (WSDM, 2020)  - Encoder 를 통해 user-item representation을 학습하고, Decoder를 통해 feedback score를 예측합니다. 여기서 추가로 composite priror 를 통해 user의 과거 선호도를 모델링하게 되는데, 위 세가지 모듈을 통해서 user-item interaction의 숨은 의미를 유의미하게 나타냅니다. Implicit feedback에 강점을 낸다고 하지만, Explicit feedback에서도 높은 성능을 보여주었고, full-ranking 계산과 달리 한번 학습을 진행할 때, 전체 유저 데이터를 matrix 형태로 사용하기 때문에 학습 시간이 매우 빨랐습니다. 시간, 성능을 고려하여, 해당 프로젝트의 실사용 모델로 선정하게 되었습니다. 


    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    class RecVAE(nn.Module):
        def __init__(self, input_dim, hidden_dim=600, latent_dim=200):
            super(RecVAE, self).__init__()

            #user-item representation 학습
            self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
            #user history preference 학습
            self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
            #user의 feedback score 예측
            self.decoder = nn.Linear(latent_dim, input_dim)

        def reparameterize(self, mu, logvar):
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return eps.mul(std).add_(mu)
            else:
                return mu

        # Encoder를 통해 user_ratings에 대한 mean 과 log variance를 구하고, reparameterzie를 통해 random noise를 추가하여 z를 구합니다. 
        # 최종적으로 z를 Decoder를 통해 x_pred를 구하게 됩니다. gamma, beta, dropout_rate 하이퍼파라미터를 조절하며 loss를 구할 수 있습니다.
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
    ```

3. EASE (RecSys,2019) - Computer Vision과는 달리 CF는 hidden layer를 적게 사용하는 것이 성능이 좋다고 하여 hidden layer를 아예 없애버린 linear한 모델입니다. Graph Embedding 방법을 착안한 모델이며, 구성이 매우 단순하여 딥러닝 모델이라고 보기 어려운 면도 있지만 성능과 학습시간은 매우 뛰어났습니다. 다른 Autoencoder 처럼 latent factor를 통해 추천을 하지는 않지만 input 데이터가 ouput 데이터로 재생성 됩니다. 성능과 학습시간을 고려하여 최종 실사용 모델로 선정하였습니다.

    ```python
    class EASE():
        def __init__(self, X, reg):
            self.X = self._convert_sp_mat_to_sp_tensor(X)
            self.reg = reg

        #scipy sparse matrix를 PyTorch sparse Tensor로 변환 하는 함수
        def _convert_sp_mat_to_sp_tensor(self, X):
            coo = X.tocoo().astype(np.float32)
            i = torch.LongTensor(np.mat([coo.row, coo.col]))
            v = torch.FloatTensor(coo.data)
            res = torch.sparse.FloatTensor(i, v, coo.shape).to(device)
            return res

        def fit(self):
            #sparse matrix인 'self.X'를 dense matrix로 변환하고 self.X 와 self.X를 곱하여 G matrix에 저장합니다.
            G = self.X.to_dense().t() @ self.X.to_dense()
            #G.shape[0] x G.shape[0] 크기의 matrix를 만들며, 대각선의 값들은 1로 채워서 만들고 나머지는 0으로 합니다.
            diagIndices = torch.eye(G.shape[0]) == 1
            #self.reg를 G matrix에 추가
            G[diagIndices] += self.reg

            #G inverse를 계산하여 P matrix에 담습니다.
            P = G.inverse()
            # P 를 -P의 대각선으로 나누고 B matrix에 담습니다. 
            #여기서 B는 아이템 간의 가중치 행렬을 나타냅니다. (유일한 학습 파라미터)
            B = P / (-1 * P.diag())
            # B의 대각선을 0으로 만듭니다.
            B[diagIndices] = 0
            self.B = B
            #self.X에 B를 곱해주며 결과값을 self.pred에 저장합니다.
            self.pred = self.X.to_dense() @ B
    ```

4. MultiVae (WWW, 2018) - 

5. VASP (ICAN, 2021) - FLVAE (Colloborative Filtering VAE) + Neural EASE 의 모델로, non-linear와 linear 성향을 모두 모델링 하기 위한 방법입니다. 두개의 모델을 따로 계산 한 뒤에 각각 sigmoid를 씌운 상태로 요소곱을 하여 합치게 되는 모델입니다. https://paperswithcode.com/sota/collaborative-filtering-on-movielens-20m 에 당당히 1위를 기록중인 SOTA 모델로, 해당 웹툰 프로젝트에 적용하기 위해 논문을 읽고 분석해보기로 하였습니다. 저자가 공식적으로 제공한 코드는 Keras로 작성되어 있었고, 인터넷 검색을 해도 참고할 만 한 PyTorch 코드가 없어서 오피셜 Keras 코드와 논문 내용을 기반으로 PyTorch 코드를 아래와 같이 간략하게 작성하였습니다.

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class DiagonalToZero(nn.Module):
        def forward(self, w):
            """Set diagonal to zero"""
            q = w.clone()
            q[torch.eye(q.size(-2), dtype=torch.bool)] = 0
            return q

    class Sampling(nn.Module):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a basket."""
        def forward(self, inputs):
            z_mean, z_log_var = inputs
            batch = z_mean.shape[0]
            dim = z_mean.shape[1]
            epsilon = torch.randn(batch, dim)
            return z_mean + torch.exp(0.5 * z_log_var) * epsilon


    class VASP(nn.Module):
        def __init__(self, num_words, latent=1024, hidden=1024, items_sampling=1.):
            super(VASP, self).__init__()

            self.sampled_items = int(num_words * items_sampling)

            assert self.sampled_items > 0
            assert self.sampled_items <= num_words

            self.s = self.sampled_items < num_words

            # ************* ENCODER ***********************
            self.encoder1 = nn.Linear(hidden)
            self.ln1 = nn.LayerNorm(hidden)

            # ************* SAMPLING **********************
            self.dense_mean = nn.Linear(latent, name="Mean")
            self.dense_log_var = nn.Linear(latent, name="log_var")

            self.sampling = Sampling(name='Sampler')

            # ************* DECODER ***********************
            self.decoder1 = nn.Linear(hidden)
            self.dln1 = nn.LayerNorm(hidden)

            self.decoder_resnet = nn.Linear(self.sampled_items,
                                            activation=torch.sigmoid,
                                            name="DecoderR")
            self.decoder_latent = nn.Linear(self.sampled_items,
                                            activation=torch.sigmoid,
                                            name="DecoderL")

            # ************* PARALLEL SHALLOW PATH *********

            self.ease = nn.Linear(
                self.sampled_items,
                activation=torch.sigmoid,
                bias=False,
                weight=nn.Parameter(torch.eye(self.sampled_items), requires_grad=False),
            )

        def forward(self, x, training=False):
            sampling = self.s
            if sampling:
                sampled_x = x[:, :self.sampled_items]
                non_sampled = x[:, self.sampled_items:] * 0.
            else:
                sampled_x = x

            z_mean, z_log_var, z = self.encode(sampled_x)
            if training:
                d = self.decode(z)
                # Add KL divergence regularization loss.
                kl_loss = 1 + z_log_var - torch.pow(z_mean, 2) - torch.exp(z_log_var)
                kl_loss = kl_loss.mean()
                kl_loss *= -0.5
                return d, kl_loss
            else:
                d = self.decode(z_mean)

            if sampling:
                d = torch.cat([d, non_sampled], dim=-1)

            ease = self.ease(sampled_x)

            if sampling:
                ease = torch.cat([ease, non_sampled], dim=-1)
            out = self.decoder_resnet(d) + self.decoder_latent(ease)
            return out

        def encode(self, x):
            x = self.encoder1(x)
            x = self.ln1(x)
            x = F.relu(x)

            mean = self.dense_mean(x)
            log_var = self.dense_log_var(x)

            z = self.sampling(mean, log_var)

            return mean, log_var, z

        def decode(self, z):
            z = self.decoder1(z)
            z = self.dln1(z)
            z = F.relu(z)

            return z
    ```

    학습에 걸리는 시간이 너무 오래 걸렸고, 딥러닝 모델 서버를 따로 운용을 하지 못하는 상황이기에 당장 적용은 힘들다고 판단하였습니다. 추후, 코드 최적화를 진행하여 Lightsail 자체 운영이 가능하게 되면 적용할 예정입니다.


    ```python
        class NCF(nn.Module):
            def __init__(self, num_users, num_items, num_genres, latent_dim=32, num_hidden_layers=3, hidden_dim=64):
                super(NCF, self).__init__()

                self.num_users = num_users
                self.num_items = num_items
                self.num_genres = num_genres
                self.latent_dim = latent_dim
                self.num_hidden_layers = num_hidden_layers
                self.hidden_dim = hidden_dim

                # User embedding layer
                self.user_embedding = nn.Embedding(num_users, latent_dim)

                # Item embedding layer
                self.item_embedding = nn.Embedding(num_items, latent_dim)

                # Genre embedding layer
                self.genre_embedding = nn.Embedding(num_genres, latent_dim)

                # MLP layers
                self.fc1 = nn.Linear(3 * latent_dim, hidden_dim)
                self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers - 1)])
                self.fc_out = nn.Linear(hidden_dim, 1)

            def forward(self, user_id, item_id, genre_id):
                user_vector = self.user_embedding(user_id)
                item_vector = self.item_embedding(item_id)
                genre_vector = self.genre_embedding(genre_id)

                concat = torch.cat([user_vector, item_vector, genre_vector], dim=-1)
                x = F.relu(self.fc1(concat))

                for i in range(self.num_hidden_layers - 1):
                    x = F.relu(self.hidden_layers[i](x))

                x = self.fc_out(x)
                x = torch.sigmoid(x)

                return x
    ```
    
    ```python
        class NeuMF(nn.Module):
            def __init__(self, cfg: Config):
                super(NeuMF, self).__init__()
                self.n_users = cfg.n_users
                self.n_items = cfg.n_items
                self.emb_dim = cfg.emb_dim
                self.layer_dim = cfg.layer_dim
                self.n_continuous_feats = cfg.n_continuous_feats
                self.n_genres = cfg.n_genres
                self.dropout = cfg.dropout

                self.build_graph()

            def build_graph(self):
                self.user_embedding_mf = nn.Embedding(self.n_users, self.emb_dim)
                self.item_embedding_mf = nn.Embedding(self.n_items, self.emb_dim)

                self.user_embedding_mlp = nn.Embedding(self.n_users, self.emb_dim)
                self.item_embedding_mlp = nn.Embedding(self.n_items, self.emb_dim)

                self.genre_embeddig = nn.Embedding(self.n_genres, self.n_genres // 2)

                self.mlp_layers = nn.Sequential(
                    nn.Linear(2 * self.emb_dim + self.n_genres // 2 + self.n_continuous_feats, self.layer_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.layer_dim, self.layer_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                )
                self.affine_output = nn.Linear(self.layer_dim // 2 + self.emb_dim, 1)
                self.apply(self._init_weights)

            def _init_weights(self, module):
                if isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0, 0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

            def forward(self, user_indices, item_indices, feats):
                user_embedding_mf = self.user_embedding_mf(user_indices)
                item_embedding_mf = self.item_embedding_mf(item_indices)
                mf_output = torch.mul(user_embedding_mf, item_embedding_mf).sum(dim=-1)
                user_embedding_mlp = self.user_embedding_mlp(user_indices)
                item_embedding_mlp = self.item_embedding_mlp(item_indices)
                genre_embedding = self.genre_embeddig(feats[:, -self.n_genres:].long())
                mlp_input = torch.cat(
                    (user_embedding_mlp, item_embedding_mlp, genre_embedding, feats[:, :self.n_continuous_feats]), dim=-1)
                mlp_output = self.mlp_layers(mlp_input)
                combined_output = torch.cat((mlp_output, mf_output), dim=-1)
                prediction = self.affine_output(combined_output).squeeze(-1)
                return prediction
    ```


6. Ensemble - 23년 2월 1일 기준으로 사용하고 있는 모델은 EASE, RecVAE, MultiVAE의 3가지 모델입니다. 각 모델의 결과의 Top-10 결과를 추출하게 되고 다수결의 원칙과 비슷한 개념인 하드 보팅으로 최종 예측값을 반환하게 됩니다. RecVAE 모델과 MultiVAE 모델은 둘 다 Variational Auto Encoder를 사용하며 유사한 결과를 반환하지만 EASE 모델은 때때로 더 독창적인 결과를 반환하기에 결정 prediction score를 활용하는 소프트 보팅보다는 하드보팅이 선택하게 되었습니다.

| 분류 | 모델 | Hit@10 | 활용 여부 |
|----------|----------|----------|----------|
| Transformer | BERT4Rec | 0.1146 | X |
| Auto Encoder | Multi-VAE | 0.2561 | O |
| Auto Encoder | RecVAE | 0.2878 | O |
| Auto Encoder | EASE | 0.3012 | O |
| Auto Encoder | VASP | 0.2769 | X |
| MLP | NCF | 0.2440 | X |
| MLP | NeuMF | 0.2648 | X |


# 3. 웹페이지 개발 (Django)
 - 장고 vs 플라스크를 고민하다가 보안에 강점이 있고 빠른 개발이 가능한 장고를 선택하게 되었습니다.에디터는 PyCharm을 사용하였습니다. 

1. "재밌게" 읽은 웹툰 입력
    - 개인화 추천결과를 return 하기 위해서는 유저가 읽었던 웹툰 정보를 알아야 했기에 (유저가 웹툰을 한번도 읽지 않은 Cold Start Problem 인 경우까지 포함) checkbox로 "재밌게" 읽었던 웹툰을 체크하고 추천결과를 받게끔 기획하였습니다. 사용자에게 시각적 도움을 주기 위해 네이버웹툰 홈페이지에서 썸네일 이미지들을 크롤링하였으며, ㄱ,ㄴ,ㄷ...,전체 탭을 생성하여 사용자가 최대한 편하게 체크박스를 클릭할 수 있게 기획하였습니다. 웹툰 선택을 완료하여 우측에 엄지 버튼을 누르면 webtoon_list 가 추천시스템의 모델에 input으로 들어가게 되고, return 값을 반환하게 됩니다. 추천결과를 연산하는 동안 추가 input을 받지 않도록 blur 처리를 하며 로딩 이미지를 보여 줍니다.

![첫화면_체크](https://user-images.githubusercontent.com/17634399/217294792-e07ee859-be1c-4161-8f45-b5180ac40c83.png)

2. 추천결과 확인
    - 추천시스템의 연산이 끝나게 되면 최대 20개의 웹툰이 추천되는데, 추천결과를 단순 나열하기 보다는 Explainable AI (XAI)를 구현하기 위해서 장르 정보를 활용하였습니다. 네이버웹툰에서 제공하는 장르들을 사용하여 input webtoon_list 와 같은 장르로 나누어 결과를 보여주게 됩니다. (모바일 웹툰앱의 해시태그 키워드가 있지만 크롤링이 어려워 추후 To-Do List에 넣어두었습니다.) 추천결과를 받은 사용자는 썸네일을 클릭하여 해당 웹툰을 볼 수 있는 페이지로 갈 수 있습니다. 또한, "처음부터 다시하기" 를 눌러 다시 추천결과를 받을 수 있습니다. 
    - 추천 결과를 "Good" 또는 "Bad"로 평가할 수 있는데, "Good"을 누른 경우에는 사용자가 input으로 지정한 webtoon_list를 현재시각 기준으로 DB에 good flag와 함께 저장하게 됩니다. 반대로 "Bad"을 누르면 bad flag로 저장하게 됩니다. 결과를 평가한 유저들의 새로운 webtoon_list input으로 추천시스템 모델을 반복학습 할 예정이며, Apache-Airflow를 통해 모델을 주기적으로 업데이트 하여 반복학습 환경을 구축하게 됩니다. 
    
![추천결과](https://user-images.githubusercontent.com/17634399/217295544-792a01b5-fb79-4aba-8199-5c04b3b170d5.png)

# 4. 배치 프로세싱 (AirFlow)

딥러닝 모델을 업데이트 시켜주는 방법으로 Batch Inference 또는 Online Inference가 있습니다.

실제로 웹툰 컨텐츠를 운영하는 회사 입장에서 생각을 해보았을 때, 유저가 웹툰을 단순히 클릭한 것으로 feedback을 얻기는 힘들다고 판단했습니다.

이커머스 도메인의 경우, 유저가 특정 아이템을 클릭한것 만으로도 data instight를 유추할 수 있는 반면, 웹툰 도메인에서는 유저와 특정 아이템간의 interaction이 일정 수준이 되어야 해당 유저가 웹툰을 "즐겁게" 소비했다고 볼 수 있습니다.

따라서, 모델을 실시간으로 업데이트 하기보다는 지정 시간에 추가 데이터를 학습하여 모델을 업데이트하고 저장하는 방식을 생각했습니다.
(실제 상용 서비스에서는 딥러닝 모델 서버가 분리되어 해당 서버의 모델을 업데이트 하는 방식이라 생각됩니다. 제 모델과 데이터양은 다행히 Lightsail None-GPU 환경에서도 무리없이 학습이 가능했습니다.)

배치 프로세싱은 Airbnb의 Apache-airflow를 사용하였으며, webtoonbot을 이용하고 "Good" 버튼을 누른 사용자 중에서 10개 이상의 웹툰을 선택했던 유저의 데이터만 추가학습에 활용 하였습니다.

```python
#10개 이상의 웹툰을 선택한 유저의 데이터만 사용
def read_ratings():
    conn = sqlite3.connect('dq.sqlite3')
    df = pd.read_sql_query("SELECT * FROM user_rating", conn)
    user_counts = df['user'].value_counts()
    users_with_more_than_10_ratings = user_counts[user_counts > 10].index
    df_filtered = df[df['user'].isin(users_with_more_than_10_ratings)]
    print(df_filtered)
    conn.close()
```

추가 데이터를 획득하여 모델에 새로 학습을 시켜주는 코드를 AirFlow를 활용하여 매일 오전 3시에 실행시켜 줍니다. (유저의 정보를 함께 불러오는 등의 여러 task가 있으면 여러 DAG를 통해 AirFlow를 풍성하게 사용할 수 있겠으나 아직 추가 task에 대한 방향 설계가 안되어있습니다.)


1. AirFlow 구현
```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'update_model_dag',
    default_args=default_args,
    description='python run update_model.py every 3am',
    schedule_interval='0 3 * * *'
)

update_model = PythonOperator(
    task_id='update_model',
    python_callable='update_model.py',
    dag=dag
)
```
해당 코드를 airflow webserver 와 airflow scheduler를 실행시키고 http://localhost:8080으로 접속해서 DAG를 on 시켜주었습니다.

# 추가 고민사항
1.웹툰회사의 관점에서 봤을 때, 새로운 아이템들이 자꾸 발굴되어야 하지 않을까?
- 네이버웹툰의 입장이 되어 생각을 해보면, 수익성을 위해서는 유저들이 새로운 웹툰을 시도하고 정착하는것이 어쩌면 단순 추천성능보다 더 중요하다는 생각이 들었습니다.
- 추천시스템에 의해 Filter Bubble 이 발생하게 되어 한 방향으로만 Filter Bubble이 발생하게 되면 결국 나중에는 추천시스템의 본 기능을 잃게 될것이라는 생각이 들었습니다. 

2. Negative Reviews도 모델링에 사용하면 좋지 않을까?
- Explicit Data인 별점 데이터에서 2.5/5.0을 Threshold로 잡고 필터링을 진행했는데, 유저가 A 웹툰을 "안 좋아했다" 라는 정보는 "안 보았다" 와는 차이가 있으므로 추천 모델링에 같이 반영해야 되겠다 라는 생각이 들었습니다.

3. Thumbnail 기반 추천?
- 추천시스템을 개발하면서 수백개의 Thumbnail들을 보게 되었는데 생각보다 패턴이 존재하는것 같았습니다. 
   (예시: 연애물의 경우에는 남녀가 썸네일로 등장하고, 액션물의 경우에는 주인공이 매서운 눈을 가지고 썸네일에 등장)
   썸네일 이미지를 모델링 하여 썸네일간 유사도를 기반으로 한 추천모델도 유의미 하지 않을까 라는 생각을 하게되었고, To-Do List에 추가 하였습니다.

4. 강화 학습 기반 추천 시스템?
- 로그데이터에 접근이 가능한 경우, 톰슨 샘플링 (MAB 알고리즘) 등을 활용하여 유저에게 웹툰을 추천하였을 때, 그 추천 결과를 "클릭" 하고 "소비" 한 데이터를 활용하여 추천에 대한 피드백을 학습하여 유저의 history 의존도를 낮출 수 있지 않을까 라는 생각을 했습니다.

5. Explainable AI (XAI) 
- XAI 를 위해 추천이 된 이유와 사용자가 흥미를 이끌만한 부가적인 정보를 제공을 기획하였습니다. 하지만 크롤링 된 데이터로는 "장르" 와 "작가" 만 사용할 수 있는 상황이기 때문에 아쉬움이 남았습니다. 이를 보완하기 위해서 왓챠피디아의 "웹툰평" 과 네이버웹툰 앱의 "해시태그"를 활용하고자 합니다.
- "웹툰평" 은 자연어 처리쪽의 TF-ID를 활용하여 웹툰에 대한 코멘트들에서 키워드를 추출 할 계획입니다.
- "해시태그" 앱을 크롤링 하는 기술은 워낙 고도의 기술이기 때문에 (불법적인 요소 포함), 매크로를 통해 웹툰 별 해시태그 화면을 스크린샷으로 촬영하고, OCR 기술을 통해 문자 추출을 진행 할 계획입니다.

