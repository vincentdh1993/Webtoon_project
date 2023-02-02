# 네이버 웹툰 추천 시스템

![Nwebtoon](https://user-images.githubusercontent.com/17634399/211155655-13b02318-0a1d-4463-8eee-bb5f4bc8503f.gif)

www.webtoonbot.com

딥러닝을 활용한 추천시스템 구현 및 서빙 경험을 위해 진행한 개인 프로젝트 입니다. 개인화된 웹툰 추천시스템 개발을 위해 진행한 작업과정은 다음과 같았습니다.

웹툰 정보+감상이력 크롤링 → SOTA 추천 모델 학습 → 웹페이지 개발 → 배포


# 1. 데이터 수집 (Python)
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

![최소데이터](https://user-images.githubusercontent.com/17634399/215339688-99b4cc8c-c68f-48c1-b987-31ee7f2f1590.png)


![EDA결과](https://user-images.githubusercontent.com/17634399/215339547-3fcda472-64df-4fbd-b0e7-da2b41351a2f.png)

최종적으로 총 972개의 웹툰과 1,759명의 사용자로 이루어진 104,660 개의 데이터를 정제했습니다.

# 2. 추천시스템 모델 개발 (Pytorch)

SOTA 추천 모델 중 ~~~~

1. BERT4REC (ACM, 2019) - 해당 프로젝트의 데이터 특성상 sequential dependency를 가지지 못합니다. (유저가 어느 순서대로 읽었는지 알 수 없음)
때문에 BERT4REC의 모델을 실험으로 돌렸을 때 결과가 처참했음.

2. RecVae (WSDM, 2020)  - 

3. EASE (RecSys,2019) - Computer Vision과는 달리 CF는 hidden layer를 적게 사용하는 것이 성능이 좋다고 하여 hidden layer를 아예 없애버린 linear한 모델. Sparse data에 유리하기 때문에 모델 최종 선택. 

4. MultiVae (WWW, 2018) - 



# 3. 웹페이지 개발 (Django)
1. 페이지 기획

2. 구현

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

# 기타
- 웹툰회사의 관점에서 봤을 때 새로운 아이템들이 자꾸 발굴되어야 하지 않을까?
- Negative Reviews도 모델링에 사용하면 좋지 않을까?
- 쿠팡에서 상품을 둘러보는 것과 달리, 웹툰은 시간을 가지고 보는 컨텐츠인데 웹툰의 특성상 배치 Inference가 더 맞다고 판단.
- 추천시스템의 중립성? Filter Bubble ?
- MAB - 피드백이 실시간으로 모델에 적용되는건데 이건 웹툰 소비 사이클과 관련이 없어보임
- Thumbnail 기반 추천은 어떨까?
