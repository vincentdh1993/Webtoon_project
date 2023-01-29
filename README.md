# 네이버 웹툰 추천 시스템

![Nwebtoon](https://user-images.githubusercontent.com/17634399/211155655-13b02318-0a1d-4463-8eee-bb5f4bc8503f.gif)\

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
데이터를 알아보는 과정에서 왓챠피디아(https://pedia.watcha.com/ko-KR/)의 데이터를 크롤링하기에 적합하다고 판단하였고 python의 Selenium 라이브러리를 활용하여 크롤링을 진행하였습니다. 

![왓챠피디아1](https://user-images.githubusercontent.com/17634399/215336912-30400c93-052d-4238-84b4-02887cf1d51f.png)
#### <타임인조선의 유저 평가 예시>

![왓챠피디아2](https://user-images.githubusercontent.com/17634399/215337333-6e389443-7f2d-4ec9-a800-b82f8d6f3b4a.png)
#### <왓챠피디아 유저A가 평가한 웹툰들 예시>


1. User ↔ Item rating data 수집
  1-1. 왓챠 피디아 - 데이터 수집은 왓챠피디아(https://pedia.watcha.com/ko-KR/)에서 진행을 했습니다. 
2. Item Data 수집
  2-1. 네이버 웹툰 - 

# 2. 추천시스템 모델 개발 (Pytorch)
1. RecVae
  - RecVae
2. EASE
  - Ease
3. Multi-Vae
  - Multi-Vae
4. Ensemble (Hard Voting)
  - Ensemble

# 3. 웹페이지 개발 (Django)
1. 페이지 기획

2. 구현

# 4. 배치 프로세싱 (AirFlow)
Batch Inference vs Online Inference
1. AirFlow 구현
