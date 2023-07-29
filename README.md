# BABIP
#### 쉬운 머신 러닝 모델 개선을 위한 파이썬 라이브러리
## Features
* 개발 의도: 인공 지능 분야의 높은 진입장벽을 없애고자 개발
* 한국어 데이터 최적화: 한국어 데이터 전처리에 어려움을 겪지 않도록 한국어 데이터에 최적화
* 쉬운 사용: 모든 기능을 코드 한 줄로 구현 가능
## Installation
### Dependencies
BABIP requires:
* scikit-learn
### User Installation
```
pip install BABIP
```
## Modules
### rec
------------
#### 추천 시스템 개발을 위한 모듈
* tfidf(): TF-IDF 행렬
```
rec.tfidf(data, col_contents)
```
* cos_similarity(): 코사인 유사도
```
rec.cos_similarity(data, col_contents)
```
* recommend(): 추천 시스템
```
rec.recommend(title, data, col_title, col_contents)
```

###### 예시 데이터 (movie_data.csv)
|title|story|
|---|---|
|클래식|같은 대학에 다니는 지혜(손예진 분)와 수경(이수인 분)은 연극반 선배 상민(조인성 분)을 좋아한다. 하지만...|
|인터스텔라|세계 각국의 정부와 경제가 완전히 붕괴된 미래가 다가온다. 지난 20세기에 범한 잘못이 전 세계적인 식량 부족을 불러왔고, NASA도 해체되었다. 이때...|
|인셉션|드림머신이라는 기계로 타인의 꿈과 접속해 생각을 빼낼 수 있는 미래사회.‘돔 코브’(레오나르도 디카프리오)는 생각을 지키는 특수보안요원이면서 또한 최고의 실력으로 생각을 훔치는 도둑이다...|
###### 예시 코드
``` python
from babip import rec
import pandas as pd

# 데이터 로드
data_path = './../movie_data.csv'
data = pd.read_csv(data_path, encoding='cp949')

# 칼럼 이름
col_title = 'title' # 아이템 이름 정보를 담는 칼럼의 이름
col_contents = 'story' # 문서 유사도 비교에 쓰일 자연어 정보를 담는 칼럼의 이름

# TF-IDF 행렬
matrix = rec.tfidf(data, col_contents)

# 코사인 유사도
similarity = rec.cos_similarity(data, col_contents)

# 추천 시스템
title = '인터스텔라' # col_title 칼럼의 값 중 하나 선택
recommendations = rec.recommend(title, data, col_title, col_contents)
```
##### 
### koda
------------
#### 한국어 데이터 증강을 위한 모듈
* synonym_replace(): 유의어 대체
```
koda.synonym_replace(sentence)
```
* random_insert(): 단어 랜덤 삽입
```
koda.random_insert(sentence)
```
* random_swap(): 단어 자리 바꾸기
```
koda.random_swap(sentence)
```
* random_delete(): 단어 랜덤 삭제
```
koda.random_delete(sentence)
```
###### 예시 코드
``` python
from babip import koda


# 예시 문장
sentence = '롯데 자이언츠 유니폼을 16년째 입고 있는 전준우는 포스트시즌 진출이 너무나도 간절하다. 그는 "올해 우리 팀이 가을야구에 진출하지 못하면 너무 아쉬울 것 같다"고 말했다.'

# 유의어 대체
sr = koda.synonym_replace(sentence)

# 단어 랜덤 삽입
ri = koda.random_insert(sentence)

# 단어 자리 바꾸기
rs = koda.random_swap(sentence)

# 단어 랜덤 삭제
rd = koda.random_delete(sentence)
```
### overfit
------------
#### 과적합 방지를 위한 모듈
* is_overfit(): 과적합 탐지
```
overfit.is_overfit(model, X_train, y_train, X_test, y_test)
```
###### 예시 코드
``` python
from babip import overfit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# 데이터 로드
data_path = './../data.csv'
data = pd.read_csv(data_path)

# 데이터 분할
X = data['data']
Y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=777)

# 모델 훈련
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 과적합 탐지
overfit_result = overfit.is_overfit(model, X_train, y_train, X_test, y_test)
```
### hyperopt
------------
#### 하이퍼파라미터 최적화를 위한 모듈
* grid_search(): Grid Search를 통해 하이퍼파라미터 최적화
```
hyperopt.grid_search(model, parameters, X_train, y_train)
```
* random_search(): Random Search를 통해 하이퍼파라미터 최적화
```
hyperopt.random_search(model, parameters, X_train, y_train)
```
###### 예시 코드
``` python
from babip import hyperopt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# 데이터 로드
data_path = './../data.csv'
data = pd.read_csv(data_path)

# 데이터 분할
X = data['data']
Y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=777)

# 모델 훈련
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 하이퍼파라미터 탐색 구간 설정
parameters = {
    'criterion': ['gini', 'entropy'], 
    'max_depth': [None, 2, 3, 4, 5, 6],
    'max_leaf_nodes': range(5, 101, 5)
}

# Grid Search를 통해 최적의 하이퍼파라미터 값 탐색
best_parameters_gs = hyperopt.grid_search(model, parameters, X_train, y_train)

# Random Search를 통해 최적의 하이퍼파라미터 값 탐색
best_parameters_rs = hyperopt.random_search(model, parameters, X_train, y_train)
```