## NEWSUM의 딥러닝 모델: 기사 원문 요약 모델과 해시태그 분류 모델 (어플리케이션용)

### NEWSUM의 2가지 딥러닝 모델 구조와 흐름
  ![image](https://user-images.githubusercontent.com/67627471/170497767-a23efa2e-f771-41c3-af8d-39e2c90126cc.png)

##

### 사용 데이터
* 주요 언론 3사에서 정치/경제/사회 분야의 실시간 인기 기사를 크롤링해 사용
* 해시 태그 종류: ‘산업’, ‘금융/경제’, ‘사건/법’, ‘외교/해외’, ‘정당/선거’, ’고용/근로’, ‘의료’, ‘교육/복지’, ‘주거/부동산’, ‘환경’, ‘교통/안전’, ‘역사/문화’  

##

### 구현 모델
* 사용 언어: python
* 사용 라이브러리: pytorch, KoNLPy, Gensim, pandas, NumPy 등
#### 요약 모델
* KorBERT에 summarize layer 더해 fine-tuning
* 사용 언어: python
#### 분류 모델
* TextCNN
* 관련 논문: https://arxiv.org/pdf/1408.5882.pdf
* 주의: 원활한 torchtext 사용을 위해 torch1.9.0+cu111 버전으로 구현하였음

##

### 폴더 설명
* 해당 내용은 딥러닝 서버에 올려 실행하는 파일들로, 학습된 딥러닝 모델을 불러와 실시간으로 크롤링해오는 기사 원문을 input하여 결과를 받아오고
  DB에 저장하는 내용까지 구현되어 있음
* content:
  * 학습된 요약 모델 불러올 때 필요한 파일들
* hashtag-classifier/data:
  * 학습된 모델 파일과 word2vec으로 임베딩한 토큰 파일
* 그 외:
  * classifier_TextCNN.py: 학습된 해시태그 분류 모델을 불러와 predict하는 파일
  * dbdbdeep.py: DB에 저장하는 파일
  * deep.py: 요약 모델을 불러와 predict하여 결과를 내고 해시태그 분류 모델의 predict 결과와 함께 DB에 저장하는 최종 파일
  * makebow.py: 크롤링한 기사 원문을 분류 모델에 넣기 전 bag of words를 생성하는 파일

##

### 모델 predict 및 DB 저장 방법
* 서버에서 터미널에 python deep.py 명령어로 deep.py를 실행하면 됨
