# Open Domain Question Answering
네이버 부스트캠프 AI Tech 4기 NLP 6조 HAPPY팀의 MRC task repository입니다.  

## 목차
1. 프로젝트 개요
2. 팀원 소개
3. 파일 구성
4. How to Use
5. wrap-up report</br></br>


## 1.프로젝트 개요
MRC  task에서는  모델의  ODQA(Open-Domain  Question  Answering)  의  수행  능력을  평가하여  모델의  기계독해  성능을  측정합니다.  
 ODQA  모델은  two-stage로  구성됩니다.  
![image](https://user-images.githubusercontent.com/112468961/211195170-f0425396-82ef-41f6-bd16-3f2e56ec523b.png)  
 첫  번째  단계는  입력된  질문에  대해  관련된  문서를 찾아주는  retriever  model이고,  
 두  번째  단계인  reader  model에서는  retriever model이  전달한  context를  이용해  입력된  query에  대한  정답을  찾게  됩니다.</br>  
 대회 평가 기준은 EM(Exact Match) 와 micro F1 score 입니다.</br></br>


## 2.팀원 소개
- [박승현](https://github.com/koohack) : PM, Reader Model, Model Tuning, 결과 분석  
- [김준휘](https://github.com/intrandom5) : DPR, Retriever Model 
- [류재환](https://github.com/risolate) : 코드 리뷰어, Elastic Search  
- [박수현](https://github.com/HitHereX) : EDA, 전처리, Reader Model, 결과 분석  
- [설유민](https://github.com/ymnseol) : DPR, Retriever Model, Model Tuning  </br></br>

## 3.파일 구성
```
├──reader
|   ├──analysis
|   ├──model
|   |  ├──model_selection.py
|   |  └──models.py
|   ├──preprocessing
|   |  └──preprocessor.py
|   ├──arg.yaml.template
|   ├──inference.py
|   ├──test_arg.yaml.template
|   ├──train.py
|   └──utils_qa.py
|
└──retriever
    ├──dense_retriever
    |   ├──dataset
    |   |   ├──retriever_dataset.py
    |   |   └──utils.py
    |   ├──model
    |   |   └──dense_retriever.py
    |   ├──utils
    |   |   ├──seed.py
    |   |   └──topk.py
    |   ├──config.yaml.template
    |   ├──inferene.py
    |   ├──train.py
    |   └──validation.py
    ├──elasticsearch_retriever
    |   ├──arg.yaml.template
    |   ├──body.json
    |   ├──elastic.py
    |   └──inference.py
    └──sparse_retriever
        ├──BM25.py
        ├──config.yaml.template
        ├──inference.py
        └──tf_idf.py
```
**!!TODO:논의 사항!!**
- dataset+wikipedia 추가해 놓을지?
- TODO:if dataset upload, 저작권 마크 주의
- requirements.txt(add.for Elasticsearch) 추가해 놓을지? 
- TODO : if so, add requirements.txt on GitHub/code tree/readme-howtouse
- elasticsearch readme 지우고(add install elasticsaearch-8.5.3 on requirements.txt) tip 은 main readme로 이동 어떤찌?

## 4.How to Use
1. retriever model과 reader model을 따로 train 합니다
2. retriever model을 이용하여 wikipedia corpus에서 top-k passage를 inference 합니다
3. 2.의 top-k passage를 reader model에 넣어 최종 답안을 inference합니다.

### retriever model

`dense_retriever`, `sparse_retriever`, `elasticsearch_retriever` 3가지 retriever가 구현되어 있습니다.  
#### dense_retriever
`retriever/dense_retriever/config.yaml.template`을 참고하여 config를 설정할 수 있습니다.
- train : 아래 코드를 실행시켜 학습을 시작할 수 있습니다.
```
python3 retriever/dense_retriever/train.py --conf config.yaml
```
- validation : 아래 코드를 실행시켜 평가를 시작할 수 있습니다. 평가 기준은 top-k accuracy (k=5,10,20,50,100) 입니다 

```
python3 retriever/dense_retriever/validation.py --conf config.yaml
```
- inference : 아래 코드를 실행시켜 reader model이 정답을 찾기 적절한 passage를 추론할 수 있습니다.
```
python3 retriever/dense_retriever/inference.py --conf config.yaml
```
`retriever/dense_retriever/config.yaml.template` 에서 학습 및 추론 설정을 변경할 수 있습니다.</br></br>

#### sparse_retriever
아래 코드를 실행시켜 reader model이 정답을 찾기 적절한 passage를 추론할 수 있습니다. 별도의 학습 과정은 필요하지 않습니다.
```
python3 retriever/sparse_retriever/inference.py
```
`retriever/sparse_retriever/config.yaml.template` 에서 추론 설정을 변경할 수 있습니다</br></br>


#### elasticsearch_retriever
아래 코드를 실행시켜 reader model이 정답을 찾기 적절한 passage를 추론할 수 있습니다. 별도의 학습 과정은 필요하지 않습니다.
```
python3 retriever/elasticsearch_retriever/inference.py
```
`retriever/sparse_retriever/arg.yaml.template` 에서 추론 설정을 변경할 수 있습니다.   
연결 문제가 생긴다면 elasticsearch-8.5.3/config/elasticsearch.yml파일에서 xpack.security관련 옵션들을 false로 해주어야 합니다.</br></br>



---
### reader model
#### train
아래 코드를 실행시켜 학습을 시작할 수 있습니다
```
python3 reader/train.py
```
`reader/arg.yaml.template` 에서 학습 설정을 변경할 수 있습니다. 훈련 및 평가는 동시에 진행됩니다. 평가는 EM/micro F1으로 진행됩니다.  </br></br>
#### inference
아래 코드를 실행시켜 retriever model이 추론한 passage를 이용, 정답에 대한 추론을 시작할 수 있습니다.
```
pythhon3 reader/inference.py
```
`reader/test_arg.yaml.template` 에서 추론 설정을 변경할 수 있습니다.</br></br>
## 5.wrap-up report
[MRC_NLP_팀 리포트(06조)_final.pdf](https://github.com/boostcampaitech4lv23nlp1/level2_mrc_nlp-level2-nlp-06/files/10369961/MRC_NLP_.06._final.pdf)
