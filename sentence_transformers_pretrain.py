#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datasets
# datasets load
oscar = datasets.load_dataset('oscar', 'unshuffled_deduplicated_en', split='train', streaming=True)


# In[1]:


import re
#corpus를 문장 단위로 전처리
splitter = re.compile(r'\.\s?\n?')


# In[2]:


#훈련 데이터 line단위로 읽어오기
with open('train.txt',encoding = 'utf-8') as f:
    dataset = f.readlines()


# In[3]:


num_sentences = 0  #sentence 개수 세기
sentences = []  # sentence를 리스트로 저장

for row in dataset: #데이터셋에 존재하는 각 행을
    new_sentences = splitter.split(row) #문장으로 나누어 새로 저장
    new_sentences = [line for line in new_sentences if len(line) > 10] #분리된 문장의 길이가 10이 넘을경우만 포함
    sentences.extend(new_sentences) #추가된 sentence를 extend
    num_sentences += len(new_sentences) #추가된 sentence의 개수만큼 추가


# In[4]:


len(sentences)


# In[5]:


from sentence_transformers.datasets import DenoisingAutoEncoderDataset # TSDAE방법을 이용한 비지도 학습에 이용되는 클래스
# 공식 repo참고 https://www.sbert.net/docs/package_reference/datasets.html?highlight=denoisingautoencoderdataset
from torch.utils.data import DataLoader
# 파이토치 DataLoader

#noise 기능이 포함된 데이터 클래스를 생성
train_data = DenoisingAutoEncoderDataset(sentences)

#기본 파이토치 DataLoader
loader = DataLoader(train_data, batch_size=2, shuffle=True, drop_last=True)


# In[6]:


from sentence_transformers import SentenceTransformer, models
# models는 다른 scratch로부터 SentenceTransformer네트워크를 생성하기위해 이용되는 block 구조를 정의, 
bert = models.Transformer('sentence-transformers/stsb-xlm-r-multilingual') # models로 부터 ST를 생성하기 위한 트랜스포머 모델 생성
pooling = models.Pooling(bert.get_word_embedding_dimension(), 'cls') # 모델로부터 word embedding의 차원 크기 Pooling연산 생성
model = SentenceTransformer(modules=[bert, pooling]) #트랜스포머 모델과 pooling 연산식을 이용하여 ST생성


# In[7]:


model


# In[8]:


from sentence_transformers.losses import DenoisingAutoEncoderLoss # noise로 부터 손상된 문장과 원본 문장의 오차를 계산
#공식 repo 참고 
#https://sbert.net/docs/package_reference/losses.html?highlight=denoisingautoencoderloss
loss = DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)


# In[ ]:


model.fit(
    train_objectives=[(loader, loss)],
    epochs=24,
    weight_decay=0,
    scheduler='constantlr',
    """
    매우 작은 정수로 그룹된 매개변수의 lr를 썩힘(사용하지 않는다는 뜻?) 
    각 에폭의 수가 사전에 정의된 milestone(중요한 단계: total_iters)에 도달하기 전까지  
    """
    optimizer_params={'lr': 3e-5},
    show_progress_bar=True,
    checkpoint_path = 'output/checkpoint',
    checkpoint_save_steps = 10000
)

model.save('output/your_model')


# In[ ]:


# 평가 데이터셋을 load
sts = datasets.load_dataset('glue', 'stsb', split='validation')
sts


# In[ ]:


# 평가 데이터셋을 샘플링
sts = sts.map(lambda x: {'label': x['label'] / 5.0})


# In[ ]:


from sentence_transformers import InputExample

samples = []
for sample in sts:
    # reformat to use InputExample
    # InputExample을 이용하기 위해 재구성하여 저장
    samples.append(InputExample(
        texts=[sample['sentence1'], sample['sentence2']],
        label=sample['label']
    ))


# In[ ]:


#임베딩 비교를 위한 evaluator생성
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    samples, write_csv=False
)


# In[ ]:


evaluator(model) #평가


# In[ ]:




