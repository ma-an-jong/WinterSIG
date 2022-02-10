#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#허깅페이스 트랜스포머 설치
get_ipython().system('pip install -q git+https://github.com/huggingface/transformers.git')


# In[ ]:


import io
import os
import math
import torch
import warnings
from tqdm.notebook import tqdm
from transformers import (
                          CONFIG_MAPPING,
                          MODEL_FOR_MASKED_LM_MAPPING,
                          MODEL_FOR_CAUSAL_LM_MAPPING,
                          PreTrainedTokenizer,
                          TrainingArguments,
                          AutoConfig,
                          AutoTokenizer,
                          AutoModelWithLMHead,
                          AutoModelForCausalLM,
                          AutoModelForMaskedLM,
                          LineByLineTextDataset,
                          TextDataset,
                          DataCollatorForLanguageModeling,
                          DataCollatorForWholeWordMask,
                          DataCollatorForPermutationLanguageModeling,
                          PretrainedConfig,
                          Trainer,
                          set_seed,
                          )
 
set_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


device


# In[ ]:


#존재하는 dataset 가져오기
with open('train.txt','r',encoding = 'utf-8') as f:
    data = f.readlines()


# In[ ]:


data[0:5]


# In[ ]:


class ModelDataArguments(object):
  r"""
    모델을 정의하고 사전학습에 사용할 데이터 config 작성
    
   모든 args는 optional, 하지만 특정 숫자나 값을 요구하는 경우 존재
   ex) bert-base layer개수 , 은닉층 개수 , max_seq_len 통일 등
   
  args:
    train_data_file (:obj:`str`, `optional`): 훈련할 데이터의 위치
    파일이 line단위로 이루어 져있다면 line_by_line=True로 설정
    만약 데이터 파일이 특별한 묶음으로 이루어져있지 않으면 False로 설정
    default로 None으로 설정
 
    eval_data_file (:obj:`str`, `optional`): 평가 데이터의 위치
    train_data_file과 동일
    default로 None으로 설정
 
    line_by_line (:obj:`bool`, `optional`, defaults to :obj:`False`): 
      학습 데이터셋과 검증 데이터셋이 라인단위로 구분된다면 
      구분이 없다면 False로 지정
 
    mlm (:obj:`bool`, `optional`, defaults to :obj:`False`): 
    모델 구조에 의존적인 손실함수를 변경할지 정하는 flag
    이 변수는 masked language model의 경우 True로 설정해야 하며 다른 경우는 False로 지정
    이렇게 설정하지 않으면 ValueError를 일으키는 함수가 설정되어있음
 
    whole_word_mask (:obj:`bool`, `optional`, defaults to :obj:`False`):
    전체 단어 마스킹 사용 여부를 나타내는 flag
    전체 단어 마스킹이란 학습중 모든 단어가 masking이 될수있는지
 
    mlm_probability(:obj:`float`, `optional`, defaults to :obj:`0.15`): 
    masked language model 이 학습중에 사용
    mlm=True일때 필요로함
    마스킹되는 token의 비율을 표기
 
    plm_probability (:obj:`float`, `optional`, defaults to :obj:`float(1/6)`): 
      permutation language modeling을 위한 context를 감싸고 있는 마스킹된 토큰의 폭의 길이의 비율을 정의
      XLNet을 사용할때 필요
 
    max_span_length (:obj:`int`, `optional`, defaults to :obj:`5`): 
    permutation language modeling에서 최대 마스킹된 토큰의 길이 제한을 설정
    XLNet을 이용할때 필요.
 
    block_size (:obj:`int`, `optional`, defaults to :obj:`-1`): 
      텍스트 파일에서 이동하는 윈도우 사이즈를 나타냄
      -1 ~ max_seq_len 까지 사용이 가능

    overwrite_cache (:obj:`bool`, `optional`, defaults to :obj:`False`): 
      다른 캐시파일이 존재한다면 덮어쓰기
 
    model_type (:obj:`str`, `optional`): 
      Type of model used: bert, roberta, gpt2. 
      More details: https://huggingface.co/transformers/pretrained_models.html
      default로 None 설정
 
    model_config_name (:obj:`str`, `optional`):
      Config of model used: bert, roberta, gpt2. 
      More details: https://huggingface.co/transformers/pretrained_models.html
      default로 None 설정
 
    tokenizer_name: (:obj:`str`, `optional`)
      반드시 학습시킬 모델과 같은 이름으로 tokenizer호출 해야됨 
      It usually has same name as model_name_or_path: bert-base-cased, 
      roberta-base, gpt2 etc.
      default로 None 설정.
 
    model_name_or_path (:obj:`str`, `optional`): 
      Path to existing transformers model or name of 
      transformer model to be used: bert-base-cased, roberta-base, gpt2 etc. 
      More details: https://huggingface.co/transformers/pretrained_models.html
      default로 None 설정
 
    model_cache_dir (:obj:`str`, `optional`): 
      재 실행 할때 빠른 전개를 위한 .cache 디렉토리 경로
 
  Raises:
        ValueError: If `CONFIG_MAPPING` is not loaded in global variables.
        ValueError: If `model_type` is not present in `CONFIG_MAPPING.keys()`.
        ValueError: If `model_type`, `model_config_name` and 
          `model_name_or_path` variables are all `None`. At least one of them 
          needs to be set.
          
        warnings: If `model_config_name` and `model_name_or_path` are both 
          `None`, the model will be trained from scratch.
          
        ValueError: If `tokenizer_name` and `model_name_or_path` are both 
          `None`. We need at least one of them set to load tokenizer.
  """
 
  def __init__(self, 
               train_data_file=None,
               eval_data_file=None, 
               line_by_line=False,
               mlm=False,
               mlm_probability=0.15, 
               whole_word_mask=False,
               plm_probability=float(1/6), 
               max_span_length=5,
               block_size=-1,
               overwrite_cache=False, 
               model_type=None,
               model_config_name=None,
               tokenizer_name=None, 
               model_name_or_path=None,
               model_cache_dir=None):
     
    # CONFIG_MAPPING이 transformer로 부터  import가 되었는지 확인
    if 'CONFIG_MAPPING' not in globals():
      raise ValueError('Could not find `CONFIG_MAPPING` imported! Make sure'                        ' to import it from `transformers` module!')
 
    # model_type이 유효한지 확인
    if (model_type is not None) and (model_type not in CONFIG_MAPPING.keys()):
      raise ValueError('Invalid `model_type`! Use one of the following: %s' %
                       (str(list(CONFIG_MAPPING.keys()))))
       
    # model_type, model_config_name and model_name_or_path 등이 None으로 설정되었는지 확인
    if not any([model_type, model_config_name, model_name_or_path]):
      raise ValueError('You can`t have all `model_type`, `model_config_name`,'                        ' `model_name_or_path` be `None`! You need to have'                        'at least one of them set!')
    
    # scratch로 부터 새로운 모델이 Load 되는지 확인
    if not any([model_config_name, model_name_or_path]):
      # 경고창 설정
      warnings.formatwarning = lambda message,category,*args,**kwargs:                                '%s: %s\n' % (category.__name__, message)
      #  경고 출력
      warnings.warn('You are planning to train a model from scratch! 🙀')
 
    # 새로운 Tokenizer가 생성되는지 확인 , but 제대로 지원하지 않음
    if not any([tokenizer_name, model_name_or_path]):
      #scratch 한 tokenizer는 학습시킬수 없어서 오류를 발생시킴
      raise ValueError('You want to train tokenizer from scratch! '                     'That is not possible yet! You can train your own '                     'tokenizer separately and use path here to load it!')
       
    # agrs 설정
    self.train_data_file = train_data_file
    self.eval_data_file = eval_data_file
    self.line_by_line = line_by_line
    self.mlm = mlm
    self.whole_word_mask = whole_word_mask
    self.mlm_probability = mlm_probability
    self.plm_probability = plm_probability
    self.max_span_length = max_span_length
    self.block_size = block_size
    self.overwrite_cache = overwrite_cache
 
    # model,tokenizer 매개변수 설정
    self.model_type = model_type
    self.model_config_name = model_config_name
    self.tokenizer_name = tokenizer_name
    self.model_name_or_path = model_name_or_path
    self.model_cache_dir = model_cache_dir
     
    return


# In[ ]:


# data 설정 arguments
model_data_args = ModelDataArguments(
                                    train_data_file='train.txt', 
                                    eval_data_file='test.txt', 
                                    line_by_line=False, 
                                    mlm=True,
                                    whole_word_mask=True,
                                    mlm_probability=0.15,
                                    plm_probability=float(1/6), 
                                    max_span_length=5,
                                    block_size=50,
                                    overwrite_cache=False,
                                    model_type='bert',
                                    model_config_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 
                                    tokenizer_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 
                                    model_name_or_path='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 
                                    model_cache_dir=None,
                                    )


# In[ ]:


def get_model_config(args: ModelDataArguments):
  r"""

  학습할 모델의 config를 가져옴

  ModelDataArgument를 사용하여 model config를 요청하여 가져옴
  
  매개변수:
    args (:obj:`ModelDataArguments`):
      모델,데이터 config 매개변수들은 사전 학습에 필요
  Returns:
    :obj:`PretrainedConfig`: Model_transformers_configuration. ex.bert_config
     
  Raises:
    ValueError: If `mlm=True` and `model_type` is NOT in ["bert", 
          "roberta", "distilbert", "camembert"]. We need to use a masked 
          language model in order to set `mlm=True`.
          
          masked language model = True인데 모델이 bert,distil,camem 등 마스크 언어 모델의 파생이 아니면 안됨
  """
 
  # model config 체크
  if args.model_config_name is not None:
        
    # 1.cofig로 부터 이름을 얻어와서 사용
    model_config = AutoConfig.from_pretrained(args.model_config_name, 
                                      cache_dir=args.model_cache_dir)
 
  elif args.model_name_or_path is not None:
    # 2.config로 부터 경로를 얻어와서 사용
    model_config = AutoConfig.from_pretrained(args.model_name_or_path, 
                                      cache_dir=args.model_cache_dir)
 
  else:
    # 만약 scratch한 모델이라면 config mapping을 사용
    model_config = CONFIG_MAPPING[args.model_type]()
 
  # mlm옵션이 MASKED LANGUAGE MODEL을 위한 설정이 되어있는지 확인
  if (model_config.model_type in ["bert", "roberta", "distilbert", 
                                  "camembert"]) and (args.mlm is False):
    raise ValueError('BERT and RoBERTa-like models do not have LM heads '                     'butmasked LM heads. They must be run setting `mlm=True`')
   
  # xlnet이라면 맞는 block_size를 설정
  if model_config.model_type == "xlnet":
    # xlnet은 사전 학습에 block_size = 512로 설정
    args.block_size = 512
    # setup memory length
    model_config.mem_len = 1024
   
  return model_config


# In[ ]:


# model configuration.
print('Loading model configuration...')
config = get_model_config(model_data_args)


# In[ ]:


config


# In[ ]:


def get_tokenizer(args: ModelDataArguments):
  r"""
  Tokenizer 얻어오는 메소드

  ModelArguments를 이용하여 모델의 tokenizer를 생성 및 block_size등, args를 변경 할 수 있음

  매개변수:
    args (:obj:`ModelDataArguments`):
      Model and data configuration arguments needed to perform pretraining.
      모델과 data config arguments는 사전학습에 필요로 함
      
  Returns:
    :obj:`PreTrainedTokenizer`: Model transformers tokenizer.
  """

  # tokenizer config 체크
  if args.tokenizer_name:
    # 1.tokenizer의 이름으로 생성
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, 
                                              cache_dir=args.model_cache_dir)
 
  elif args.model_name_or_path:
    # 2.tokenizer의 경로로 생성
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, 
                                              cache_dir=args.model_cache_dir)
     
  # data의 block_size를 설정
  if args.block_size == 0:
    # tokenizer의 최대 길이로 block_size를 설정
    # 입력 block_size는 model이 입력의 최대치까지 가능
    # 가끔 max_lengths 가 매우 크면 문제가 발생가능
    args.block_size = tokenizer.model_max_length
  else:
    # tokenizer의 max_length를 넘어서지 말것
    args.block_size = min(args.block_size, tokenizer.model_max_length)
 
  return tokenizer


# In[ ]:


tokenizer = get_tokenizer(model_data_args)


# In[ ]:


tokenizer


# In[ ]:


def get_model(args: ModelDataArguments, model_config):
  r"""
  모델 가져오기
  ModelDataArguments를 이용하여 실제 모델을 리턴
  
  매개변수:
    args (:obj:`ModelDataArguments`):
      모델과 data config 매개변수는 사전학습에 필요
      
    model_config (:obj:`PretrainedConfig`):
      Model transformers configuration.
      
  Returns:
    :obj:`torch.nn.Module`: ★PyTorch★ model.
 
  """
 

  # MODEL_FOR_MASKED_LM_MAPPING 과 MODEL_FOR_CAUSAL_LM_MAPPING 이 transformers 로부터 import 되어야함
  if ('MODEL_FOR_MASKED_LM_MAPPING' not in globals()) and                 ('MODEL_FOR_CAUSAL_LM_MAPPING' not in globals()):
    raise ValueError('Could not find `MODEL_FOR_MASKED_LM_MAPPING` and'                      ' `MODEL_FOR_MASKED_LM_MAPPING` imported! Make sure to'                      ' import them from `transformers` module!')
     

  # 사전 학습 모델 사용 또는 scratch로 부터 훈련인지 확인
  if args.model_name_or_path:
    # 사전 학습 모델을 사용하는 경우
    if type(model_config) in MODEL_FOR_MASKED_LM_MAPPING.keys():
      # Masked language modeling head.
      return AutoModelForMaskedLM.from_pretrained(
                        args.model_name_or_path,
                        from_tf=bool(".ckpt" in args.model_name_or_path),
                        config=model_config,
                        cache_dir=args.model_cache_dir,
                        )
    elif type(model_config) in MODEL_FOR_CAUSAL_LM_MAPPING.keys():
      # Causal language modeling head.
      return AutoModelForCausalLM.from_pretrained(
                                          args.model_name_or_path, 
                                          from_tf=bool(".ckpt" in
                                                        args.model_name_or_path),
                                          config=model_config, 
                                          cache_dir=args.model_cache_dir)
    else:
      raise ValueError(
          'Invalid `model_name_or_path`! It should be in %s or %s!' %
          (str(MODEL_FOR_MASKED_LM_MAPPING.keys()), 
           str(MODEL_FOR_CAUSAL_LM_MAPPING.keys())))
     
  else:
    # Use model from configuration - train from scratch.
      print("Training new model from scratch!")
      return AutoModelWithLMHead.from_config(config)


# In[ ]:


print('Loading actual model...')
model = get_model(model_data_args, config)


# In[ ]:


def get_dataset(args: ModelDataArguments, tokenizer: PreTrainedTokenizer, 
                evaluate: bool=False):
  r"""
  데이터셋을 pytorch Dataset으로 만들어줌
 
  Using the ModelDataArguments return the actual model.
  ModelDataArguments을 사용하여 실제 model을 return
  
  args: 
    args (:obj:`ModelDataArguments`):

    tokenizer (:obj:`PreTrainedTokenizer`):
      
    evaluate (:obj:`bool`, `optional`, defaults to :obj:`False`):
      If set to `True` the test / validation file is being handled.
      True = 테스트/검증 세트 False = 훈련 세트
  Returns:
    파일 데이터를 가진 Pytorch Dataset
  """

  # train 또는 evaluate 파일 경로를 얻어옴
  file_path = args.eval_data_file if evaluate else args.train_data_file
 
  # line_by_line이 True인지 확인
  if args.line_by_line:
    #데이터 파일에 존재하는 각 example은 각 라인임
    return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, 
                                 block_size=args.block_size)
     
  else:
    #파일내 모든 데이터는 분리 없이 함께 존재
    return TextDataset(tokenizer=tokenizer, file_path=file_path, 
                       block_size=args.block_size, 
                       overwrite_cache=args.overwrite_cache)


# In[ ]:


# 학습에 필요한 arguments를 설정
"""
Note: 

I only used the arguments I care about.
`TrainingArguments` contains a lot more arguments.

For more details check the awesome documentation:

https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments
"""

training_args = TrainingArguments(
                       #체크포인트와 모델 prediction이 쓰여질 디렉토리 경로 설정
                         output_dir='pretrain_kobert',
                       #출력 디렉토리의 content를 덮어쓰기 할지 설정
                         overwrite_output_dir=True,
   
                       #trainning,evaluate을 할것인가
                         do_train=True, 
                         do_eval=True, 
   
                       # Batch size GPU/TPU core/CPU training,evaluation.
                         per_device_train_batch_size=16,
                         per_device_eval_batch_size=100,
   
                         # training도중 evaluation 적용 방법
                             # `no`: 학습도중 평가를 하지 않는다.
                             # `steps`: `eval_steps` 마다 평가를 진행.
                             # `epoch`: 매 epoch이 끝날때마다 평가.
                         evaluation_strategy='steps',

                       #얼마나 자주 log를 띄울것인가, loss와 perplexity의 기록
                         logging_steps=2000,
   
                       #만약 evaluationg_strategy = "steps"이면 eval과 eval간의 update steps의 개수
                       #설정을 하지 않는다면 logging_step과 같은 값으로 설정
                         eval_steps = None,
                          
                       #True로 설정시 perplexity 계산을 위한 손실을 출력
                         prediction_loss_only=True,

                       #Adam을 위한 learning_rate를 설정 기본값 5e-5
                         learning_rate = 5e-5,

                       #Weight decay는 학습된 모델의 복잡도를 줄이기 위해서
                       #학습 중 weight가 너무 큰 값을 가지지 않도록 
                       #Loss function에 Weight가 커질경우에 대한 패널티를 적용
                         weight_decay=0,
   
                       #epsilon은 Adam optimizer에서 분모가 0이 되지 않도록 하기 위한 작은 값 
                       # 기본값은 1e-8
                         adam_epsilon = 1e-8,

                       # 그래디언트 clipping을 위한 최대 그래디언트 norm
                       # 기본값은 0
                         max_grad_norm = 1.0,
   
                       # epoch
                         num_train_epochs = 3,

                       # 체크포인트 간격  
                       # 기본값 500
                         save_steps = 2000,
                         )


# In[ ]:


# train dataset 설정
print('Creating train dataset...')
train_dataset = get_dataset(model_data_args, tokenizer=tokenizer, evaluate=False) if training_args.do_train else None
 
# eval dataset 설정
print('Creating evaluate dataset...')
eval_dataset = get_dataset(model_data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None


# In[ ]:


train_dataset[0:10]


# In[ ]:


def get_collator(args: ModelDataArguments, model_config: PretrainedConfig, 
                 tokenizer: PreTrainedTokenizer):
  r"""
  적합한 collator 함수를 가져옴
  Collator 함수는 Pytorch Dataset 객체를 수집하는데 이용됨
  
  매개변수:
    args (:obj:`ModelDataArguments`):
      Model 과 data config 매개변수는 사전학습에 필요함
      
    model_config (:obj:`PretrainedConfig`):
      Model transformers configuration.
      
    tokenizer (:obj:`PreTrainedTokenizer`):
      Model transformers tokenizer.

  Returns:
    :obj:`data_collator`: Transformers specific data collator.
 
  """
 
  # Special dataset handle depending on model type.
  # 특별한 dataset은 모델 type에 의존하여 관리함
    
  if model_config.model_type == "xlnet":
    # XLNET를 위한 collator
    return DataCollatorForPermutationLanguageModeling(
                                          tokenizer=tokenizer,
                                          plm_probability=args.plm_probability,
                                          max_span_length=args.max_span_length,
                                          )
  else:
    # 나머지 model을 위한 data
    if args.mlm and args.whole_word_mask:
      # 전체 단어 마스킹을 사용
      return DataCollatorForWholeWordMask(
                                          tokenizer=tokenizer, 
                                          mlm_probability=args.mlm_probability,
                                          )
    else:
      # 일반적인 언어 모델
      return DataCollatorForLanguageModeling(
                                          tokenizer=tokenizer, 
                                          mlm=args.mlm, 
                                          mlm_probability=args.mlm_probability,
                                          )


# In[ ]:


data_collator = get_collator(model_data_args, config, tokenizer)


# In[ ]:


# Trainer 생성
print('Loading `trainer`...')
trainer = Trainer(model=model,
                  args=training_args,
                  data_collator=data_collator,
                  train_dataset=train_dataset,
                  eval_dataset=eval_dataset,
                  )
 
 

 # 저장을 위한 모델 경로
if training_args.do_train:
  print('Start training...')
 
  # 훈련할 모델이local path를 통해 로딩되었다면 model경로를 설정
  model_path = (model_data_args.model_name_or_path 
                if model_data_args.model_name_or_path is not None and
                os.path.isdir(model_data_args.model_name_or_path) 
                else None
                )
  # 학습 시작
  trainer.train(model_path=model_path)
  # 모델 저장
  trainer.save_model()
 
  # 편의를 위해서 우리는 tokenizer를 같은 경로에 다시 저장함
  # 그래서 학습한 모델을 쉽게 공유가 가능
  if trainer.is_world_process_zero():
    tokenizer.save_pretrained(training_args.output_dir)


# In[ ]:


# do_eval이 True인지 확인
if training_args.do_eval:
    
# trainer의 평가 결과
  eval_output = trainer.evaluate()

#모델의 loss를 통해 perplexity를 계산
  perplexity = math.exp(eval_output["eval_loss"])
  print('\nEvaluate Perplexity: {:10,.2f}'.format(perplexity))

else:
  print('No evaluation needed. No evaluation data provided, `do_eval=False`!')


# In[ ]:


trainer.save_model()


# In[ ]:


if trainer.is_world_process_zero():
    tokenizer.save_pretrained(training_args.output_dir)


# In[ ]:




