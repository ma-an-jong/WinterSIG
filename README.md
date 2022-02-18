# WinterSIG
겨울 SIG 기간동안 얻어간 코드나 내용들 정리함

## pytorch_pretraining_code.py
 pytorch를 이용하여 bert 모델을 pre-train하는 코드를 작성
 
 - ✔️ 허깅페이스 트렌스 포머 설치 !pip install -q git+https://github.com/huggingface/transformers.git
 - ModelDataArguments : 모델의 사전 학습에 필요한 data cofing를 저장한 클래스
 - get_model_config : 학습할 모델의 config를 가져옴(허깅페이스 모델의 config파일)
 - get_tokenizer : Tokenizer를 얻어옴 
 - get_model : Model을 얻어옴
 - get_dataset : 내가 가진 데이터셋을 pytorch Dataset클래스로 만들어줌(pretrain이므로 TextDataset,LineByLineTextDataset)
 - TrainingArguments : train시킬때 필요한 parameter들 설정
 - get_collator : pytorch의 DataLoader라고 생각 Dataset객체를 수집하는데 이용

## sentence_transformers_pretrain.py
 pytorch를 이용한 SentenceTransformer 모델 학습
 
 - corpus를 문장 단위로 split하여 저장한뒤 학습
 - 기존 입력 문장을 noise를 통해 손상시킨뒤 디코더가 original한 문장을 생성하는지 여부를 통한 학습을 진행
 - DenoisingAutoEncoderDataset : 기존 데이터에 noise를 포함시키는 데이터셋
 - DenoisingAutoEncoderLoss : 손상된 문장을 통한 생성과 origin을 비교한 loss 계산
 
# 코드 안의 주석으로 상세한 내용이 설명되어있음
