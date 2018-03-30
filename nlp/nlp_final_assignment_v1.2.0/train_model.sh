
# 모든 기사를 하나의 텍스트로 저장, raw_text와 morphs_text가 생성됨
python save_whole_sentences.py

# 모델 학습
python train_model.py --size 100 200 300 --iter 5 10 --sg 0 1

