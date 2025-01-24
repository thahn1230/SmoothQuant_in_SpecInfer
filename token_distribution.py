# import torch
# import matplotlib.pyplot as plt
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from datasets import load_dataset
# from scipy.spatial.distance import jensenshannon

# import torch.nn as nn
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from smoothquant.smooth import smooth_lm
# from smoothquant.fake_quant import quantize_opt
# import tqdm
# from datasets import load_dataset

# # smoothquant 진행
# model_path = 'facebook/opt-6.7b'

# print("Loading model...")

# model = AutoModelForCausalLM.from_pretrained(
#     model_path, torch_dtype=torch.bfloat16, device_map="auto"
# )

# print("Loading activation scales and applying smoothquant...")
# act_scales = torch.load("./act_scales/opt-6.7b.pt")
# smooth_lm(model, act_scales, 0.5)
# model_smoothquant_w8a8 = quantize_opt(model)

# # 모델 및 토크나이저 로드
# model2_name = "facebook/opt-1.3b"
# tokenizer1 = AutoTokenizer.from_pretrained(model_path)
# tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
# model1 = model_smoothquant_w8a8
# # model1 = AutoModelForCausalLM.from_pretrained(model_path)
# model2 = AutoModelForCausalLM.from_pretrained(model2_name)

# # 데이터셋 로드 및 입력 프롬프트 선택
# print("Loading dataset...")
# # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# # prompt = dataset[0]["text"]
# prompt = "Robert Boulter is an English film , television and theatre actor ."

# # 토크나이징
# print("Tokenizing input prompt...")
# inputs1 = tokenizer1(prompt, return_tensors="pt")
# inputs2 = tokenizer2(prompt, return_tensors="pt")

# # 모델 예측 (logits 계산)
# print("Generating logits from models...")
# with torch.no_grad():
#     logits1 = model1(**inputs1).logits
#     logits2 = model2(**inputs2).logits

# # 확률 계산 (softmax)
# print("Calculating probabilities (softmax)...")

# # BFloat16을 float32로 변환 후 GPU에서 softmax 계산
# logits1 = logits1.to(torch.float32)  # 모델과 동일한 device로 이동
# logits2 = logits2.to(torch.float32) # 모델과 동일한 device로 이동

# probs1 = torch.softmax(logits1, dim=-1)  # 전체 시퀀스에 대해 softmax 계산
# probs2 = torch.softmax(logits2, dim=-1)  # 전체 시퀀스에 대해 softmax 계산

# # JS Divergence 계산
# print("Calculating JS Divergence...")
# def calculate_js_divergence(p, q):
#     p = np.array(p)
#     q = np.array(q)
#     return jensenshannon(p, q) ** 2

# # js_divergence = calculate_js_divergence(probs1, probs2)
# js_divergence = calculate_js_divergence(probs1[0, -1].cpu().numpy(), probs2[0, -1].cpu().numpy())  # 마지막 토큰만 비교
# print(js_divergence)

# # 토큰 별 확률 분포 비교 (vocab_size가 가로축, 확률이 세로축)
# vocab_size = probs1.size(-1)  # vocab_size = 마지막 차원 크기 (토큰 개수)
# tokens = np.arange(vocab_size)  # vocab_size 만큼의 인덱스

# # 토큰 인덱스를 실제 토큰으로 변환
# tokens_list1 = tokenizer1.convert_ids_to_tokens(tokens)
# tokens_list2 = tokenizer2.convert_ids_to_tokens(tokens)

# print("Saving token-by-token distribution graphs...")
# for i in range(probs1.size(1)):  # 각 토큰에 대해 반복
#     token_prob1 = probs1[0, i].cpu().numpy()  # 첫 번째 모델에서 예측한 확률
#     token_prob2 = probs2[0, i].cpu().numpy()  # 두 번째 모델에서 예측한 확률
    
#     # 그래프 생성 (가로축: vocab_size, 세로축: 각 단어의 확률)
#     plt.figure(figsize=(10, 6))
#     plt.plot(tokens, token_prob1, label=f"{model1.config._name_or_path} Token {i} Distribution", alpha=0.7)
#     plt.plot(tokens, token_prob2, label=f"{model2_name} Token {i} Distribution", alpha=0.7)
#     plt.title(f"Token {i} Probability Distribution")
#     plt.xlabel("Vocab Tokens")
#     plt.ylabel("Probability")
#     plt.legend()
#     # plt.xticks(rotation=90)  # x축 라벨이 겹치지 않도록 회전
#     # plt.grid(True)
    
#     plt.ylim(0, 1)

#     # 개별 토큰 분포 그래프 저장
#     plt.savefig(f"./results/w4a4/token_{i}_distribution.png")
#     plt.close()

# print("Process completed successfully. All graphs are saved.")


import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_opt

# smoothquant 진행
model_path = 'facebook/opt-6.7b'

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)

print("Loading activation scales and applying smoothquant...")
act_scales = torch.load("./act_scales/opt-6.7b.pt")
smooth_lm(model, act_scales, 0.5)
model_smoothquant_w8a8 = quantize_opt(model)

# 모델 및 토크나이저 로드
model2_name = "facebook/opt-1.3b"
tokenizer1 = AutoTokenizer.from_pretrained(model_path)
tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
model1 = model_smoothquant_w8a8
# model1 = AutoModelForCausalLM.from_pretrained(model_path)
model2 = AutoModelForCausalLM.from_pretrained(model2_name)

# 데이터셋 로드 (CNN/Daily Mail)
print("Loading dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:100]")  # 100개의 문장만 사용
prompts = dataset["article"]  # 'article' 열에 문장이 저장되어 있음

# JS Divergence 계산 함수
def calculate_js_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    return jensenshannon(p, q) ** 2

# JS Divergence 평균 계산
js_divergences = []

print("Processing 100 sentences...")
for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):  # tqdm 추가
    # 토크나이징
    inputs1 = tokenizer1(prompt, return_tensors="pt")
    inputs2 = tokenizer2(prompt, return_tensors="pt")

    # 모델 예측 (logits 계산)
    with torch.no_grad():
        logits1 = model1(**inputs1).logits
        logits2 = model2(**inputs2).logits

    # 확률 계산 (softmax)
    logits1 = logits1.to(torch.float32)  # 모델과 동일한 device로 이동
    logits2 = logits2.to(torch.float32)  # 모델과 동일한 device로 이동

    probs1 = torch.softmax(logits1, dim=-1)  # 전체 시퀀스에 대해 softmax 계산
    probs2 = torch.softmax(logits2, dim=-1)  # 전체 시퀀스에 대해 softmax 계산

    # 마지막 토큰만 비교하여 JS Divergence 계산
    js_divergence = calculate_js_divergence(probs1[0, -1].cpu().numpy(), probs2[0, -1].cpu().numpy())
    js_divergences.append(js_divergence)

# JS Divergence 평균 계산
average_js_divergence = np.mean(js_divergences)
print(f"Average JS Divergence for 100 sentences: {average_js_divergence}")
