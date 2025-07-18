# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 17:19:05 2025

@author: jy129
"""

import cv2
from openvino.runtime import Core # OpenVINO 런타임 Core 라이브러리 임포트
import numpy as np # 이미지 전처리 및 후처리를 위해 numpy 임포트
import sys
import os

# --- 0. 설정: OpenVINO 모델 경로 및 입력 파일 ---
# !!! 여기에 당신의 OpenVINO 모델 파일 경로를 지정하세요 !!!
# model.xml 파일의 절대 또는 상대 경로를 입력해야 합니다.
MODEL_XML_PATH = "D:/torch/geti/flower_detect/deployment/Classification/model/model.xml"

# !!! 여기에 추론할 이미지 파일 경로를 지정하세요 !!!
# 예시: "D:/torch/geti/sample_image.jpg"
INPUT_IMAGE_PATH = "D:/torch/geti/flower_detect/sample_water_lily.jpg" 

# 추론할 디바이스 설정: "CPU", "GPU" 등
DEVICE = "GPU" 

# --- 1. OpenVINO 런타임 초기화 및 모델 로드 ---
print("--- 1. OpenVINO 런타임 초기화 및 모델 로드 시작 ---")

# OpenVINO 런타임 Core 객체 생성
ie = Core()

# 모델 로드 (XML 파일 경로 지정)
# model.bin 파일은 model.xml과 같은 디렉토리에 자동으로 로드됩니다.
try:
    ov_model = ie.read_model(model=MODEL_XML_PATH)
    print(f"모델 '{os.path.basename(MODEL_XML_PATH)}' 로드 성공.")
except Exception as e:
    print(f"오류: OpenVINO 모델을 로드할 수 없습니다. 경로를 확인하세요: {e}")
    sys.exit()

# 모델의 입력 및 출력 정보 확인 (전처리/후처리 시 필요)
input_layer = ov_model.input(0)
output_layer = ov_model.output(0)


# 입력 텐서의 기대 형태 (shape) 확인: (Batch, Channel, Height, Width) 또는 (Batch, Height, Width, Channel)
# Classification 모델은 보통 (Batch, Channel, Height, Width) 형태
input_shape = input_layer.shape
print(f"모델 입력 형태 (Input Shape): {input_shape}")
# 보통 분류 모델의 출력은 (Batch, Num_Classes) 형태
output_shape = output_layer.shape
print(f"모델 출력 형태 (Output Shape): {output_shape}")

# 모델을 지정된 디바이스에 컴파일하여 실행 가능한 형태로 준비
compiled_model = ie.compile_model(model=ov_model, device_name=DEVICE)
print(f"모델을 {DEVICE} 디바이스에 컴파일 완료.")

# --- 2. 입력 이미지 로드 및 전처리 ---
print("\n--- 2. 입력 이미지 로드 및 전처리 시작 ---")

# 이미지 로드 (OpenCV는 기본적으로 BGR 형식으로 로드)
image = cv2.imread(INPUT_IMAGE_PATH)
if image is None:
    print(f"오류: 이미지 파일 '{INPUT_IMAGE_PATH}'을(를) 로드할 수 없습니다.")
    exit()

print(f"원본 이미지 크기: {image.shape}")

# 모델의 입력 크기에 맞게 이미지 리사이즈 (예: (1, 3, 224, 224) 이면 224x224로 리사이즈)
# 입력 형태의 높이와 너비는 input_shape[2]와 input_shape[3]에 해당
h, w = input_shape[2], input_shape[3]
resized_image = cv2.resize(image, (w, h))

# BGR 이미지를 RGB로 변환 (많은 딥러닝 모델이 RGB를 선호)
image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# 이미지 픽셀 값을 0-1 범위로 정규화 (선택 사항, 모델 학습 시 전처리 방식에 따름)
# float32 타입으로 변환
input_data = image_rgb.astype(np.float32) / 255.0 

# 채널 순서 변경: (Height, Width, Channel) -> (Channel, Height, Width)
# OpenVINO 모델은 주로 NCHW (Batch, Channel, Height, Width) 형식을 요구
input_data = input_data.transpose((2, 0, 1)) 

# 배치 차원 추가: (Channel, Height, Width) -> (Batch, Channel, Height, Width)
# 여기서는 단일 이미지이므로 Batch=1
input_data = np.expand_dims(input_data, 0) 

print(f"전처리된 입력 데이터 형태: {input_data.shape}")

# --- 3. 추론 실행 ---
print("\n--- 3. 추론 실행 ---")

# 컴파일된 모델을 사용하여 추론 수행
# 입력은 딕셔너리 형태로, {입력_레이어: 입력_데이터}
try:
    results = compiled_model([input_data])
    print("추론 완료.")
except Exception as e:
    print(f"오류: 추론 중 문제가 발생했습니다: {e}")
    exit()

# --- 4. 추론 결과 후처리 ---
print("\n--- 4. 추론 결과 후처리 ---")

# 분류 모델의 출력은 일반적으로 소프트맥스(softmax)를 거치지 않은 로짓(logits) 값입니다.
# 따라서 확률로 변환하기 위해 소프트맥스 함수를 적용해야 합니다.
# (모델 학습 시 Softmax가 이미 포함되어 있다면 이 단계는 필요 없을 수 있습니다.
#  하지만 안전하게 적용하는 것이 좋습니다.)

# 로짓 결과를 넘파이 배열로 변환 (OpenVINO 결과는 openvino.runtime.Tensor 객체일 수 있음)
#output_logits = results[output_layer] # results는 보통 딕셔너리 또는 튜플/리스트 형태
output_logits = results[0] # 첫 번째 출력 레이어의 데이터를 인덱스 0으로 접근

# 소프트맥스 함수 정의
def softmax(x):
    e_x = np.exp(x - np.max(x)) # 오버플로우 방지
    return e_x / e_x.sum(axis=-1, keepdims=True)

# 소프트맥스 적용하여 확률 얻기
probabilities = softmax(output_logits[0]) # 첫 번째 배치 (단일 이미지)의 결과

# 클래스 이름 (이 부분은 당신의 모델에 맞게 수정해야 합니다)
# 예를 들어, 당신의 모델이 꽃 종류를 분류한다면, 그 꽃들의 이름 리스트를 만드세요.
# 'flower_detect'라는 이름으로 보아 꽃 분류 모델일 가능성이 높습니다.
CLASS_NAMES = ["Tulip", "Water Lily"] 
# 실제 클래스 이름 목록으로 교체하세요! (예: ["장미", "튤립", "백합"])

# 가장 높은 확률을 가진 클래스 찾기
predicted_class_id = np.argmax(probabilities)
predicted_confidence = probabilities[predicted_class_id]
predicted_class_name = CLASS_NAMES[predicted_class_id]

print(f"예측된 클래스: {predicted_class_name}")
print(f"예측 신뢰도: {predicted_confidence:.4f}")

# --- 5. 결과 시각화 (선택 사항) ---
print("\n--- 5. 결과 시각화 ---")

# 원본 이미지에 예측 결과 텍스트 오버레이
display_image = image.copy()
text = f"Predicted: {predicted_class_name} ({predicted_confidence:.2f})"
cv2.putText(display_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# 이미지 창 표시
cv2.imshow("OpenVINO Inference Result", display_image)
cv2.waitKey(0) # 아무 키나 누를 때까지 대기
cv2.destroyAllWindows() # 모든 OpenCV 창 닫기

print("\n--- 스크립트 실행 완료 ---")