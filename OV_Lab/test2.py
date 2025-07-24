# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 12:56:56 2025

@author: jy129
"""

import torch
import torch.nn as nn
import os
import shutil
from openvino import convert_model, save_model, Core, Layout, Type, PartialShape
import numpy as np

# --- 1. PyTorch 모델 정의 (학습 과정 없이, 고정된 가중치/편향) ---
print("--- 1. PyTorch 모델 정의 (최소 버전) ---")

class MinimalModel(nn.Module):
    def __init__(self):
        super(MinimalModel, self).__init__()
        # nn.Linear는 기본적으로 무작위 초기화되므로,
        # 여기에 고정된 값을 직접 설정하여 학습 과정을 생략합니다.
        self.linear = nn.Linear(1, 1) # 입력 1개, 출력 1개
        
        # 가중치와 편향을 수동으로 설정 (예: y = 2x + 5)
        self.linear.weight.data.fill_(2.0) # 가중치를 2.0으로 설정
        self.linear.bias.data.fill_(5.0)   # 편향을 5.0으로 설정

    def forward(self, x):
        return self.linear(x)

pytorch_model = MinimalModel()
print("PyTorch 모델 정의 완료 (가중치/편향 고정).")
print(f"고정된 가중치: {pytorch_model.linear.weight.item():.1f}")
print(f"고정된 편향: {pytorch_model.linear.bias.item():.1f}")

# --- 2. PyTorch 모델을 ONNX 형식으로 내보내기 ---
print("\n--- 2. PyTorch 모델을 ONNX 형식으로 내보내기 ---")
onnx_model_path = "minimal_model.onnx"
dummy_input = torch.randn(1, 1, dtype=torch.float32) # 더미 입력 (배치 1, 특성 1)

try:
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        onnx_model_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )
    print(f"PyTorch 모델이 ONNX 형식으로 성공적으로 내보내졌습니다: {onnx_model_path}")
except Exception as e:
    print(f"ONNX 내보내기 중 오류 발생: {e}")
    exit()

# --- 3. ONNX 모델을 OpenVINO IR로 변환 ---
print("\n--- 3. ONNX 모델을 OpenVINO IR로 변환 ---")
ir_output_dir = "./my_ov_minimal_ir"
if os.path.exists(ir_output_dir):
    shutil.rmtree(ir_output_dir)
os.makedirs(ir_output_dir, exist_ok=True)

try:
    # ONNX 모델을 OpenVINO IR로 변환
    # input 인자에 [PartialShape([1, 1])] 형태로 입력 형태를 명시
    ov_model = convert_model(onnx_model_path, input=[PartialShape([1, 1])]) 
    
    # IR 파일 저장 (XML 및 BIN)
    output_xml_path = os.path.join(ir_output_dir, "minimal_model.xml")
    save_model(ov_model, output_xml_path) # True (FP16 압축)는 기본값이 아니므로 제거
    
    print(f"OpenVINO IR 파일이 다음 경로에 성공적으로 생성되었습니다: {output_xml_path}")

except Exception as e:
    print(f"OpenVINO IR 변환 중 오류 발생: {e}")
    exit()

# --- 4. 변환된 OpenVINO IR 모델로 추론 ---
print("\n--- 4. 변환된 OpenVINO IR 모델로 추론 ---")
core = Core()

try:
    compiled_model = core.compile_model(output_xml_path, "CPU")
    test_input = np.array([[10.0]], dtype=np.float32) # 테스트 입력
    
    print(f"테스트 입력: {test_input.item()}")

    results = compiled_model([test_input])
    output_data = results['output'].item() 
    
    # 기대 값 계산 (y = 2x + 5 이므로 2*10 + 5 = 25)
    expected_output = 2.0 * test_input.item() + 5.0
    
    print(f"OpenVINO 모델 예측: {output_data:.4f}")
    print(f"기대되는 값: {expected_output:.4f}")

except Exception as e:
    print(f"OpenVINO IR 모델 추론 중 오류 발생: {e}")

print("\n스크립트 실행 완료.")