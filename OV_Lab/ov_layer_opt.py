import tensorflow as tf
import numpy as np
import os
import xml.etree.ElementTree as ET
import shutil 
import openvino as ov # 

# 1. TensorFlow 모델 생성 및 학습

print("1. Tensorflow model 생성 및 학습")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], activation=None)
])

model.compile(optimizer='sgd', loss='mean_squared_error') # 학습전 설정 단계

x_train = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
y_train = np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float32)

model.fit(x_train, y_train, epochs=50, verbose=0)

saved_model_path = "./my_tf_model"
tf.saved_model.save(model, saved_model_path)
print(f"2. TensorFlow 모델이 다음 경로에 저장되었습니다: {saved_model_path}\n")

print(f"\n3. --- openvino.convert_model() 파이썬 API 사용 ---\n")

ir_output_dir_api = "./my_optimized_model_ir_api"

if os.path.exists(ir_output_dir_api): 
    shutil.rmtree(ir_output_dir_api)
os.makedirs(ir_output_dir_api, exist_ok=True) 

print(f"3.OpenVINO Model Converter  파이썬 API 실행 중...\n")

ov_model = ov.convert_model(saved_model_path, input=[[1, 1]])  # batch 1개당 1개의 feature

# IR 저장
output_xml_path_api = os.path.join(ir_output_dir_api, "model_api.xml")
ov.save_model(ov_model, output_xml_path_api)
print(f"OpenVINO IR 파일이 다음 경로에 생성되었습니다: {output_xml_path_api}\n")

# IR XML 파일 파싱
if os.path.exists(output_xml_path_api):
    print(f"OpenVINO IR (XML) 파일 내용 분석 (파이썬 API 변환):\n")
    tree = ET.parse(output_xml_path_api)
    root = tree.getroot()
    print("--- OpenVINO IR 노드 (Layers) 목록 (파이썬 API 변환) ---")
    for layer in root.findall(".//layer"):
        layer_id = layer.get('id')
        layer_name = layer.get('name')
        layer_type = layer.get('type')
        print(f"  ID: {layer_id}, Name: {layer_name}, Type: {layer_type}")
else:
    print(f"오류: {output_xml_path_api} 파일을 찾을 수 없습니다. 파이썬 API 변환이 실패했거나 경로가 잘못되었습니다.")

