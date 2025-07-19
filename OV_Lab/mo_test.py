import tensorflow as tf
import numpy as np # NumPy 라이브러리 임포트
import os

# 예시: 간단한 Keras 모델 생성 및 학습
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer='sgd', loss='mean_squared_error')

# 파이썬 리스트를 NumPy 배열로 변환
x_train = np.array([1.0, 2.0, 3.0, 4.0])
y_train = np.array([2.0, 4.0, 6.0, 8.0])

# NumPy 배열을 model.fit()에 전달
model.fit(x_train, y_train, epochs=5)

# 모델을 SavedModel 형식으로 저장 (디렉토리 생성)
saved_model_path = "./my_saved_model"
tf.saved_model.save(model, saved_model_path)

print(f"모델이 다음 경로에 저장되었습니다: {saved_model_path}")
print("내용:")
os.system(f"ls -R {saved_model_path}")