from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# 모델 로드 함수
def load_model():
    with open('generator_model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    generator = tf.keras.models.model_from_json(loaded_model_json)
    generator.load_weights('generator_weights.weights.h5')
    return generator

# 이미지 전처리 함수
def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))
    image = np.array(image)
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    return image

# 이미지 후처리 및 저장 함수
def save_image(image_array, output_path):
    image = (image_array + 1) * 127.5  # Denormalize to [0, 255]
    image = np.array(image, dtype=np.uint8)
    Image.fromarray(image).save(output_path)

# 예측 함수
def predict_and_save(input_image_path, output_image_path):
    generator = load_model()
    input_image = load_image(input_image_path)
    input_image = np.expand_dims(input_image, axis=0)

    prediction = generator(input_image, training=False)
    prediction = prediction[0].numpy()

    save_image(prediction, output_image_path)

# 이미지 업로드 및 예측을 처리하는 메인 라우트
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']

        if file:
            # 디렉토리 생성 및 업로드된 파일 저장
            upload_dir = 'static/uploads'
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            input_file_path = os.path.join(upload_dir, 'uploaded_image.jpg')
            file.save(input_file_path)

            # 예측 결과를 저장할 경로
            output_file_path = os.path.join(upload_dir, 'predicted_image.png')

            # 모델을 이용한 예측
            predict_and_save(input_file_path, output_file_path)

            # 결과 이미지를 클라이언트에 전달
            return render_template('index.html', image_path=output_file_path)

    # 이미지가 업로드되지 않았을 때 메인 페이지 렌더링
    return render_template('index.html', image_path=None)

# Flask 앱 실행
if __name__ == '__main__':
    app.run(debug=True)

