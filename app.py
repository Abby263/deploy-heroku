import os
import uuid

import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
from tensorflow.keras.models import load_model

# Flask utils
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_resnet.h5')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
model = None


def get_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH, compile=False)
        print("*** Model Loaded ***")
    return model


def model_predict(img_path, loaded_model):
    image_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (224, 224))
    x = tf.expand_dims(tf.cast(img, tf.float32), axis=0)
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = loaded_model.predict(x, verbose=0)
    return preds


def format_prediction(preds):
    try:
        pred_class = decode_predictions(preds, top=1)
        return str(pred_class[0][0][1])
    except Exception as exc:
        print(f'*** decode_predictions failed: {exc} ***')
        return f'class_{int(preds.argmax(axis=-1)[0])}'


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files.get('file')
        if f is None or f.filename == '':
            return 'No file uploaded', 400

        _, extension = os.path.splitext(secure_filename(f.filename))
        extension = extension.lower()
        if extension not in ALLOWED_EXTENSIONS:
            return 'Unsupported file type', 400

        # Save the file to ./uploads
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file_path = os.path.join(UPLOAD_FOLDER, f'{uuid.uuid4().hex}{extension}')
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, get_model())
        print('*** preds ***')

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        print('*** pred class ***')
        result = format_prediction(preds)               # Convert to string
        print('*** result ***')
        return result
    elif request.method == 'GET':
        return render_template('index.html')
    return None


if __name__ == '__main__':
    print('*** App Started ***')
    app.run(debug=True)
