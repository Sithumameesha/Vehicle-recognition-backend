from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.optimizers import Adam
import numpy as np
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

custom_model = load_model('vehicle_recognition_model.h5')

dataset_path = 'vehicle'
train_data_dir = os.path.join(dataset_path, 'train')

class_labels = sorted(os.listdir(train_data_dir))

custom_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


def process_predictions(predictions, class_labels):
    if len(predictions.shape) == 2 and predictions.shape[0] == 1:
        predictions = predictions[0]

    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = predictions[predicted_class_index]

    return predicted_class_label, confidence


def process_image(file):
    try:
        img_bytes = io.BytesIO(file.file.read())

        img = image.load_img(img_bytes, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = custom_model.predict(img_array)
        predicted_class, confidence = process_predictions(predictions, class_labels)

        return {'vehicle_name': predicted_class, 'confidence': float(confidence)}

    except Exception as e:
        print(f'Error processing image {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/uploadFile/")
async def create_upload_file(file: UploadFile = File(...)):
    result = process_image(file)
    return JSONResponse(content=result)
