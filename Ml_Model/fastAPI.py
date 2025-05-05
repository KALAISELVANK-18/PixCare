from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
import io
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2

app = FastAPI()

def preprocess_image(image_path, size):
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    normalized_image = (image_array / 127.5) - 1
    return normalized_image

def generate_thermal_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    cv2.imwrite('thermal.jpg', thermal)
    return thermal.tolist()

@app.post("/upload")
async def upload(image: UploadFile = File(...), red_difference: float = Form(...)):
    contents = await image.read()
    with open("solution.jpg", "wb") as f:
        f.write(contents)

    thermal_data = generate_thermal_image("solution.jpg")

    test_model = load_model("categorization_model.h5", compile=False)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = preprocess_image("solution.jpg", (224, 224))
    prediction = test_model.predict(data)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]

    if index == 0:
        return JSONResponse({'result': ["No diseases found"], 'diseases': False})

    skin_model = load_model("diseases_verification_model.h5", compile=False)
    prediction = skin_model.predict(data)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]

    if index == 1 and confidence_score >= 0.4:
        return JSONResponse({'result': ["No diseases found"], 'diseases': False})

    model = load_model("disease_prediction_model.h5", compile=False)
    label_file = "cancer_label.txt" if (red_difference * 0.0625) / 3 >= 0.7 else "dieases_label.txt"
    class_names = open(label_file, "r").readlines()

    data = np.ndarray(shape=(1, 299, 299, 3), dtype=np.float32)
    data[0] = preprocess_image("solution.jpg", (299, 299))
    dprediction = model.predict(data)
    dindex = np.argmax(dprediction)

    listofimg = [[pred, name] for pred, name in zip(dprediction[0], class_names)]
    listofimg.sort(key=lambda x: x[0], reverse=True)

    if label_file == "dieases_label.txt":
        for i in range(3):
            listofimg[i][0] *= 0.60

    top3 = [[listofimg[i][1], str(listofimg[i][0])] for i in range(3)]
    return JSONResponse({'result': top3, 'diseases': True, "thermal": str(thermal_data)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
