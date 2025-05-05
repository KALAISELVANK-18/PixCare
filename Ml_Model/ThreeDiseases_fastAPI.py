from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
import io
import json
import numpy as np
import cv2
from PIL import Image, ImageOps
from keras.models import load_model

app = FastAPI()

@app.post("/upload")
async def upload(image: UploadFile = File(...), Data: str = Form(...)):
    # Parse form data
    Datas = json.loads(Data)
    
    # Read and save the uploaded image
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_array = np.array(pil_image)
    cv2.imwrite("solution.jpg", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    
    # Preprocess image for the first model
    resized_image = ImageOps.fit(pil_image, (224, 224), Image.Resampling.LANCZOS)
    normalized_array = ((np.asarray(resized_image).astype(np.float32)) / 127.5) - 1
    input_data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    input_data[0] = normalized_array

    # Load and predict with categorization model
    cat_model = load_model("categorizationmodel.h5", compile=False)
    prediction = cat_model.predict(input_data)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]

    # If no disease found
    if index == 0:
        return JSONResponse({'result': ["No diseases found"], 'diseases': False})
    
    # Load and predict with skin classification model
    skin_model = load_model("three_diseases_model.h5", compile=False)
    prediction = skin_model.predict(input_data)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]

    # If skin is not affected enough
    if index == 1 and confidence_score >= 0.4:
        return JSONResponse({'result': ["No diseases found"], 'diseases': False})

    # Load and prepare for disease classification
    image_thermal = cv2.imread("solution.jpg")
    gray_image = cv2.cvtColor(image_thermal, cv2.COLOR_BGR2GRAY)
    thermal_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
    cv2.imwrite("thermal.jpg", thermal_image)
    image_u8int_list = thermal_image.tolist()

    # Resize and normalize for disease model
    resized_img = ImageOps.fit(pil_image, (299, 299), Image.Resampling.LANCZOS)
    normalized_array = ((np.asarray(resized_img).astype(np.float32)) / 127.5) - 1
    input_data_disease = np.ndarray(shape=(1, 299, 299, 3), dtype=np.float32)
    input_data_disease[0] = normalized_array

    # Predict with disease model
    disease_model = load_model("diseases_model.h5", compile=False)
    dprediction = disease_model.predict(input_data_disease)
    dindex = np.argmax(dprediction)
    class_names = open("three_diseases_label.txt", "r").readlines()
    dconfidence_score = dprediction[0][dindex]

    # Prepare top-3 class results with adjusted scores
    class_scores = [[pred * 0.60, name.strip()] for pred, name in zip(dprediction[0], class_names)]
    class_scores.sort(key=lambda x: x[0], reverse=True)
    top_3 = [[name, str(score)] for score, name in class_scores[:3]]

    return JSONResponse({
        'result': top_3,
        'diseases': True,
        'thermal': str(image_u8int_list)
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
