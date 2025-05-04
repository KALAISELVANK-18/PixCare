from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import io
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2


app = FastAPI()


@app.post("/upload")
async def upload(image: UploadFile = File(...)):

    # Read the image file as bytes
    contents = await image.read()

    # Convert bytes to a PIL Image
    pil_image = Image.open(io.BytesIO(contents))

    # Convert PIL Image to a NumPy array
    image_array = np.array(pil_image)

    # Save the NumPy array as an image file
    cv2.imwrite("solution.jpg", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    testmodel=load_model("newmodel.h5", compile=False)
    testlabel=open("testlabel.txt", "r").readlines()
    model = load_model("main3_model.h5", compile=False)

    # Load the labels
    class_names = open("cancerlabel.txt", "r").readlines()

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open("solution.jpg").convert("RGB")
    image_thermal = cv2.imread('solution.jpg')
    # resizing the image to be at least 224x224 and then cropping from the center
    cv2.resize(image_thermal,(224,224))
    size = (224, 224)
    gray_image = cv2.cvtColor(image_thermal, cv2.COLOR_BGR2GRAY)

    # Apply a color mapping to simulate thermal imaging
    # Here, we map darker pixels (cooler areas) to blue and brighter pixels (warmer areas) to red
    thermal_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
    cv2.imwrite('thermal.jpg',thermal_image)
    image_u8int_list = thermal_image.tolist()

    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = testmodel.predict(data)
    index = np.argmax(prediction)
    class_name = testlabel[index]
    confidence_score = prediction[0][index]
    print(class_name,confidence_score)
    if((class_name!="1 normal" and confidence_score>=0.51)):
        dprediction = model.predict(data)
        dindex = np.argmax(dprediction)
        print(dindex)
        listofimg=list(zip(dprediction[0],class_names))
        print(listofimg)
        listofimg.sort(key=lambda x:x[0],reverse=True)
        print(listofimg)
        dclass_name_score = [[listofimg[0][1],str(listofimg[0][0])],[listofimg[1][1],str(listofimg[1][0])],[listofimg[2][1],str(listofimg[2][0])]]
        dconfidence_score = dprediction[0][dindex]

        class_name = class_names[dindex]
        print(class_name,dconfidence_score)
        return JSONResponse({'result':dclass_name_score,'diseases':True,"thermal":str(image_u8int_list)})
    else:
        return JSONResponse({'result':["No diseases found"],'diseases':False})



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)