from keras.models import load_model  # TensorFlow is required for Keras to work
import tensorflow as tf
from sklearn.metrics import classification_report,confusion_matrix
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
                                    fill_mode='wrap',
                                   # Randomly shift images vertically by up to 20% of the height
                                   shear_range=0.2,  # Randomly apply shearing transformations
                                   )


test_generator= train_datagen.flow_from_directory(r"C:\Users\kalai\Pictures\refined\trainfinal",target_size=(299,299),color_mode='rgb',batch_size=32,class_mode='categorical',shuffle=True)

# Load the model
model = load_model(r"C:\sih\new model\disease2_model.h5", compile=True)

scores = model.evaluate(test_generator)
print(scores)
print(f"Test Accuracy..........: {scores[1]*100}")

import numpy as np
test_X, test_Y = next(test_generator)
for i in range(int(len(test_generator)/1)-1): #1st batch is alread fetched before the for loop for i in range(int(len(test_set)/batch_size)-1)
    img, label = next(test_generator)
    test_X = np.append(test_X, img, axis=0 )
    test_Y = np.append(test_Y, label, axis=0)

print(len(test_generator),"  ", type(test_generator))

print(test_X.shape)
print("...........")
print(test_Y.shape)
print("...........")


pred_test_Y = model.predict(test_X, batch_size = 1, verbose = True)

print(pred_test_Y.shape)

test_Y_cat = np.argmax(test_Y, axis=1)
pred_test_Y_cat = np.argmax(pred_test_Y,axis=1)
print(pred_test_Y_cat,test_Y_cat)



mcm=confusion_matrix(test_Y_cat, pred_test_Y_cat, labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22])


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


x_axis_labels = [
    "Acne and Rosacea",
    "Actinic Keratosis Basal Cell Carcinoma",
    "Atopic Dermatitis",
    "Bullous Disease",
    "Cellulitis Impetigo and other Bacterial",
    "Eczema",
    "Exanthems and Drug Eruptions",
    "Hair Loss,Alopecia and others",
    "Herpes HPV and other STDs",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other connective Tissue",
    "Melanoma Skin Cancer",
    "Nail Fungus and other Nail diseases",
    "Poison Ivy",
    "Psoriasis,Lichen Planus and related",
    "Scabies Lyme and other Infestations",
    "Seberrheic keratoses and other benign tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis",
    "Warts molluscum and other viral Infections"
  ]
y_axis_labels = [
    "Acne and Rosacea",
    "Actinic Keratosis Basal Cell Carcinoma",
    "Atopic Dermatitis",
    "Bullous Disease",
    "Cellulitis Impetigo and other Bacterial",
    "Eczema",
    "Exanthems and Drug Eruptions",
    "Hair Loss,Alopecia and others",
    "Herpes HPV and other STDs",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other connective Tissue",
    "Melanoma Skin Cancer",
    "Nail Fungus and other Nail diseases",
    "Poison Ivy",
    "Psoriasis,Lichen Planus and related",
    "Scabies Lyme and other Infestations",
    "Seberrheic keratoses and other benign tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis",
    "Warts molluscum and other viral Infections"
  ]

df_cm = pd.DataFrame(mcm, range(23), range(23))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1) # for label size
sn.heatmap(df_cm, xticklabels=x_axis_labels, yticklabels=y_axis_labels, annot=True, annot_kws={"size": 14},cmap='Reds', fmt='g') # font size

plt.show()



