from keras.callbacks import EarlyStopping
from keras.models import load_model
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import GlobalAveragePooling2D,Conv2D,AveragePooling2D,Dropout
from keras.layers import Dense,Input
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import asyncio

from sklearn.svm import SVC

import tensorflow as tf



test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
    fill_mode='wrap',
    shear_range=0.2
)

test_path = r"C:\Users\kalai\Pictures\well_refined\testfinal"
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(299, 299),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)


train_generator =test_datagen.flow_from_directory(r"C:\Users\kalai\Pictures\well_refined\trainfinal",
                                                  target_size=(299,299),
                                                  color_mode='rgb',
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  shuffle=True)



NO_CLASSES = len(train_generator.class_indices.values())


base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False,input_shape=(299,299,3))

# Load weights into the model
base_model.load_weights("C:\sih\models\inception_v3.h5")

base_model.trainable=False
co=0
# don't train the first 19 layers - 0..18
for layer in base_model.layers[260:]:
    print(co)
    layer.trainable = True
    co+=1




models=tf.keras.Sequential(
    [
        base_model,
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.AveragePooling2D((2, 2),),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(3, activation='softmax')
    ]
)


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

models.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

history=models.fit(train_generator,validation_steps=len(test_generator),validation_data=test_generator,steps_per_epoch=len(train_generator),batch_size = 32,epochs = 16,
                  callbacks=[early_stopping])
models.save('disease4_' + 'model.h5')
models.save('disease4_' + 'model.keras')
models.evaluate(test_generator)




import matplotlib.pyplot as plt


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy for the diseases(17000 images")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


plot_hist(history)

