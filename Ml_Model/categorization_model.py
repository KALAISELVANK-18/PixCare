from keras.layers import GlobalAveragePooling2D,MaxPooling2D,Conv2D,GlobalMaxPooling2D
from keras.layers import Dense,Input
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model,Sequential
from keras.applications import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.efficientnet_v2 import EfficientNetV2L
from keras.applications.efficientnet import EfficientNetB2
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.optimizers import Adam,SGD,RMSprop
from keras.callbacks import EarlyStopping

from keras import layers
from optuna import trial
import optuna

import tensorflow as tf


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=40,  # Randomly rotate images by up to 40 degrees
                                   width_shift_range=0.2,
                                   # Randomly shift images horizontally by up to 20% of the width
                                   height_shift_range=0.2,
                                   # Randomly shift images vertically by up to 20% of the height
                                   shear_range=0.2,  # Randomly apply shearing transformations
                                   zoom_range=0.2,  # Randomly zoom in on images
                                   horizontal_flip=True,  # Randomly flip images horizontally
                                   fill_mode='nearest'
                                   )

train_generator =train_datagen.flow_from_directory(r"C:\Users\kalai\Pictures\characterize\train",target_size=(224,224),color_mode='rgb',batch_size=16,class_mode='categorical',shuffle=True)

test_generator= train_datagen.flow_from_directory(r"C:\Users\kalai\Pictures\characterize\test",target_size=(224,224),color_mode='rgb',batch_size=16,class_mode='categorical',shuffle=True)





NO_CLASSES = len(train_generator.class_indices.values())


base_model = InceptionV3(include_top=False,input_shape=(224,224,3),weights="imagenet") #basemodel


base_model.trainable=False

# don't train the first 19 layers - 0..18
for layer in base_model.layers[275:]:
    layer.trainable = True




x=base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dense(256, activation='relu')(x)

# final layer with softmax activation
output = Dense(NO_CLASSES, activation='softmax')(x)


model = Model(inputs =base_model.inputs, outputs = output)
# model.build((224, 224, 3))
# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

model.compile(optimizer=Adam(learning_rate=0.00001),loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(train_generator,validation_steps=len(test_generator),validation_data=test_generator,steps_per_epoch=len(train_generator),batch_size = 16,epochs = 5,
                  )
model.save('categorization' + 'model.h5')
# model.save('new' + 'model.keras')
model.evaluate(test_generator)




import matplotlib.pyplot as plt


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


plot_hist(history)

