import optuna
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os

# Load data generators once
data_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
                               fill_mode='wrap', shear_range=0.2)

train_generator = data_gen.flow_from_directory(
    r"C:\Users\kalai\Pictures\well_refined\trainfinal",
    target_size=(299, 299),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

test_generator = data_gen.flow_from_directory(
    r"C:\Users\kalai\Pictures\well_refined\testfinal",
    target_size=(299, 299),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

NO_CLASSES = len(train_generator.class_indices)


def objective(trial):
    tf.keras.backend.clear_session()

    # Trial parameters
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dense_units = trial.suggest_categorical("dense_units", [128, 256, 512])
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
    regularizer_val = trial.suggest_float("regularizer", 1e-5, 1e-2, log=True)
    conv_filter_size = trial.suggest_categorical("filter_size", [3, 5])
    optimizer_choice = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    unfreeze_start = trial.suggest_int("unfreeze_start", 200, 310)

    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.load_weights("C:/sih/models/inception_v3.h5")

    # Freeze layers
    for layer in base_model.layers[:unfreeze_start]:
        layer.trainable = False
    for layer in base_model.layers[unfreeze_start:]:
        layer.trainable = True

    model = tf.keras.Sequential([
        base_model,
        layers.Conv2D(64, (conv_filter_size, conv_filter_size), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(regularizer_val)),
        layers.AveragePooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(regularizer_val)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(dense_units, activation='relu',
                     kernel_regularizer=regularizers.l2(regularizer_val)),
        layers.Dropout(dropout_rate),
        layers.Dense(dense_units, activation='relu',
                     kernel_regularizer=regularizers.l2(regularizer_val)),
        layers.Dropout(dropout_rate),
        layers.Dense(NO_CLASSES, activation='softmax')
    ])

    # Optimizer selection
    if optimizer_choice == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(train_generator,
                        validation_data=test_generator,
                        epochs=20,
                        steps_per_epoch=len(train_generator),
                        validation_steps=len(test_generator),
                        callbacks=[early_stopping],
                        verbose=0)

    val_acc = max(history.history['val_accuracy'])
    return val_acc


# Run Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Best trial:")
print(study.best_trial)

# Optional: Save the best parameters
import json
with open("best_hyperparameters.json", "w") as f:
    json.dump(study.best_trial.params, f, indent=4)
