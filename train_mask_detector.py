from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# Hyperparameters
INIT_LR = 1e-4
EPOCHS = 20
BS = 32     # Batch size
DIRECTORY = r"data" 


print("[INFO] Initiating Data Generators...")

# Data augmentation for Train dataset
train_datagen = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest",
    preprocessing_function=preprocess_input, # Scaling for MobileNet
    validation_split=0.2 # 20% for validation set
)

# Train dataset (80%)
train_generator = train_datagen.flow_from_directory(
    DIRECTORY,
    target_size=(224, 224),
    batch_size=BS,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# Validation set (20%)
val_generator = train_datagen.flow_from_directory(
    DIRECTORY,
    target_size=(224, 224),
    batch_size=BS,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# Building model
print("[INFO] Building model...")
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

# Trainning model
print("[INFO] Compiling and Trainning model...")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BS,
    epochs=EPOCHS
)

# Saving model
print("[INFO] Saving model...")
model.save("mask_detector.h5")
print("Done!")