# Import thư viện cần thiết
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# Hyperparameters
INIT_LR = 1e-4
EPOCHS = 20 
BS = 32     # Batch size

DIRECTORY = r"data" 
CATEGORIES = ["with_mask", "without_mask"]

print("[INFO] Loading image ...")

data = []
labels = []

# PRE-PROCESSING
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        try:
            image = load_img(img_path, target_size=(224, 224)) # MobileNet requires input 224x224
            image = img_to_array(image)
            image = preprocess_input(image) # Scale pixel for MobileNet

            data.append(image)
            labels.append(category)
        except Exception as e:
            print(f"Lỗi ảnh {img_path}: {e}")

#  One-Hot Encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# train/test split (80% train, 20% test)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# Data Augmentation (Data enhancement: rotate, zoom, flip images...)
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# Modeling (Transfer learing)
# MobileNetV2 (include_top=False)
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# Building new head for face mask detection
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel) # avoid overfitting
headModel = Dense(2, activation="softmax")(headModel) # Output 2 class: Mask/No Mask

# Base + head
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze the baseModel layers (do not retrain this part).
for layer in baseModel.layers:
	layer.trainable = False

# Compile and train
print("[INFO] Compiling model ...")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training model ...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# Evaluate and save model
print("[INFO] Evaluating model...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

print("[INFO] Saving model...")
model.save("mask_detector.h5")
print("Done!")