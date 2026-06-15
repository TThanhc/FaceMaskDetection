from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

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

# No data augmentation for Validation dataset (only preprocessing)
val_datagen = ImageDataGenerator(
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
    seed=42,
    shuffle=True
)

# Validation set (20%)
val_generator = val_datagen.flow_from_directory(
    DIRECTORY,
    target_size=(224, 224),
    batch_size=BS,
    class_mode="categorical",
    subset="validation",
    seed=42,
    shuffle=False
)

print(f"[INFO] Class Indices Mapping: {train_generator.class_indices}")

# Building model
print("[INFO] Building model...")
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = GlobalAveragePooling2D()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

# Training model
print("[INFO] Compiling and Training model...")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Saving model
print("[INFO] Saving model...")
model.save("mask_detector.h5")
print("Done!")