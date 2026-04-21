import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 1. Load Data
# Increased image size slightly for better feature extraction (128x128)
IMG_SIZE = (128, 128)
BATCH_SIZE = 8

train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Crucial: Identify which folder is 0 and which is 1
class_names = train_ds.class_names
print(f"Detected Classes: {class_names}")
# Standard: 0 = Dry_Road, 1 = Waterlogging (Alphabetical order)

# 2. Build the Robust "Smart" Brain
# Updated Section 2 for train_water.py
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    
    # Layer 1 + Normalization
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(), # This forces the model to learn faster
    layers.MaxPooling2D((2, 2)),
    
    # Layer 2 + Normalization
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Layer 3 + Normalization
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'), # Increased neurons
    layers.Dropout(0.5), 
    layers.Dense(1, activation='sigmoid')
])
# 3. Compile with a stable optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# 4. Train the model
print("\nTraining started... focusing on generalization...")
# Using 40-50 epochs is fine for 800 images
model.fit(train_ds, validation_data=val_ds, epochs=40)

# 5. Save in the NEW Native Keras format (Removes the Warning)
model.save('water_model.keras') 
print("\nSuccess! Model saved as 'water_model.keras' in native format.")

# Verification Step
print(f"\nIMPORTANT: In your detection script, remember:")
print(f"If output is close to 0, it is: {class_names[0]}")
print(f"If output is close to 1, it is: {class_names[1]}")
