import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, layers, models

# Paths to the dataset
dataset_path = "Dataset.csv"

# Define your parameters
batch_size = 32
img_height, img_width = 224, 224  # Example size

# Create an ImageDataGenerator object to load images directly
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)  # rescaling pixel values

# Load the data directly using flow_from_directory method
train_data_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # or 'binary' depending on your dataset
    subset='training',  # for training data
    shuffle=True
)

val_data_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',  # for validation data
    shuffle=True
)

# Build a simple CNN model
model = models.Sequential([
    Input(shape=(img_height, img_width, 3)),  # Using Input layer instead of input_shape
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data_gen.num_classes, activation='softmax')  # Number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data_gen,
    validation_data=val_data_gen,
    epochs=10,
    verbose=1  # This will print accuracy during training
)

# Print the final accuracy after training
print(f"Training accuracy: {history.history['accuracy'][-1]:.2f}")
print(f"Validation accuracy: {history.history['val_accuracy'][-1]:.2f}")

# Save the model
model_save_path = "saved_model/my_model"   #in.h5
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# If you want to evaluate the model on a separate test dataset (if available), you can use:
# test_data_gen = datagen.flow_from_directory(test_path, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')
# test_loss, test_accuracy = model.evaluate(test_data_gen)
# print(f"Test accuracy: {test_accuracy:.2f}")
