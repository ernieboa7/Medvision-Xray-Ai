from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from data import *

# Define the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet50_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(2, activation='softmax')
])

# Compile the model
resnet50_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


from keras.callbacks import EarlyStopping

# Define the early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitoring the validation loss
    patience=10,          # Number of epochs with no improvement to wait before stopping
    verbose=1,           # Set to 1 to see progress messages
    restore_best_weights=True  # Restore the model's weights to the best epoch
)

epochs=2
# Train the ResNet50 model
history_resnet50 = resnet50_model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs,
    callbacks=[early_stopping],
    verbose=2
)



# Evaluate the ResNet50 model
resnet50_eval = resnet50_model.evaluate(test_data, verbose=2)