from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from data import *

# Define the MobileNet model
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(2, activation='sigmoid')(x)

mobile_net_model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
mobile_net_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 2


from keras.callbacks import EarlyStopping


# Train the MobileNet model
history_mobile_net = mobile_net_model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs,
    verbose=2
)


# Evaluate the MobileNet model
mobile_net_eval = mobile_net_model.evaluate(test_data, verbose=2)