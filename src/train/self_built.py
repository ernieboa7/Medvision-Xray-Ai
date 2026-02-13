import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from pathlib import Path

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
image_size = (128, 128)
batch_size = 32
epochs = 10

DATASET_PATH = Path("data/archive/chest_xray/chest_xray")

train_dir = DATASET_PATH / "train"
test_dir = DATASET_PATH / "test"
val_dir = DATASET_PATH / "val"

# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="binary"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="binary"
)


# -------------------------------------------------
# MODEL BLOCKS
# -------------------------------------------------
def con_block(filters):
    return tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ])


def den_block(units, dropout_rate):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])


# -------------------------------------------------
# BUILD MODEL
# -------------------------------------------------
def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(image_size[0], image_size[1], 3)),

        tf.keras.layers.Rescaling(1./255),

        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(),

        con_block(32),
        con_block(64),

        con_block(128),
        tf.keras.layers.Dropout(0.2),

        con_block(256),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),

        den_block(512, 0.7),
        den_block(128, 0.5),
        den_block(64, 0.3),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model


# -------------------------------------------------
# COMPILE
# -------------------------------------------------
model = build_model()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)

model.summary()

# -------------------------------------------------
# CALLBACKS
# -------------------------------------------------
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "xray_model.h5",
    save_best_only=True
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True
)

# -------------------------------------------------
# TRAIN
# -------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# -------------------------------------------------
# TEST
# -------------------------------------------------
loss, acc, prec, rec = model.evaluate(test_ds)

print("\nFinal Results")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
