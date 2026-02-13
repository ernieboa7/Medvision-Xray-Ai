import tensorflow as tf


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


def build_model():
    #image_size = (224, 224)  # adjust if different
    image_size = (128, 128)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(image_size[0], image_size[1], 3)),

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

        tf.keras.layers.Dense(1, activation='softmax')
    ])

    return model
