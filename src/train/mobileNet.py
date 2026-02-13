base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
nas = base_model.output
nas = GlobalAveragePooling2D()(nas)
nas = Dense(units=2, activation='relu')(nas)
predictions = Dense(units=2,activation='softmax')(nas)


MobN_model = Model(inputs=base_model.input, outputs=predictions)

optimizer = Adam(learning_rate=0.01)
loss = CategoricalCrossentropy()
metrics = [Accuracy()]

MobN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


image_size = (150, 150)
batch_size = 64

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'archive/chest_xray/val',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


num_epochs = 10

MobN_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)


test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
