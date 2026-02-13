import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score





train_dir ='archive/chest_xray/train'



base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))



res = base_model.output
res = GlobalAveragePooling2D()(res)
res = Dense(units=2, activation='relu')(res)
predictions = Dense(units=2,activation='softmax')(res)


model = Model(inputs=base_model.input, outputs=predictions)

optimizer = Adam(learning_rate=0.01)
loss = CategoricalCrossentropy()
metrics = [Accuracy()]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



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


val_dir = 'resized_images_val'
validation_generator = train_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)



num_epochs = 10

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)


test_dir ='archive/chest_xray/test'



test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")


predictions2 = model.predict(test_generator)

#predicted_labels = (predictions2 > 0.5).astype(int)

predicted_labels = np.argmax(predictions2, axis=1)

true_labels2 = test_generator.classes



precision, recall, f1_scor, accuracy = precision_recall_fscore_support(true_labels2, predicted_labels, average='weighted')
accuracy = accuracy_score(true_labels2, predicted_labels)
precision = precision_score(true_labels2, predicted_labels)
recall = recall_score(true_labels2, predicted_labels)
f1_scor = f1_score(true_labels2, predicted_labels)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_scor)
print("Accuracy:", accuracy)






conf_matrix = confusion_matrix(true_labels2, predicted_labels)
tp = conf_matrix[1, 1]
tn = conf_matrix[0, 0]
fp = conf_matrix[0, 1]
fn = conf_matrix[1, 0]

print("Confusion Matrix:")
print(conf_matrix)

print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)

classification_rep = classification_report(true_labels2, predicted_labels, target_names=test_generator.class_indices)
print("Classification Report:")
print(classification_rep)

# You can create a heatmap of the confusion matrix to visualize classification results.


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# Create a bar chart to visualize the metrics.
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1_scor]

plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.title('Model Evaluation Metrics')
plt.ylim(0, 1)  
plt.show()
