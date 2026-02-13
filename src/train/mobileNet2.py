import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import ResNet50
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


test_dir ='archive/chest_xray/test'


test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_accuracy = MobN_model.evaluate(test_generator)
print(f"Test_loss: {test_loss:.4f}")
print(f"Test_accuracy: {test_accuracy:.4f}")


prediction = MobN_model.predict(test_generator)
print(prediction)


assigned_labels = np.argmax(prediction, axis=1)
true_labels = test_generator.classes

print(true_labels)



precision, recall, f1_scor, accuracy = precision_recall_fscore_support(true_labels, assigned_labels, average='weighted')
accuracy = accuracy_score(true_labels, assigned_labels)
precision = precision_score(true_labels, assigned_labels)
recall = recall_score(true_labels, assigned_labels)
f1_scor = f1_score(true_labels, assigned_labels)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_scor)
print("Accuracy:", accuracy)






con_matx = confusion_matrix(true_labels, assigned_labels)
tp = con_matx[1, 1]
tn = con_matx[0, 0]
fp = con_matx[0, 1]
fn = con_matx[1, 0]

print("Confusion Matrix:")
print(con_matx)

print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)

class_rep = classification_report(true_labels, assigned_labels, target_names=test_generator.class_indices)
print("Classification Report:")
print(class_rep)

# You can create a heatmap of the confusion matrix to visualize classification results.


plt.figure(figsize=(8, 6))
sns.heatmap(con_matx, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# Create a bar chart to visualize the metrics.
metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
values = [accuracy, precision, recall, f1_scor]

plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.title('Model Evaluation Metrics')
plt.ylim(0, 1)  
plt.show()
