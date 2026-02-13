import re
import os
import numpy as np
import pandas as pd
import tensorflow as tsf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from train_cnn1 import *
import seaborn as sns
from tensorflow.keras.metrics import Accuracy
import math 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy


# Create a convolutional block and dense model

def con_block(filters):
    blk_c = tsf.keras.Sequential([
        tsf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tsf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tsf.keras.layers.BatchNormalization(),
        tsf.keras.layers.MaxPool2D()
    ]
    )
    
    return blk_c


def den_block(units, dropout_rate):
    blk_d = tsf.keras.Sequential([
        tsf.keras.layers.Dense(units, activation='relu'),
        tsf.keras.layers.BatchNormalization(),
        tsf.keras.layers.Dropout(dropout_rate)
    ])
    
    return blk_d



def build_model():
    model = tsf.keras.Sequential([
        tsf.keras.Input(shape=(image_size[0], image_size[1], 3)),
        
        tsf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tsf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tsf.keras.layers.MaxPool2D(),
        
        con_block(32),
        con_block(64),
        
        con_block(128),
        tsf.keras.layers.Dropout(0.2),
        
        con_block(256),
        tsf.keras.layers.Dropout(0.2),
        
        tsf.keras.layers.Flatten(),
        den_block(512, 0.7),
        den_block(128, 0.5),
        den_block(64, 0.3),
        
        tsf.keras.layers.Dense(1, activation='softmax')
    ])
    
    return model
        
        
'''CORRECTING DATA INBALANCE IN THE DATASET'''
init_bias = np.log([count_pneumonia/count_normal])
print(init_bias)



wgt_for_0 = (1 / count_normal)*(count_train)/2.0 
wgt_for_1 = (1 / count_pneumonia)*(count_train)/2.0

class_wgt = {0: wgt_for_0, 1: wgt_for_1}

print('Weight for class 0: ', wgt_for_0)
print('Weight for class 1: ', wgt_for_1) 




optimizer = Adam(learning_rate=0.01)
loss = CategoricalCrossentropy()
#loss = CategoricalCrossentropy()
metrics = [Accuracy()]

''' TRAINING THE CNN MODEL '''  

with strategy.scope():
    model = build_model()

    METRICS = [
        'accuracy',
        tsf.keras.metrics.Precision(name='precision'),
        tsf.keras.metrics.Recall(name='recall')        
    ]
    
    model.compile(
        optimizer= optimizer,
        loss='categorical_crossentropy',
        metrics=METRICS
        )
    

history = model.fit(
    train_data,  
    train_labels,
    batch_size,
    epochs,
    validation_data=(val_data, val_labels) # Numerical data for validation
)



''' FINE TUNING THE MODEL'''
checkpoint_cb = tsf.keras.callbacks.ModelCheckpoint("xray_model.h5", save_best_only=True)
                                                    
early_stopping_cb = tsf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

def expo_decay(lr, sc):
    def expo_decay_fn(epoch):
        return lr * 0.1 **(epoch / sc)
    return expo_decay_fn

expo_decay_fn = expo_decay(0.01, 20)

lr_scheduler = tsf.keras.callbacks.LearningRateScheduler(expo_decay_fn)
  
  
history = model.fit(
    train_data,  # Numerical data for training
    train_labels,
    batch_size,
    epochs,
    validation_data= (val_data, val_labels), # Numerical data for validation
    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler]
)



loss, acc, prec, rec = model.evaluate(test_ds)




    
test_data, true_labels = next(iter(test_ds))
predictions = model.predict(test_data)

print(true_labels)

print(predictions)

# Assuming test_data contains true labels and predictions contains predicted labels

assigned_labels = (predictions > 0.5).astype(int)
#assigned_labels = np.argmax(predictions, axis=1)


con_matx = confusion_matrix(true_labels, assigned_labels)
tp = con_matx[1, 1]
tn = con_matx[0, 0]
fp = con_matx[0, 1]
fn = con_matx[1, 0]


print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)


accuracy = accuracy_score(true_labels, assigned_labels)
precision = precision_score(true_labels, assigned_labels)
recall = recall_score(true_labels, assigned_labels)
f1 = f1_score(true_labels, assigned_labels)


print("Confusion Matrix:")
print(con_matx)


print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("F1 Score:", f1)



print(classification_report(true_labels, assigned_labels))



plt.figure(figsize=(8, 6))
sns.heatmap(con_matx, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Create a bar chart to visualize the metrics.
metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
values = [accuracy, precision, recall, f1]

plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.title('Model Evaluation Metrics')
plt.ylim(0, 1)  
plt.show()
