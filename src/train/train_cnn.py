# SETTING UP AND LOADING OF PACKAGES

import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split






autotune = tf.data.experimental.AUTOTUNE
batch_size = 20 
image_size = [200, 200]
epochs = 10


# LOADING DATASET, TESTING, EVALUATION OF PERFORMANCE AND HYPERPARAMETER TUNING.


file_x= tf.io.gfile.glob(str('resized_images_train/*/*'))
file_y= tf.io.gfile.glob(str('resized_images_val/*/*'))
mainfile =np.concatenate([file_x, file_y])
print(len(mainfile))



# Spliting the merge dataset into traing and validation dataset
train_file, val_file = train_test_split(mainfile, test_size=0.2)


COUNT_NORMAL = len([file for file in train_file if "NORMAL" in file])
print("Normal images count in training set: " + str(COUNT_NORMAL))

COUNT_PNEUMONIA = len([file for file in train_file if "PNEUMONIA" in file])
print("Pneumonia images count in training set: " + str(COUNT_PNEUMONIA))


# The above codes shows that we have a imbalance in our data. 
# This will be corrected later on.



train_list = tf.data.Dataset.from_tensor_slices(train_file)
val_list = tf.data.Dataset.from_tensor_slices(val_file)


''' View the first 5 data in both train and val data'''
for train in train_list.take(5):
    print('training:', train.numpy())
    
for val in val_list.take(5):
    print('validation:', val.numpy())
    
    
    
    
    
    
'''To check if our data is properly splitted into 0.8 training data 
and 0.2 validating data'''    
TRAIN_IMG_COUNT = tf.data.experimental.cardinality(train_list).numpy()
print("Training images count: " + str(TRAIN_IMG_COUNT))

VAL_IMG_COUNT = tf.data.experimental.cardinality(val_list).numpy()
print("Validating images count: " + str(VAL_IMG_COUNT))



# To determine the number of classes we have in the train data.
CLASS_NAMES = np.array([str(tf.strings.split(item, os.path.sep)[-1].numpy())[2:-1]
                        for item in mainfile])
print(CLASS_NAMES)   

'''
Top map each filename to the corresponding (image, label) pair. 
The following methods will help us do that.

There are two labels, rewrite the label so that 1 or True indicates pneumonia and 0 or False indicates normal.''' 


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == "PNEUMONIA"


'''
CNNs work better with smaller numbers so it is better to reduce the scale'''

def decode_image(image):
  # convert the compressed string to a 3D uint8 tensor
  image = tf.image.decode_jpeg(image, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  image = tf.image.convert_image_dtype(image, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(image, image_size)


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    image = tf.io.read_file(file_path)
    image = decode_image(image)
    return image, label

train_ds = train_list.map(process_path, num_parallel_calls=autotune)

val_ds = val_list.map(process_path, num_parallel_calls=autotune)


for image, label in train_ds.take(3):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())
    
    
    
    
    
# LOADING AND FORMATTING THE TEST DATA.  
test_list_ds = tf.data.Dataset.list_files(str('archive/chest_xray/test/*/*'))
test_image_count = tf.data.experimental.cardinality(test_list_ds).numpy()
test_ds = test_list_ds.map(process_path, num_parallel_calls=autotune)
test_ds1 = test_ds.batch(batch_size)

print(test_image_count)  


# VISUALIZING THE DATA

'''Buffered could be used for prefetching so we can yield data from disk without 
having I/O become blocking.'''  

def prepare_for_training(dataset, cache=True, shuffle_buffer_size=1000):
    '''This is a small dataset, only load it once, and keep it in memory.
    use `.cache(filename)` to cache preprocessing work for datasets that don't fit into memory'''
    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else:
            dataset = dataset.cache()

    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    '''prefetch: lets the dataset fetch batches in the background while the model
    is training.'''
    dataset = dataset.prefetch(buffer_size=autotune)

    return dataset

'''Call the batch iteration of the training data'''
train_ds = prepare_for_training(train_ds)
val_ds = prepare_for_training(val_ds)
test_ds = prepare_for_training(test_ds)

train_element_spec = train_ds.element_spec
val_element_spec = val_ds.element_spec
test_element_spec = test_ds.element_spec

if train_element_spec != val_element_spec:
    raise ValueError("Element not match.")

concat_ds = train_ds.concatenate(val_ds)


train_data, train_labels = next(iter(train_ds))
test_data, test_labels = next(iter(test_ds))
val_data, val_labels = next(iter(val_ds))
image_batch, label_batch = next(iter(concat_ds))


# Visualize data

def show_images(image_batch, label_batch):
    batch_size = len(image_batch)
    plt.figure(figsize=(12,10))
    for i in range(20):
        ax = plt.subplot(4,5,i+1)
        plt.imshow(image_batch[i])
        if label_batch[i]:
            plt.title("PNEUMONIA")
        else:
            plt.title("NORMAL")
        plt.axis("off")
    plt.show()
    
    
show_images(image_batch.numpy(), label_batch.numpy())    
