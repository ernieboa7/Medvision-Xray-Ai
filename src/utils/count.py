import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf

Total_train_data = tf.io.gfile.glob(str('archive/chest_xray/train/*/*'))
print(len(Total_train_data))

count_train_Normal = tf.io.gfile.glob(str('archive/chest_xray/train/NORMAL/*'))
print(len(count_train_Normal))

count_train_Pneumonia = tf.io.gfile.glob(str('archive/chest_xray/train/PNEUMONIA/*'))
print(len(count_train_Pneumonia))




Total_val_data = tf.io.gfile.glob(str('archive/chest_xray/val/*/*'))
print(len(Total_val_data))

count_val_Normal = tf.io.gfile.glob(str('archive/chest_xray/val/NORMAL/*'))
print(len(count_val_Normal))

count_val_Pneumonia = tf.io.gfile.glob(str('archive/chest_xray/val/PNEUMONIA/*'))
print(len(count_val_Pneumonia))



Total_test_data = tf.io.gfile.glob(str('archive/chest_xray/test/*/*'))
print(len(Total_test_data))

count_test_Normal = tf.io.gfile.glob(str('archive/chest_xray/test/NORMAL/*'))
print(len(count_test_Normal))

count_test_Pneumonia = tf.io.gfile.glob(str('archive/chest_xray/test/PNEUMONIA/*'))
print(len(count_test_Pneumonia))