# Sample Python code for data loading and preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Import dataset
train_dir=('archive/chest_xray/train')
val_dir=('archive/chest_xray/val')
test_dir=('archive/chest_xray/test')



# Loading and preprocessing the dataset
dat_gen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split the dataset into train and validation sets
)

# Load the dataset
train_dat_gen = dat_gen.flow_from_directory(
    'train_dir',
    target_size=(150, 150),
    batch_size=64,
    class_mode='categorical',
    subset='training'
)

val_dat_gen = dat_gen.flow_from_directory(
    'val_dir',
    target_size=(150, 150),
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)

test_dat_gen = dat_gen.flow_from_directory(
    'test_dir',
    target_size=(150, 150),
    batch_size=64,
    class_mode='categorical'
)
