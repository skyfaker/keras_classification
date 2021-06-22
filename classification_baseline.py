'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

import os
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
test_data_dir = 'data/test'
num_train_samples = 80000
[os.listdir(path) for path in os.listdir(train_data_dir)]
num_validation_samples = 20000
max_epochs = 100
batch_size = 64

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=False)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=num_train_samples // batch_size,
    epochs=max_epochs,
    validation_data=validation_generator,
    validation_steps=num_validation_samples // batch_size)

model.save_weights('first_try.h5')

def predict_labels(model):
    """writes test image labels and predictions to csv"""
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        shuffle=False,
        class_mode=None)

    base_path = test_data_dir + "/test/"

    with open("prediction.csv", "w") as f:
        p_writer = csv.writer(f, delimiter=',', lineterminator='\n')
        for _, _, imgs in os.walk(base_path):
            for im in imgs:
                pic_id = im.split(".")[0]
                img = load_img(base_path + im)
                img = imresize(img, size=(img_height, img_width))
                test_x = img_to_array(img).reshape(3, img_height, img_width)
                test_x = test_x.reshape((1,) + test_x.shape)
                test_generator = test_datagen.flow(test_x,
                                                   batch_size=1,
                                                   shuffle=False)
                prediction = model.predict_generator(test_generator, 1)[0][0]
                p_writer.writerow([pic_id, prediction])
